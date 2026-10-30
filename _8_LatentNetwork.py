"""Fit session neural activity or retrain a context 1 network to reverse its readout."""

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SAVE_FIGURE_PATH = 'figure/latent_network/'
TRAINING_DATA_PATH = 'data/training/recurrent/'
SAVE_LATENT_PATH = TRAINING_DATA_PATH
CONDITIONS = ['P1', 'A1']
DEFAULT_DATASET_ID = '0'
DEFAULT_N_INPUT_TYPES = 2
DEFAULT_N_INPUT_COMPONENTS = 1
DEFAULT_INPUT_NOISE_STD = 0.1
DEFAULT_N_INITIAL_TIMEPOINTS = 2
DEFAULT_DEVICE = 'cpu'
DEFAULT_RECURRENT_TYPE = 'fullconn'
DT = 0.1
CHECKPOINT_VERSION = 4
READOUT_CHECKPOINT_VERSION = 6
READOUT_WINDOW = (12, 16)
DEFAULT_MIN_CD2_NORM_RATIO = 1e-6
CONTEXT1_REFERENCE_KEYS = (
    'rP1', 'rA1', 'cd1', 'norm_cd1', 'r_half1',
    'delta_rP1', 'delta_rA1', 'zP1', 'zA1',
)


def _validate_scalar(value, name, allow_zero=False):
    if (
        isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        bound = 'nonnegative' if allow_zero else 'positive'
        raise ValueError(f'{name} must be finite and {bound}.')
    return float(value)


def readout_retrained_checkpoint_path(source_checkpoint, save_path=None):
    """Keep the new SVD readout format separate from earlier retrained files."""
    source = Path(source_checkpoint)
    root = source.parent if save_path is None else Path(save_path)
    return str(root / f'{source.stem}_retrained_readout_svd.pt')


def _validate_readout_session(session):
    if set(session.conditions) != {'P1', 'A1'}:
        raise ValueError('Readout reversal requires a context 1 model with P1/A1 conditions.')
    if session.n_time < READOUT_WINDOW[1]:
        raise ValueError('Readout reversal requires at least 16 time bins for activity[12:16].')


@torch.no_grad()
def compute_context1_reference(session, initial_activity=None):
    """Compute fixed P1/A1 references from one fresh-noise model simulation."""
    _validate_readout_session(session)
    activity = session(initial_activity=initial_activity)
    start, stop = READOUT_WINDOW
    rP1 = activity['P1'][start:stop].mean(dim=0)
    rA1 = activity['A1'][start:stop].mean(dim=0)
    cd1 = rP1 - rA1
    norm_cd1 = cd1.norm()
    r_half1 = (rP1 + rA1) / 2
    # The context 1 norm is assumed nonzero; only context 2 is guarded.
    delta_rP1 = (rP1 - r_half1) / norm_cd1
    delta_rA1 = (rA1 - r_half1) / norm_cd1
    return {
        'rP1': rP1, 'rA1': rA1, 'cd1': cd1, 'norm_cd1': norm_cd1,
        'r_half1': r_half1, 'delta_rP1': delta_rP1, 'delta_rA1': delta_rA1,
        'zP1': torch.dot(cd1, delta_rP1), 'zA1': torch.dot(cd1, delta_rA1),
    }


def readout_reversal_loss(session, readout, lambda_readout=1.0, lambda_coding=1.0):
    """Reversed fixed targets plus squared, dimensionless constraint residuals."""
    norm_cd1_sq = session.reference_cd1.square().sum()
    loss_reverse = (
        (session.reference_zP1 - readout['zA2']).square()
        + (session.reference_zA1 - readout['zP2']).square()
    )
    readout_residual = (
        session.alpha.square()
        + readout['u_ortho'].square().sum() / norm_cd1_sq - 1
    )
    # Use the actual cd2, not the protected readout denominator.
    coding_residual = readout['cd2'].square().sum() / norm_cd1_sq - 1
    loss_readout_constraint = readout_residual.square()
    loss_coding_constraint = coding_residual.square()
    return {
        'loss': (loss_reverse + lambda_readout * loss_readout_constraint
                 + lambda_coding * loss_coding_constraint),
        'loss_reverse': loss_reverse,
        'loss_readout_constraint': loss_readout_constraint,
        'loss_coding_constraint': loss_coding_constraint,
        'readout_residual': readout_residual,
        'coding_residual': coding_residual,
    }


def validate_initial_timepoints(n_initial_timepoints, n_time):
    """None is reserved for restoring legacy zero-state networks."""
    if n_initial_timepoints is None:
        return None
    if (
        isinstance(n_initial_timepoints, (bool, np.bool_))
        or not isinstance(n_initial_timepoints, (int, np.integer))
        or not 1 <= n_initial_timepoints <= n_time
    ):
        raise ValueError(f'n_initial_timepoints must be an integer between 1 and {n_time}.')
    return int(n_initial_timepoints)


def validate_recurrent_config(recurrent_type, recurrent_rank, n_units=None):
    """Validate the recurrent type and return its rank (None for fullconn)."""
    if recurrent_type not in ('fullconn', 'lowrank'):
        raise ValueError("recurrent_type must be 'fullconn' or 'lowrank'.")
    if recurrent_type == 'fullconn':
        return None
    if (
        isinstance(recurrent_rank, (bool, np.bool_))
        or not isinstance(recurrent_rank, (int, np.integer))
        or recurrent_rank < 1
    ):
        raise ValueError("recurrent_rank must be a positive integer for 'lowrank'.")
    recurrent_rank = int(recurrent_rank)
    if n_units is not None and recurrent_rank > n_units:
        raise ValueError(
            f'recurrent_rank={recurrent_rank} exceeds the session size of {n_units} neurons.'
        )
    return recurrent_rank


def latent_network_checkpoint_path(
    dataset_id=DEFAULT_DATASET_ID,
    n_input_types=DEFAULT_N_INPUT_TYPES,
    n_input_components=DEFAULT_N_INPUT_COMPONENTS,
    input_noise_std=DEFAULT_INPUT_NOISE_STD,
    save_path=None,
    recurrent_type=DEFAULT_RECURRENT_TYPE,
    recurrent_rank=None,
    training_path=TRAINING_DATA_PATH,
):
    """Store each recurrent type/rank in its own folder and filename."""
    recurrent_rank = validate_recurrent_config(recurrent_type, recurrent_rank)
    rank_suffix = f'_Rank{recurrent_rank}' if recurrent_type == 'lowrank' else ''
    filename = (
        f'latent_network_Dataset{dataset_id}'
        f'_In{n_input_components}'
        f'{rank_suffix}_{recurrent_type}.pt'
    )
    root = Path(training_path if save_path is None else save_path)
    return str(root / recurrent_type / filename)


def resolve_device(device=DEFAULT_DEVICE):
    """Choose CUDA or MPS for 'gpu'; PyTorch handles explicit device errors."""
    requested = str(device).strip().lower()
    if requested == 'gpu':
        if torch.cuda.is_available():
            requested = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            requested = 'mps'
        else:
            raise ValueError('GPU requested, but neither CUDA nor MPS is available.')
    return torch.device(requested)


# Each session has its own size, recurrent parameters, and condition-specific inputs.
class SessionRNN(nn.Module):
    """Euler simulation with dense/low-rank recurrence trained in the middle third."""

    def __init__(
        self,
        n_units,
        n_time,
        conditions=CONDITIONS,
        n_input_types=DEFAULT_N_INPUT_TYPES,
        n_input_components=DEFAULT_N_INPUT_COMPONENTS,
        input_noise_std=DEFAULT_INPUT_NOISE_STD,
        dt=DT,
        input_masks=None,
        recurrent_type=DEFAULT_RECURRENT_TYPE,
        recurrent_rank=None,
        n_initial_timepoints=DEFAULT_N_INITIAL_TIMEPOINTS,
    ):
        super().__init__()
        self.n_units = int(n_units)
        self.n_time = int(n_time)
        self.conditions = tuple(conditions)
        self.n_conditions = len(self.conditions)
        self.n_input_types = int(n_input_types)
        self.n_input_components = int(n_input_components)
        self.input_noise_std = float(input_noise_std)
        self.dt = float(dt)
        if not self.conditions or len(set(self.conditions)) != self.n_conditions:
            raise ValueError('conditions must be nonempty and contain no duplicates.')
        if self.n_input_types < 1 or self.n_input_components < 1:
            raise ValueError('n_input_types and n_input_components must be positive.')
        if not np.isfinite(self.input_noise_std) or self.input_noise_std < 0:
            raise ValueError('input_noise_std must be finite and nonnegative.')
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('dt must be finite and positive.')
        self.recurrent_type = recurrent_type
        self.recurrent_rank = validate_recurrent_config(
            recurrent_type, recurrent_rank, self.n_units,
        )

        if input_masks is None:
            if self.n_input_types != 2:
                raise ValueError(
                    'The default sensory/response masks need n_input_types=2. '
                    'Supply input_masks for other input types.'
                )
            input_masks = torch.zeros(self.n_input_types, self.n_time)
            input_masks[0, :self.n_time // 3] = 1.0
            input_masks[1, 2 * self.n_time // 3:] = 1.0
        else:
            input_masks = torch.as_tensor(input_masks, dtype=torch.float32).clone()
        if tuple(input_masks.shape) != (self.n_input_types, self.n_time):
            raise ValueError('input_masks must have shape (n_input_types, n_time).')
        if not torch.isfinite(input_masks).all() or not torch.all(
            (input_masks == 0) | (input_masks == 1)
        ):
            raise ValueError('input_masks must contain only zeros and ones.')
        self.register_buffer('input_masks', input_masks)

        # All recurrent parameters are trainable during the middle third.
        if self.recurrent_type == 'fullconn':
            self.W = nn.Parameter(
                (0.1 / np.sqrt(self.n_units))
                * torch.randn(self.n_units, self.n_units)
            )
        elif self.recurrent_type == 'lowrank':
            self.U = nn.Parameter(torch.randn(self.n_units, self.recurrent_rank))
            self.V = nn.Parameter(torch.randn(self.n_units, self.recurrent_rank))
            # Var(W_ij) = scale^2 * rank = 0.01 / n_units, as for fullconn.
            self.recurrent_scale = 0.1 / np.sqrt(self.n_units * self.recurrent_rank)
        input_scale = np.sqrt(0.01 / self.n_input_components)
        self.input_vectors = nn.Parameter(
            input_scale * torch.randn(
                self.n_input_types,
                self.n_conditions,
                self.n_input_components,
                self.n_units,
            )
        )
        self.n_initial_timepoints = validate_initial_timepoints(n_initial_timepoints, self.n_time)
        if self.n_initial_timepoints is None:
            # Retain exact simulation behavior when loading version 2/3 checkpoints.
            self.initial_state_layer = None
            self.register_buffer('h0', torch.zeros(self.n_conditions, self.n_units))
        else:
            n = self.n_initial_timepoints
            self.initial_state_layer = nn.Linear(
                self.n_units * n, self.n_units,
                device=self.input_vectors.device, dtype=self.input_vectors.dtype,
            )
            # Flattening is neuron-major: initialize each row to its neuron's mean.
            with torch.no_grad():
                identity = torch.eye(
                    self.n_units, device=self.input_vectors.device, dtype=self.input_vectors.dtype,
                )
                self.initial_state_layer.weight.copy_(identity.repeat_interleave(n, dim=1) / n)
                self.initial_state_layer.bias.zero_()

    def initial_activity_from_targets(self, targets):
        """Extract only the observed prefix, ordered by this session's conditions.

        targets maps condition names to (neurons, time) arrays/tensors. The
        returned tensor has shape (conditions, neurons, n_initial_timepoints).
        """
        if self.initial_state_layer is None:
            return None
        windows = []
        for condition in self.conditions:
            values = torch.as_tensor(
                targets[condition], device=self.input_vectors.device,
                dtype=self.input_vectors.dtype,
            )
            if (
                values.ndim != 2 or values.shape[0] != self.n_units
                or values.shape[1] < self.n_initial_timepoints
            ):
                raise ValueError(
                    f'{condition}: expected {self.n_units} neurons and at least '
                    f'{self.n_initial_timepoints} initial time points.'
                )
            windows.append(values[:, :self.n_initial_timepoints].detach())
        return torch.stack(windows)

    def initial_state(self, initial_activity=None):
        """Infer the internal state; ReLU converts it to the first activity bin."""
        if self.initial_state_layer is None:
            if initial_activity is not None:
                raise ValueError('This legacy network has no learned initial-state layer.')
            return self.h0
        if initial_activity is None:
            raise ValueError(
                'initial_activity is required for a learned initial state; '
                'use initial_activity_from_targets(targets).'
            )
        window = torch.as_tensor(
            initial_activity, device=self.input_vectors.device, dtype=self.input_vectors.dtype,
        ).detach()
        expected_shape = (self.n_conditions, self.n_units, self.n_initial_timepoints)
        if tuple(window.shape) != expected_shape:
            raise ValueError(f'initial_activity must have shape {expected_shape}.')
        if not torch.isfinite(window).all():
            raise ValueError('initial_activity must contain only finite values.')
        return self.initial_state_layer(window.flatten(start_dim=1))

    def sample_input_timecourses(self):
        """Draw fresh, untrained 1 + Gaussian noise, including in eval mode."""
        shape = (
            self.n_input_types,
            self.n_conditions,
            self.n_input_components,
            self.n_time,
        )
        return 1.0 + self.input_noise_std * torch.randn(
            shape, device=self.input_vectors.device, dtype=self.input_vectors.dtype
        )

    def condition_input(self, icond, input_timecourses=None):
        """Sum separately masked input types; return (time, neurons)."""
        if input_timecourses is None:
            input_timecourses = self.sample_input_timecourses()
        phase_inputs = torch.einsum(
            'pkn,pkt->ptn',
            self.input_vectors[:, icond],
            input_timecourses[:, icond],
        )
        return (phase_inputs * self.input_masks[:, :, None]).sum(dim=0)

    def recurrent_weight(self):
        """Return W, reconstructing it from U and V for low-rank recurrence."""
        if hasattr(self, 'svd_left'):
            return (self.svd_left * self.svd_values) @ self.svd_right.T
        if self.recurrent_type == 'fullconn':
            return self.W
        elif self.recurrent_type == 'lowrank':
            return self.recurrent_scale * (self.U @ self.V.T)

    @torch.no_grad()
    def truncated_recurrent_weight(self, truncation_rank):
        W = self.recurrent_weight().detach()
        U_svd, singular_values, Vh_svd = torch.linalg.svd(W, full_matrices=False)
        k = int(truncation_rank)
        return (U_svd[:, :k] * singular_values[:k]) @ Vh_svd[:k, :]

    def recurrent_drive(self, rates, train_recurrent):
        """Compute rates @ W.T, detaching recurrent parameters when requested."""
        if hasattr(self, 'svd_left'):
            U = self.svd_left if train_recurrent else self.svd_left.detach()
            s = self.svd_values if train_recurrent else self.svd_values.detach()
            V = self.svd_right if train_recurrent else self.svd_right.detach()
            return ((rates @ V) * s) @ U.T
        if self.recurrent_type == 'fullconn':
            W = self.W if train_recurrent else self.W.detach()
            return rates @ W.T
        elif self.recurrent_type == 'lowrank':
            U = self.U if train_recurrent else self.U.detach()
            V = self.V if train_recurrent else self.V.detach()
            # W = scale * U @ V.T; apply its transpose through the factors.
            return self.recurrent_scale * ((rates @ V) @ U.T)

    def initialize_readout_retraining(
        self, reference, initial_activity=None,
        min_cd2_norm_ratio=DEFAULT_MIN_CD2_NORM_RATIO, *, svd_factors=None,
        train_initial_state_layer=False, train_sensory_input=False,
        train_left_vectors=False,
    ):
        """Freeze the source model and attach unconstrained SVD/readout factors.

        The initial left SVD factor stays fixed unless train_left_vectors=True.
        The unconstrained factors need not remain an SVD after updates.
        svd_factors restores saved factors without recomputing a decomposition.
        """
        _validate_readout_session(self)
        if hasattr(self, 'svd_left'):
            raise ValueError('Readout retraining is already initialized for this session.')
        self.min_cd2_norm_ratio = _validate_scalar(min_cd2_norm_ratio, 'min_cd2_norm_ratio')
        self.requires_grad_(False)
        self.zero_grad(set_to_none=True)
        if train_initial_state_layer:
            self.initial_state_layer.requires_grad_(True)
        self.input_vectors.requires_grad_(train_sensory_input)
        if svd_factors is None:
            with torch.no_grad():
                # CPU initialization also supports models running on MPS.
                if self.recurrent_type == 'lowrank':
                    # Exact SVD of scale * U @ V.T via its small QR core.
                    # This avoids an expensive neurons-by-neurons dense SVD.
                    q_left, r_left = torch.linalg.qr(self.U.detach().cpu(), mode='reduced')
                    q_right, r_right = torch.linalg.qr(self.V.detach().cpu(), mode='reduced')
                    left, values, right_h = torch.linalg.svd(
                        self.recurrent_scale * (r_left @ r_right.T), full_matrices=False,
                    )
                    svd_factors = (q_left @ left, values, q_right @ right_h.T)
                else:
                    left, values, right_h = torch.linalg.svd(
                        self.W.detach().cpu(), full_matrices=False,
                    )
                    svd_factors = (left, values, right_h.T)
        device, dtype = self.input_vectors.device, self.input_vectors.dtype
        left, values, right = (
            value.detach().to(device=device, dtype=dtype).clone() for value in svd_factors
        )
        self.svd_left = nn.Parameter(left, requires_grad=train_left_vectors)
        self.svd_values = nn.Parameter(values)
        self.svd_right = nn.Parameter(right)
        for name in CONTEXT1_REFERENCE_KEYS:
            self.register_buffer(
                f'reference_{name}', reference[name].detach().to(device=device, dtype=dtype).clone(),
            )
        self.register_buffer(
            'readout_initial_activity', None if initial_activity is None else
            initial_activity.detach().to(device=device, dtype=dtype).clone(),
        )
        self.alpha = nn.Parameter(self.reference_cd1.new_tensor(1.0))
        self.u = nn.Parameter(torch.zeros_like(self.reference_cd1))

    def context2_readout(self, activity):
        """Read out live P1/A1 responses as P2/A2, preserving all gradients."""
        if not hasattr(self, 'svd_left'):
            raise ValueError('Initialize readout retraining before computing context 2 readouts.')
        start, stop = READOUT_WINDOW
        rP2 = activity['P1'][start:stop].mean(dim=0)
        rA2 = activity['A1'][start:stop].mean(dim=0)
        cd2 = rP2 - rA2
        r_half2 = (rP2 + rA2) / 2
        norm_cd2 = cd2.norm()
        min_norm = self.min_cd2_norm_ratio * self.reference_norm_cd1
        norm_cd2_safe = norm_cd2.clamp_min(min_norm)
        cd1 = self.reference_cd1
        u_ortho = self.u - cd1 * (torch.dot(cd1, self.u) / cd1.square().sum())
        w_o = self.alpha * cd1 + u_ortho
        delta_rP2 = (rP2 - r_half2) / norm_cd2_safe
        delta_rA2 = (rA2 - r_half2) / norm_cd2_safe
        return {
            'rP2': rP2, 'rA2': rA2, 'cd2': cd2, 'r_half2': r_half2,
            'norm_cd2': norm_cd2, 'norm_cd2_safe': norm_cd2_safe,
            'cd2_clamped': norm_cd2 < min_norm,
            'u_ortho': u_ortho, 'w_o': w_o,
            'delta_rP2': delta_rP2, 'delta_rA2': delta_rA2,
            'zP2': torch.dot(w_o, delta_rP2), 'zA2': torch.dot(w_o, delta_rA2),
        }

    def readout(self, input_timecourses=None, initial_activity=None, truncation_rank=None):
        """Simulate with fresh input noise by default and return context 2 readouts."""
        if not hasattr(self, 'svd_left'):
            raise ValueError('Initialize readout retraining before computing context 2 readouts.')
        if initial_activity is None:
            initial_activity = self.readout_initial_activity
        return self.context2_readout(self(
            input_timecourses=input_timecourses, initial_activity=initial_activity,
            truncation_rank=truncation_rank,
        ))

    def forward(self, truncation_rank=None, input_timecourses=None, initial_activity=None):
        """Return activity, training recurrence only in middle-third updates.

        The initial state is inferred from initial_activity (conditions, neurons,
        n_initial_timepoints). Only legacy networks start at zero. The observed
        window conditions the simulation starting at bin 0; later targets are
        never fed into the dynamics. Detached W remains active outside the
        middle third, preserving gradients through states and input vectors.
        For low-rank recurrence, both factors are detached outside that window.

        In eval mode, truncation_rank uses an SVD approximation throughout the
        simulation. None uses the original recurrence. Optional untrained
        input_timecourses have shape (types, conditions, components, time) and
        allow comparisons with shared noise; otherwise fresh noise is drawn.
        """
        h = self.initial_state(initial_activity)
        W_sim = None
        if truncation_rank is not None:
            if self.training:
                raise RuntimeError('Truncated simulation requires eval() mode.')
            W_sim = self.truncated_recurrent_weight(truncation_rank)

        # One fresh draw contains independent noise for every type, condition,
        # component, and time point. Different session models draw separately.
        if input_timecourses is None:
            input_timecourses = self.sample_input_timecourses()
        else:
            input_timecourses = torch.as_tensor(
                input_timecourses,
                device=self.input_vectors.device,
                dtype=self.input_vectors.dtype,
            ).detach()
            expected_shape = (
                self.n_input_types, self.n_conditions,
                self.n_input_components, self.n_time,
            )
            if tuple(input_timecourses.shape) != expected_shape:
                raise ValueError(f'input_timecourses must have shape {expected_shape}.')
            if not torch.isfinite(input_timecourses).all():
                raise ValueError('input_timecourses must contain only finite values.')
        inputs = torch.stack([
            self.condition_input(icond, input_timecourses)
            for icond in range(self.n_conditions)
        ])
        rates_over_time = [F.relu(h)]
        middle_start = self.n_time // 3
        middle_stop = 2 * self.n_time // 3
        for t in range(self.n_time - 1):
            rates = F.relu(h)
            train_recurrent = middle_start <= t < middle_stop
            if W_sim is None:
                drive = self.recurrent_drive(rates, train_recurrent)
            else:
                drive = rates @ W_sim.T
            dh = -h + drive + inputs[:, t]
            h = h + self.dt * dh
            rates_over_time.append(F.relu(h))
        activity = torch.stack(rates_over_time, dim=1)
        return {condition: activity[i] for i, condition in enumerate(self.conditions)}


class LatentRNN(nn.Module):
    """Independent session networks, with unit counts matching activity rows."""

    def __init__(
        self,
        session_shapes,
        conditions=CONDITIONS,
        n_input_types=DEFAULT_N_INPUT_TYPES,
        n_input_components=DEFAULT_N_INPUT_COMPONENTS,
        input_noise_std=DEFAULT_INPUT_NOISE_STD,
        dt=DT,
        input_masks=None,
        recurrent_type=DEFAULT_RECURRENT_TYPE,
        recurrent_rank=None,
        n_initial_timepoints=DEFAULT_N_INITIAL_TIMEPOINTS,
    ):
        super().__init__()
        self.session_shapes = {
            int(fx): tuple(map(int, shape)) for fx, shape in session_shapes.items()
        }
        self.session_ids = tuple(sorted(self.session_shapes))
        self.n_sessions = len(self.session_ids)
        self.n_units_by_session = {fx: shape[0] for fx, shape in self.session_shapes.items()}
        self.n_time_by_session = {fx: shape[1] for fx, shape in self.session_shapes.items()}
        self.n_initial_timepoints = validate_initial_timepoints(
            n_initial_timepoints, min(self.n_time_by_session.values()),
        )
        self.recurrent_type = recurrent_type
        self.recurrent_rank = validate_recurrent_config(
            recurrent_type, recurrent_rank, min(self.n_units_by_session.values()),
        )
        self.conditions = tuple(conditions)
        self.n_input_types = int(n_input_types)
        self.n_input_components = int(n_input_components)
        self.input_noise_std = float(input_noise_std)
        self.dt = float(dt)
        self.sessions = nn.ModuleDict({
            str(fx): SessionRNN(
                *self.session_shapes[fx],
                conditions=self.conditions,
                n_input_types=self.n_input_types,
                n_input_components=self.n_input_components,
                input_noise_std=self.input_noise_std,
                dt=self.dt,
                input_masks=input_masks[fx] if isinstance(input_masks, dict) else input_masks,
                recurrent_type=self.recurrent_type,
                recurrent_rank=self.recurrent_rank,
                n_initial_timepoints=self.n_initial_timepoints,
            )
            for fx in self.session_ids
        })

    def session_model(self, fx):
        return self.sessions[str(int(fx))]

    def initial_activity_from_targets(self, fx, targets):
        return self.session_model(fx).initial_activity_from_targets(targets)

    def recurrent_weight(self, fx):
        return self.session_model(fx).recurrent_weight()

    def truncated_recurrent_weight(self, fx, truncation_rank):
        return self.session_model(fx).truncated_recurrent_weight(truncation_rank)

    def readout(self, fx, input_timecourses=None, initial_activity=None, truncation_rank=None):
        return self.session_model(fx).readout(
            input_timecourses=input_timecourses, initial_activity=initial_activity,
            truncation_rank=truncation_rank,
        )

    def forward(self, fx, truncation_rank=None, input_timecourses=None, initial_activity=None):
        """Simulate one session; optional SVD truncation requires eval() mode."""
        return self.session_model(fx)(
            truncation_rank=truncation_rank,
            input_timecourses=input_timecourses,
            initial_activity=initial_activity,
        )


# Load just the requested activity file from the recurrent generator.
def available_training_datasets(training_path=TRAINING_DATA_PATH):
    """List dataset IDs in training_data/ or an explicit data directory."""
    root = Path(training_path)
    root = root / 'training_data' if (root / 'training_data').is_dir() else root
    ids = [
        path.stem[len('activity'):]
        for path in root.glob('activity*.npy')
    ]
    return sorted(ids, key=lambda value: (0, int(value)) if value.isdecimal() else (1, value))


def dataset_session_shapes(dataset, conditions=CONDITIONS):
    """Infer session sizes, validating finite targets and condition shapes."""
    if not conditions or not dataset['activity']:
        raise ValueError('Training requires conditions and at least one session.')
    shapes = {}
    for fx, values in dataset['activity'].items():
        if any(condition not in values for condition in conditions):
            raise ValueError(f'Session {fx} is missing a requested condition.')
        shape = np.asarray(values[conditions[0]]).shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 3:
            raise ValueError(f'Session {fx}: expected (neurons, time) with time >= 3.')
        for condition, target in values.items():
            if np.asarray(target).shape != shape or not np.isfinite(target).all():
                raise ValueError(f'Session {fx}, {condition}: inconsistent shape or nonfinite activity.')
        shapes[int(fx)] = shape
    return shapes


def load_training_data(
    training_path=TRAINING_DATA_PATH,
    dataset_id=DEFAULT_DATASET_ID,
    conditions=CONDITIONS,
):
    """Load from training_data/ or an explicit data directory, keeping row order."""
    dataset_id = str(dataset_id)
    available = available_training_datasets(training_path)
    if dataset_id not in available:
        raise FileNotFoundError(
            f'Dataset {dataset_id} not found in {training_path}. '
            f'Available dataset IDs: {available}'
        )
    root = Path(training_path)
    root = root / 'training_data' if (root / 'training_data').is_dir() else root
    activity = np.load(root / f'activity{dataset_id}.npy', allow_pickle=True).item()
    dataset = {
        'id': dataset_id,
        'activity': activity,
    }
    dataset_session_shapes(dataset, conditions)
    return dataset


def train_latent_network(
    n_steps=200,
    learning_rate=1e-3,
    dataset_id=DEFAULT_DATASET_ID,
    n_input_types=DEFAULT_N_INPUT_TYPES,
    n_input_components=DEFAULT_N_INPUT_COMPONENTS,
    input_noise_std=DEFAULT_INPUT_NOISE_STD,
    conditions=None,
    dt=DT,
    input_masks=None,
    training_path=TRAINING_DATA_PATH,
    save_path=None,
    checkpoint_path=None,
    device=DEFAULT_DEVICE,
    recurrent_type=DEFAULT_RECURRENT_TYPE,
    recurrent_rank=None,
    n_initial_timepoints=None,
):
    """Train new session networks, averaging MSE equally over sessions/conditions.

    Each session learns a linear map from its first n_initial_timepoints target
    bins to its initial state, shared across conditions. None uses the default
    of 2 time points. Conditions default to P1/A1. Every session has independent
    recurrent weights, input vectors, and an initial-state layer, with one unit
    per neuron. Every simulation draws new input noise.
    W receives gradients only through the middle-third recurrent updates.
    Each session backpropagates immediately, freeing its graph before the next.

    Histories contain pre-update losses from this run only; Adam starts afresh.
    The full-interval loss includes the observed prefix; predictive metrics
    should exclude those conditioning bins.
    """
    if n_steps < 1 or learning_rate <= 0:
        raise ValueError('n_steps and learning_rate must be positive.')
    conditions = tuple(CONDITIONS if conditions is None else conditions)
    if n_initial_timepoints is None:
        n_initial_timepoints = DEFAULT_N_INITIAL_TIMEPOINTS
    device = resolve_device(device)
    dataset = load_training_data(training_path, dataset_id, conditions)
    rnn = LatentRNN(
        dataset_session_shapes(dataset, conditions),
        conditions=conditions,
        n_input_types=n_input_types,
        n_input_components=n_input_components,
        input_noise_std=input_noise_std,
        dt=dt,
        input_masks=input_masks,
        recurrent_type=recurrent_type,
        recurrent_rank=recurrent_rank,
        n_initial_timepoints=n_initial_timepoints,
    ).to(device)
    targets = {
        fx: {
            condition: torch.as_tensor(
                dataset['activity'][fx][condition], dtype=torch.float32, device=device
            )
            for condition in rnn.conditions
        }
        for fx in rnn.session_ids
    }
    initial_activity_by_session = {
        fx: rnn.initial_activity_from_targets(fx, targets[fx])
        for fx in rnn.session_ids
    }
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    loss_history = []
    loss_history_by_session = {fx: [] for fx in rnn.session_ids}
    training_start_time = time.time()
    rnn.train()
    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for fx in rnn.session_ids:
            predictions = rnn(fx, initial_activity=initial_activity_by_session[fx])
            session_loss = torch.stack([
                F.mse_loss(predictions[condition].T, targets[fx][condition])
                for condition in rnn.conditions
            ]).mean()
            (session_loss / rnn.n_sessions).backward()
            value = session_loss.detach().cpu().item()
            loss_history_by_session[fx].append(value)
            step_loss += value / rnn.n_sessions
        optimizer.step()
        loss_history.append(step_loss)
        if step % 50 == 0 or step == n_steps - 1:
            print(f'step {step:04d} | loss {step_loss:.6f}')
        if step == 49:
            estimated_minutes = (time.time() - training_start_time) * n_steps / 50 / 60
            print(f'estimated total training time: {estimated_minutes:.1f} minutes')
    results = {
        'rnn': rnn, 'dataset': dataset, 'loss_history': loss_history,
        'loss_history_by_session': loss_history_by_session,
        'training_path': str(training_path),
    }
    save_latent_network(results, save_path, checkpoint_path)
    return results


def retrain_latent_network_readout(
    source_checkpoint, n_steps=200,
    learning_rate_alpha=1e-3, learning_rate_u=1e-3,
    learning_rate_singular_values=1e-3, learning_rate_right_vectors=1e-3,
    lambda_readout=1.0, lambda_coding=1.0,
    min_cd2_norm_ratio=DEFAULT_MIN_CD2_NORM_RATIO,
    input_noise_std=None, training_path=TRAINING_DATA_PATH,
    save_path=None, checkpoint_path=None, device=DEFAULT_DEVICE,
    *, train_initial_state_layer=False, train_sensory_input=False,
    learning_rate_initial_state_layer=1e-3, learning_rate_sensory_input=1e-3,
    train_left_vectors=False, learning_rate_left_vectors=1e-3,
):
    """Reverse fixed context 1 readouts with fresh input noise on every rollout.

    References use activity[12:16] from one source simulation per session.
    Optimize alpha, u, the signed SVD coefficients, and the unconstrained right
    factor. Optionally retrain the left factor, the existing initial-state
    layer (weight/bias), and sensory input vectors. Response input stays fixed.
    Recurrence receives gradients only through middle-third updates.
    None for input_noise_std inherits the source noise level, which must be
    positive. Each session/condition/component/time bin gets independent
    1 + Gaussian amplitude noise on every forward pass, including evaluation.
    """
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError('n_steps must be a positive integer.')
    rates = {
        'alpha': _validate_scalar(learning_rate_alpha, 'learning_rate_alpha'),
        'u': _validate_scalar(learning_rate_u, 'learning_rate_u'),
        'svd_values': _validate_scalar(learning_rate_singular_values, 'learning_rate_singular_values'),
        'svd_right': _validate_scalar(learning_rate_right_vectors, 'learning_rate_right_vectors'),
    }
    if train_left_vectors:
        rates['svd_left'] = _validate_scalar(learning_rate_left_vectors, 'learning_rate_left_vectors')
    if train_initial_state_layer:
        rates['initial_state_layer'] = _validate_scalar(
            learning_rate_initial_state_layer, 'learning_rate_initial_state_layer',
        )
    if train_sensory_input:
        rates['input_vectors'] = _validate_scalar(learning_rate_sensory_input, 'learning_rate_sensory_input')
    lambda_readout = _validate_scalar(lambda_readout, 'lambda_readout', allow_zero=True)
    lambda_coding = _validate_scalar(lambda_coding, 'lambda_coding', allow_zero=True)
    min_cd2_norm_ratio = _validate_scalar(min_cd2_norm_ratio, 'min_cd2_norm_ratio')
    loaded = load_latent_network(source_checkpoint, training_path=training_path, device=device)
    if 'readout_retraining' in loaded:
        raise ValueError('The source must be a context 1 checkpoint, not an already retrained readout.')
    rnn, dataset = loaded['rnn'], loaded['dataset']
    for session in rnn.sessions.values():
        _validate_readout_session(session)
    noise_std = _validate_scalar(
        rnn.input_noise_std if input_noise_std is None else input_noise_std,
        'input_noise_std (fresh-noise readout retraining)',
    )
    if checkpoint_path is None:
        root = None if save_path is None else Path(save_path) / rnn.recurrent_type
        checkpoint_path = readout_retrained_checkpoint_path(source_checkpoint, root)
    if Path(checkpoint_path).resolve() == Path(source_checkpoint).resolve():
        raise ValueError('The readout checkpoint must differ from the source checkpoint.')
    rnn.input_noise_std = noise_std
    rnn.eval()
    for fx in rnn.session_ids:
        session = rnn.session_model(fx)
        session.input_noise_std = noise_std
        initial_activity = session.initial_activity_from_targets(dataset['activity'][fx])
        reference = compute_context1_reference(session, initial_activity)
        session.initialize_readout_retraining(
            reference, initial_activity, min_cd2_norm_ratio,
            train_initial_state_layer=train_initial_state_layer,
            train_sensory_input=train_sensory_input,
            train_left_vectors=train_left_vectors,
        )

    parameter_groups = []
    for name, rate in rates.items():
        parameters = []
        for session in rnn.sessions.values():
            if name == 'initial_state_layer':
                parameters.extend(session.initial_state_layer.parameters())
            else:
                parameters.append(getattr(session, name))
        parameter_groups.append({'params': parameters, 'lr': rate, 'name': name})
    optimizer = torch.optim.Adam(parameter_groups, weight_decay=0.0)
    trainable = [parameter for group in optimizer.param_groups for parameter in group['params']]
    loss_history = []
    loss_history_by_session = {fx: [] for fx in rnn.session_ids}
    readout_history_by_session = {fx: [] for fx in rnn.session_ids}
    rnn.train()
    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        step_metrics = []
        for fx in rnn.session_ids:
            session = rnn.session_model(fx)
            # No cached input_timecourses: readout() samples fresh noise here.
            readout = session.readout()
            losses = readout_reversal_loss(session, readout, lambda_readout, lambda_coding)
            if not torch.isfinite(losses['loss']):
                raise ValueError(f'Nonfinite readout loss at step {step}, session {fx}.')
            (losses['loss'] / rnn.n_sessions).backward()
            if train_sensory_input and session.input_vectors.grad is not None:
                # Response input stays fixed; only sensory input (type 0) is updated.
                session.input_vectors.grad[1:].zero_()
            metrics = {name: value.detach().cpu().item() for name, value in losses.items()}
            metrics.update({
                'alpha': session.alpha.detach().cpu().item(),
                'norm_u_ortho': readout['u_ortho'].detach().norm().cpu().item(),
                'norm_cd2': readout['norm_cd2'].detach().cpu().item(),
                'norm_cd2_safe': readout['norm_cd2_safe'].detach().cpu().item(),
                'cd2_clamped': readout['cd2_clamped'].detach().cpu().item(),
                'zP2': readout['zP2'].detach().cpu().item(),
                'zA2': readout['zA2'].detach().cpu().item(),
            })
            loss_history_by_session[fx].append(metrics['loss'])
            readout_history_by_session[fx].append(metrics)
            step_metrics.append(metrics)
        if not torch.stack([
            torch.isfinite(parameter.grad).all()
            for parameter in trainable if parameter.grad is not None
        ]).all():
            raise ValueError(f'Nonfinite readout gradients at step {step}; lower the learning rates.')
        optimizer.step()
        loss_history.append(sum(item['loss'] for item in step_metrics) / rnn.n_sessions)
        if step % 50 == 0 or step == n_steps - 1:
            means = {
                name: sum(item[name] for item in step_metrics) / rnn.n_sessions
                for name in ('loss_reverse', 'loss_readout_constraint', 'loss_coding_constraint')
            }
            alphas = [item['alpha'] for item in step_metrics]
            clamps = sum(item['cd2_clamped'] for item in step_metrics)
            print(
                f'step {step:04d} | loss {loss_history[-1]:.6g} '
                f'| reverse {means["loss_reverse"]:.6g} '
                f'| readout constraint {means["loss_readout_constraint"]:.6g} '
                f'| coding constraint {means["loss_coding_constraint"]:.6g} '
                f'| alpha [{min(alphas):.4g}, {max(alphas):.4g}] | cd2 clamps {clamps}'
            )
    results = {
        'rnn': rnn, 'dataset': dataset, 'loss_history': loss_history,
        'loss_history_by_session': loss_history_by_session,
        'readout_history_by_session': readout_history_by_session,
        'training_path': str(training_path),
        'readout_retraining': {
            'method': 'unconstrained_svd' if train_left_vectors else 'fixed_left_unconstrained_svd',
            'source_checkpoint': str(Path(source_checkpoint).resolve()),
            'readout_window': list(READOUT_WINDOW),
            'min_cd2_norm_ratio': min_cd2_norm_ratio,
            'learning_rates': rates,
            'train_initial_state_layer': train_initial_state_layer,
            'train_sensory_input': train_sensory_input,
            'train_left_vectors': train_left_vectors,
            'lambda_readout': lambda_readout, 'lambda_coding': lambda_coding,
            'input_noise_policy': 'fresh_per_session_condition_component_time_bin_and_forward',
            'recurrent_gradient_window': 'middle_third',
        },
    }
    save_latent_network(results, checkpoint_path=checkpoint_path)
    return results


def save_latent_network(results, save_path=None, checkpoint_path=None):
    rnn = results['rnn']
    dataset = results['dataset']
    readout_metadata = results.get('readout_retraining')
    readout_sessions = [hasattr(session, 'svd_left') for session in rnn.sessions.values()]
    if any(readout_sessions) or readout_metadata is not None:
        if readout_metadata is None or not all(readout_sessions):
            raise ValueError('Readout checkpoints require metadata and initialized readouts for every session.')
    if checkpoint_path is None and readout_metadata is not None:
        root = None if save_path is None else Path(save_path) / rnn.recurrent_type
        checkpoint_path = readout_retrained_checkpoint_path(readout_metadata['source_checkpoint'], root)
    if checkpoint_path is None:
        checkpoint_path = latent_network_checkpoint_path(
            dataset['id'], rnn.n_input_types, rnn.n_input_components,
            rnn.input_noise_std, save_path,
            recurrent_type=rnn.recurrent_type,
            recurrent_rank=rnn.recurrent_rank,
            training_path=results.get('training_path', TRAINING_DATA_PATH),
        )
    path = Path(checkpoint_path)
    if readout_metadata is not None and path.resolve() == Path(readout_metadata['source_checkpoint']).resolve():
        raise ValueError('The readout checkpoint must differ from the source checkpoint.')
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'version': READOUT_CHECKPOINT_VERSION if readout_metadata is not None else CHECKPOINT_VERSION,
        'rnn_state_dict': rnn.state_dict(),
        'loss_history': results['loss_history'],
        'loss_history_by_session': results.get('loss_history_by_session', {}),
        'dataset_id': dataset['id'],
        'config': {
            'session_shapes': rnn.session_shapes,
            'conditions': list(rnn.conditions),
            'n_input_types': rnn.n_input_types,
            'n_input_components': rnn.n_input_components,
            'input_noise_std': rnn.input_noise_std,
            'dt': rnn.dt,
            'recurrent_type': rnn.recurrent_type,
            'recurrent_rank': rnn.recurrent_rank,
            'n_initial_timepoints': rnn.n_initial_timepoints,
        },
    }
    if readout_metadata is not None:
        checkpoint['readout_retraining'] = readout_metadata
        checkpoint['readout_history_by_session'] = results.get('readout_history_by_session', {})
    torch.save(checkpoint, path)
    results['checkpoint_path'] = str(path)
    print(f'Saved latent network to {path}')


def load_latent_network(
    checkpoint_path=None,
    training_path=TRAINING_DATA_PATH,
    device=DEFAULT_DEVICE,
):
    """Restore the selected dataset and variable-sized session networks.

    Ordinary forward simulations remain stochastic, including in eval mode.
    Version 6 restores fixed context 1 references and unconstrained SVD/readout
    factors. rnn.readout(fx) uses saved initial activity and fresh input noise.
    Versions 2/3 retain their fixed-zero initial states. New learned-state models
    require initial_activity when simulated directly; plotting helpers extract it.
    """
    if checkpoint_path is None:
        checkpoint_path = latent_network_checkpoint_path(training_path=training_path)
    device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get('version') not in (2, 3, CHECKPOINT_VERSION, READOUT_CHECKPOINT_VERSION):
        raise ValueError('Unsupported checkpoint format; expected a version 2, 3, 4, or 6 latent-network checkpoint.')
    readout_metadata = checkpoint.get('readout_retraining')
    if checkpoint['version'] == READOUT_CHECKPOINT_VERSION:
        if readout_metadata is None:
            raise ValueError('Readout checkpoint is missing readout_retraining metadata.')
        if tuple(readout_metadata['readout_window']) != READOUT_WINDOW:
            raise ValueError('Readout checkpoint must use activity[12:16].')
    config = dict(checkpoint['config'])
    # Version 2 checkpoints contained only fully connected recurrent weights.
    config.setdefault('recurrent_type', DEFAULT_RECURRENT_TYPE)
    config.setdefault('recurrent_rank', None)
    if checkpoint['version'] in (2, 3):
        config.setdefault('n_initial_timepoints', None)
    dataset = load_training_data(training_path, checkpoint['dataset_id'], config['conditions'])
    if dataset_session_shapes(dataset, config['conditions']) != config['session_shapes']:
        raise ValueError('Training data shapes differ from the saved model.')
    # Reconstruct custom masks too, before restoring the remaining buffers.
    input_masks = {
        fx: checkpoint['rnn_state_dict'][f'sessions.{fx}.input_masks']
        for fx in config['session_shapes']
    }
    rnn = LatentRNN(**config, input_masks=input_masks).to(device)
    if checkpoint['version'] == READOUT_CHECKPOINT_VERSION:
        state = checkpoint['rnn_state_dict']
        for fx in rnn.session_ids:
            prefix = f'sessions.{fx}.'
            reference = {name: state[prefix + 'reference_' + name] for name in CONTEXT1_REFERENCE_KEYS}
            rnn.session_model(fx).initialize_readout_retraining(
                reference, state.get(prefix + 'readout_initial_activity'),
                readout_metadata['min_cd2_norm_ratio'],
                svd_factors=tuple(state[prefix + name] for name in ('svd_left', 'svd_values', 'svd_right')),
                train_initial_state_layer=readout_metadata.get('train_initial_state_layer', False),
                train_sensory_input=readout_metadata.get('train_sensory_input', False),
                train_left_vectors=readout_metadata.get('train_left_vectors', False),
            )
    rnn.load_state_dict(checkpoint['rnn_state_dict'])
    results = {
        'rnn': rnn,
        'dataset': dataset,
        'loss_history': checkpoint['loss_history'],
        'loss_history_by_session': checkpoint.get('loss_history_by_session', {}),
        'checkpoint_path': str(checkpoint_path),
        'training_path': str(training_path),
    }
    if checkpoint['version'] == READOUT_CHECKPOINT_VERSION:
        results['readout_retraining'] = readout_metadata
        results['readout_history_by_session'] = checkpoint.get('readout_history_by_session', {})
    return results


# Compare each real neuron with the corresponding recurrent unit directly.
def plot_prediction_vs_target(
    results, fx=None, condition=None, savefig=True,
    truncation_rank=None, input_timecourses=None, initial_activity=None,
):
    """Compare matching neuron/unit rows using one shared activity scale.

    Defaults to the first trained session and condition. Each comparison uses
    a new simulation. Supply input_timecourses to reuse noise across ranks;
    otherwise it is freshly sampled. truncation_rank selects an SVD approximation.
    By default, infer the initial state from the dataset's observed prefix.
    """
    rnn = results['rnn']
    dataset = results['dataset']
    if fx is None:
        fx = rnn.session_ids[0]
    if condition is None:
        condition = rnn.conditions[0]
    target = np.asarray(dataset['activity'][fx][condition])
    if initial_activity is None:
        initial_activity = rnn.initial_activity_from_targets(fx, dataset['activity'][fx])
    was_training = rnn.training
    rnn.eval()
    try:
        with torch.no_grad():
            prediction = rnn(
                fx, truncation_rank=truncation_rank,
                input_timecourses=input_timecourses,
                initial_activity=initial_activity,
            )[condition].T.cpu().numpy()
    finally:
        rnn.train(was_training)
    vmin = 0
    vmax = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, values, title in zip(axes, (target, prediction), ('Target', 'Prediction')):
        if title=='Target':
            sorted = np.argsort(np.argmax(values,axis=1))
        im = ax.imshow(values[sorted], aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('time bin')
        ax.set_ylabel('neurons / units')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    rank_label = f', SVD rank {truncation_rank}' if truncation_rank is not None else ''
    rank_suffix = f'_TruncRank{truncation_rank}' if truncation_rank is not None else ''
    fig.suptitle(f'Dataset {dataset["id"]}, fx {fx}, {condition}{rank_label}')
    fig.tight_layout()
    if savefig:
        Path(SAVE_FIGURE_PATH).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            Path(SAVE_FIGURE_PATH) / f'compare_dataset{dataset["id"]}_fx{fx}_{condition}{rank_suffix}.png',
            dpi=300,
        )
    return fig


def plot_latent_activity_and_weights(
    results, fx=0, condition='P1', savefig=True,
    truncation_rank=None, input_timecourses=None, initial_activity=None,
):
    """Plot simulated activity and the original or truncated matrix used for it."""
    rnn = results['rnn']
    if initial_activity is None:
        initial_activity = rnn.initial_activity_from_targets(fx, results['dataset']['activity'][fx])
    was_training = rnn.training
    rnn.eval()
    try:
        with torch.no_grad():
            activity = rnn(
                fx, truncation_rank=truncation_rank,
                input_timecourses=input_timecourses,
                initial_activity=initial_activity,
            )[condition].T.cpu().numpy()
            W = (
                rnn.recurrent_weight(fx) if truncation_rank is None
                else rnn.truncated_recurrent_weight(fx, truncation_rank)
            ).cpu().numpy()
    finally:
        rnn.train(was_training)
    rank_label = f', SVD rank {truncation_rank}' if truncation_rank is not None else ''
    rank_suffix = f'_TruncRank{truncation_rank}' if truncation_rank is not None else ''
    fig_activity, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(activity, aspect='auto', cmap='jet', vmin=0, vmax=0.2)
    ax.set(title=f'Unit activity: fx {fx}, {condition}{rank_label}', xlabel='time bin', ylabel='neurons / units')
    fig_activity.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_activity.tight_layout()
    fig_weights, ax = plt.subplots(figsize=(5, 4))
    wlim = np.max(np.abs(W)) / 4 or 1.0
    im = ax.imshow(W, aspect='auto', cmap='bwr', vmin=-wlim, vmax=wlim)
    ax.set(title=f'Recurrent weight W: fx {fx}{rank_label}', xlabel='from unit', ylabel='to unit')
    fig_weights.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_weights.tight_layout()
    if savefig:
        root = Path(SAVE_FIGURE_PATH)
        root.mkdir(parents=True, exist_ok=True)
        dataset_id = results['dataset']['id']
        fig_activity.savefig(root / f'activity_dataset{dataset_id}_fx{fx}_{condition}{rank_suffix}.png', dpi=300)
        fig_weights.savefig(root / f'W_dataset{dataset_id}_fx{fx}{rank_suffix}.png', dpi=300)
    return fig_activity, fig_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Training rate, or default for all enabled retraining rates (default: 0.001).')
    mode.add_argument('--retrain_readout', action='store_true',
                      help='Reverse a context 1 checkpoint readout using activity[12:16] and fresh noise.')
    parser.add_argument('--source_checkpoint', help='Context 1 checkpoint required by --retrain_readout.')
    parser.add_argument('--learning_rate_alpha', type=float, help='Retraining rate for alpha.')
    parser.add_argument('--learning_rate_u', type=float, help='Retraining rate for the orthogonal readout vector u.')
    parser.add_argument('--learning_rate_singular_values', type=float,
                        help='Retraining rate for unconstrained signed SVD coefficients.')
    parser.add_argument('--learning_rate_right_vectors', type=float,
                        help='Retraining rate for the unconstrained right SVD factor.')
    parser.add_argument('--train_left_vectors', action='store_true',
                        help='Retrain the left SVD factor (fixed by default).')
    parser.add_argument('--learning_rate_left_vectors', type=float,
                        help='Retraining rate for the unconstrained left SVD factor.')
    parser.add_argument('--train_initial_state_layer', action='store_true',
                        help='Retrain the existing initial-state layer weights and bias.')
    parser.add_argument('--train_sensory_input', action='store_true',
                        help='Retrain existing sensory input vectors; response input stays fixed.')
    parser.add_argument('--learning_rate_initial_state_layer', type=float,
                        help='Retraining rate for the initial-state layer.')
    parser.add_argument('--learning_rate_sensory_input', type=float,
                        help='Retraining rate for sensory input vectors.')
    parser.add_argument('--lambda_readout', type=float, default=1.0,
                        help='Readout constraint penalty weight (default: 1).')
    parser.add_argument('--lambda_coding', type=float, default=1.0,
                        help='Relative coding-norm constraint penalty weight (default: 1).')
    parser.add_argument('--min_cd2_norm_ratio', type=float, default=DEFAULT_MIN_CD2_NORM_RATIO,
                        help='Readout denominator floor as a fraction of norm(cd1) (default: 1e-6).')
    parser.add_argument('--training_path', default=TRAINING_DATA_PATH)
    parser.add_argument(
        '--save_path',
        help='Checkpoint root override (default: training_path).',
    )
    parser.add_argument('--device', default=DEFAULT_DEVICE)
    parser.add_argument('--dataset_id', default=DEFAULT_DATASET_ID,
                        help='Dataset to use for training or selecting a saved network.')
    parser.add_argument('--recurrent_type', choices=('fullconn', 'lowrank'), default=DEFAULT_RECURRENT_TYPE)
    parser.add_argument('--recurrent_rank', type=int, help='Required rank for lowrank recurrence.')
    mode.add_argument('--list_datasets', action='store_true')
    parser.add_argument('--n_input_types', type=int, default=DEFAULT_N_INPUT_TYPES)
    parser.add_argument('--n_input_components', type=int, default=DEFAULT_N_INPUT_COMPONENTS)
    parser.add_argument('--input_noise_std', type=float,
                        help='Input amplitude noise SD (training: 0.1; retraining: inherit checkpoint, must be > 0).')
    parser.add_argument(
        '--n_initial_timepoints', type=int,
        help='Observed target bins used to learn the initial state (default: 2).',
    )
    parser.add_argument('--conditions', nargs='+',
                        help='Training targets (default: P1 A1).')
    parser.add_argument('--fx', type=int, help='Session to plot after training (default: first trained session).')
    parser.add_argument('--condition', help='Condition to plot after training (default: first trained condition).')
    parser.add_argument('--dt', type=float, default=DT)
    mode.add_argument('--load_only', action='store_true')
    parser.add_argument('--checkpoint_path',
                        help='Output file for training/retraining, or input file with --load_only.')
    args = parser.parse_args()
    if args.source_checkpoint and not args.retrain_readout:
        parser.error('--source_checkpoint requires --retrain_readout.')
    if args.retrain_readout and not args.source_checkpoint:
        parser.error('--retrain_readout requires --source_checkpoint.')
    if not args.retrain_readout and any(rate is not None for rate in (
        args.learning_rate_alpha, args.learning_rate_u,
        args.learning_rate_singular_values, args.learning_rate_right_vectors,
        args.learning_rate_left_vectors,
        args.learning_rate_initial_state_layer, args.learning_rate_sensory_input,
    )):
        parser.error('Readout retraining learning-rate options require --retrain_readout.')
    if (args.train_initial_state_layer or args.train_sensory_input) and not args.retrain_readout:
        parser.error('--train_initial_state_layer and --train_sensory_input require --retrain_readout.')
    if args.train_left_vectors and not args.retrain_readout:
        parser.error('--train_left_vectors requires --retrain_readout.')
    if args.learning_rate_left_vectors is not None and not args.train_left_vectors:
        parser.error('--learning_rate_left_vectors requires --train_left_vectors.')
    if args.learning_rate_initial_state_layer is not None and not args.train_initial_state_layer:
        parser.error('--learning_rate_initial_state_layer requires --train_initial_state_layer.')
    if args.learning_rate_sensory_input is not None and not args.train_sensory_input:
        parser.error('--learning_rate_sensory_input requires --train_sensory_input.')
    training_noise_std = DEFAULT_INPUT_NOISE_STD if args.input_noise_std is None else args.input_noise_std
    if args.list_datasets:
        print('Available dataset IDs:', ', '.join(available_training_datasets(args.training_path)))
    elif args.load_only:
        checkpoint_path = args.checkpoint_path or latent_network_checkpoint_path(
            args.dataset_id, args.n_input_types, args.n_input_components,
            training_noise_std, args.save_path,
            recurrent_type=args.recurrent_type,
            recurrent_rank=args.recurrent_rank,
            training_path=args.training_path,
        )
        loaded_results = load_latent_network(
            checkpoint_path=checkpoint_path,
            training_path=args.training_path,
            device=args.device,
        )
        print(f'Loaded dataset {loaded_results["dataset"]["id"]}; '
              f'{loaded_results["rnn"].n_sessions} sessions.')
    elif args.retrain_readout:
        retrained_results = retrain_latent_network_readout(
            source_checkpoint=args.source_checkpoint, n_steps=args.n_steps,
            learning_rate_alpha=(args.learning_rate if args.learning_rate_alpha is None else args.learning_rate_alpha),
            learning_rate_u=(args.learning_rate if args.learning_rate_u is None else args.learning_rate_u),
            learning_rate_singular_values=(
                args.learning_rate if args.learning_rate_singular_values is None else args.learning_rate_singular_values
            ),
            learning_rate_right_vectors=(
                args.learning_rate if args.learning_rate_right_vectors is None else args.learning_rate_right_vectors
            ),
            train_left_vectors=args.train_left_vectors,
            learning_rate_left_vectors=(
                args.learning_rate if args.learning_rate_left_vectors is None else args.learning_rate_left_vectors
            ),
            train_initial_state_layer=args.train_initial_state_layer,
            train_sensory_input=args.train_sensory_input,
            learning_rate_initial_state_layer=(
                args.learning_rate if args.learning_rate_initial_state_layer is None
                else args.learning_rate_initial_state_layer
            ),
            learning_rate_sensory_input=(
                args.learning_rate if args.learning_rate_sensory_input is None else args.learning_rate_sensory_input
            ),
            lambda_readout=args.lambda_readout, lambda_coding=args.lambda_coding,
            min_cd2_norm_ratio=args.min_cd2_norm_ratio,
            input_noise_std=args.input_noise_std, training_path=args.training_path,
            save_path=args.save_path, checkpoint_path=args.checkpoint_path, device=args.device,
        )
    else:
        trained_results = train_latent_network(
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            dataset_id=args.dataset_id,
            n_input_types=args.n_input_types,
            n_input_components=args.n_input_components,
            input_noise_std=training_noise_std,
            conditions=args.conditions,
            dt=args.dt,
            training_path=args.training_path,
            save_path=args.save_path,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            recurrent_type=args.recurrent_type,
            recurrent_rank=args.recurrent_rank,
            n_initial_timepoints=args.n_initial_timepoints,
        )
        prediction_figure = plot_prediction_vs_target(
            trained_results, fx=args.fx, condition=args.condition,
        )
        plt.show()
