"""Fit session-specific recurrent unit activity directly to neural activity."""

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
READOUT_CHECKPOINT_VERSION = 5
DEFAULT_DELAY_WINDOW = (12, 16)


def validate_delay_window(delay_window, n_time):
    """Validate a half-open time window without silently shortening it."""
    if (
        len(delay_window) != 2
        or any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer))
               for t in delay_window)
        or not 0 <= delay_window[0] < delay_window[1] <= n_time
    ):
        raise ValueError(f'delay_window must satisfy 0 <= start < stop <= {n_time}.')
    return tuple(map(int, delay_window))


def readout_retrained_checkpoint_path(source_checkpoint, save_path=None):
    """Save beside the source, using the parent model folder for previous/ archives."""
    source = Path(source_checkpoint)
    root = source.parent if save_path is None else Path(save_path)
    if save_path is None and root.name == 'previous':
        root = root.parent
    return str(root / f'{source.stem}_retrained_readout.pt')


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
        # None entries do not change ordinary model state_dicts/checkpoints.
        self.register_parameter('Vout', None)
        for name in ('CD1', 'readout_targets', 'readout_input_timecourses',
                     'readout_initial_activity'):
            self.register_buffer(name, None)
        self.readout_delay_window = None
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
        if self.recurrent_type == 'fullconn':
            return self.W
        elif self.recurrent_type == 'lowrank':
            return self.recurrent_scale * (self.U @ self.V.T)

    def recurrent_parameters(self):
        """Return the parameters underlying the effective recurrent matrix."""
        return [self.W] if self.recurrent_type == 'fullconn' else [self.U, self.V]

    def initialize_readout(
        self, cd1, targets, input_timecourses, initial_activity,
        delay_window=DEFAULT_DELAY_WINDOW,
    ):
        """Attach a trainable readout and fixed reference data for this session."""
        self.readout_delay_window = validate_delay_window(delay_window, self.n_time)
        self.requires_grad_(False)
        self.zero_grad(set_to_none=True)
        for parameter in self.recurrent_parameters():
            parameter.requires_grad_(True)
        self.CD1 = cd1.detach().clone()
        self.Vout = nn.Parameter(self.CD1.clone())
        self.readout_targets = targets.detach().clone()
        self.readout_input_timecourses = input_timecourses.detach().clone()
        self.readout_initial_activity = (
            None if initial_activity is None else initial_activity.detach().clone()
        )

    def readout(self, input_timecourses=None, initial_activity=None, truncation_rank=None):
        """Return [P2, A2] readouts using the original P1/A1 drives.

        Saved input noise and observed prefixes are reused by default. Explicit
        input_timecourses allow evaluation with fresh noise. The original
        forward() interface and its P1/A1 condition labels remain unchanged.
        """
        if self.Vout is None:
            raise ValueError('This session has no Vout; retrain it with a readout first.')
        if input_timecourses is None:
            input_timecourses = self.readout_input_timecourses
        if initial_activity is None:
            initial_activity = self.readout_initial_activity
        activity = self(
            input_timecourses=input_timecourses, initial_activity=initial_activity,
            truncation_rank=truncation_rank,
        )
        start, stop = self.readout_delay_window
        return torch.stack([
            self.Vout @ activity[condition][start:stop].mean(dim=0)
            for condition in ('P1', 'A1')
        ])

    @torch.no_grad()
    def truncated_recurrent_weight(self, truncation_rank):
        W = self.recurrent_weight().detach()
        U_svd, singular_values, Vh_svd = torch.linalg.svd(W, full_matrices=False)
        k = int(truncation_rank)
        return (U_svd[:, :k] * singular_values[:k]) @ Vh_svd[:k, :]

    def recurrent_drive(self, rates, train_recurrent):
        """Compute rates @ W.T, detaching recurrent parameters when requested."""
        if self.recurrent_type == 'fullconn':
            W = self.W if train_recurrent else self.W.detach()
            return rates @ W.T
        elif self.recurrent_type == 'lowrank':
            U = self.U if train_recurrent else self.U.detach()
            V = self.V if train_recurrent else self.V.detach()
            # W = scale * U @ V.T; apply its transpose through the factors.
            return self.recurrent_scale * ((rates @ V) @ U.T)

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
        """Return a session's [P2, A2] readouts; defaults to saved retraining inputs."""
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


def _positive_learning_rate(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive.')
    return float(value)


@torch.no_grad()
def _readout_optimizer_step(session, optimizer, predictions, measure_effects=False):
    """Take a joint step and optionally measure each group's isolated effect.

    Probes reuse the saved input draw. Joint post-step parameters are restored
    even if a probe fails; Adam's state advances only for the actual joint step.
    Forward probes include effects of W outside the gradient-enabled window.
    """
    if not measure_effects:
        optimizer.step()
        return None
    recurrent = session.recurrent_parameters()
    before = [p.detach().clone() for p in recurrent]
    v_before = session.Vout.detach().clone()
    optimizer.step()
    after = [p.detach().clone() for p in recurrent]
    v_after = session.Vout.detach().clone()
    try:
        session.Vout.copy_(v_before)
        recurrent_delta = session.readout() - predictions.detach()
        for parameter, value in zip(recurrent, before):
            parameter.copy_(value)
        session.Vout.copy_(v_after)
        readout_delta = session.readout() - predictions.detach()
    finally:
        for parameter, value in zip(recurrent, after):
            parameter.copy_(value)
        session.Vout.copy_(v_after)
    recurrent_delta = recurrent_delta.cpu().double()
    readout_delta = readout_delta.cpu().double()
    recurrent_rms = recurrent_delta.square().mean().sqrt().item()
    readout_rms = readout_delta.square().mean().sqrt().item()
    return {
        'recurrent_delta': recurrent_delta.tolist(),
        'readout_delta': readout_delta.tolist(),
        'recurrent_change_rms': recurrent_rms,
        'readout_change_rms': readout_rms,
        'readout_to_recurrent_ratio': readout_rms / recurrent_rms if recurrent_rms > 0 else None,
    }


def calibrate_readout_learning_rate(session, learning_rate_recurrent=1e-3):
    """Match the RMS effect of the first Adam step on [P2, A2] readouts.

    Both probe groups initially use learning_rate_recurrent. Because readouts
    are linear in Vout, scaling its rate by recurrent_RMS/readout_RMS matches
    their isolated effects at this starting point. The probe optimizer is
    discarded, all parameters are restored, and its gradients are cleared.
    """
    rate = _positive_learning_rate(learning_rate_recurrent, 'learning_rate_recurrent')
    if session.Vout is None:
        raise ValueError('Initialize the session readout before calibrating its learning rate.')
    parameters = session.recurrent_parameters() + [session.Vout]
    original = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.Adam([
        {'params': session.recurrent_parameters(), 'lr': rate},
        {'params': [session.Vout], 'lr': rate},
    ], weight_decay=0.0)
    try:
        optimizer.zero_grad(set_to_none=True)
        predictions = session.readout()
        loss = (predictions - session.readout_targets).square().sum()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite readout loss during learning-rate calibration.')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('Nonfinite gradients during learning-rate calibration.')
        effects = _readout_optimizer_step(session, optimizer, predictions, measure_effects=True)
        rec_rms = effects['recurrent_change_rms']
        out_rms = effects['readout_change_rms']
        finfo = torch.finfo(predictions.dtype)
        tolerance = 10 * finfo.eps * max(predictions.detach().abs().max().item(), finfo.tiny)
        if not np.isfinite([rec_rms, out_rms]).all() or min(rec_rms, out_rms) <= tolerance:
            raise ValueError(
                'Cannot balance learning rates: a parameter group has no measurable '
                'readout effect at the probe rate. Check the delay window and gradients, '
                'or supply learning_rate_readout explicitly.'
            )
        readout_rate = _positive_learning_rate(rate * rec_rms / out_rms, 'learning_rate_readout')
        return readout_rate, {
            'probe_learning_rate': rate,
            'suggested_learning_rate_readout': readout_rate,
            **effects,
        }
    finally:
        with torch.no_grad():
            for parameter, value in zip(parameters, original):
                parameter.copy_(value)
        optimizer.zero_grad(set_to_none=True)


def retrain_latent_network_readout(
    source_checkpoint,
    n_steps=200,
    learning_rate_recurrent=1e-3,
    learning_rate_readout=None,
    delay_window=DEFAULT_DELAY_WINDOW,
    training_path=TRAINING_DATA_PATH,
    save_path=None,
    checkpoint_path=None,
    device=DEFAULT_DEVICE,
    balance_check_interval=50,
    stop_on_readout_tolerance=False,
    max_steps=None,
):
    """Retrain recurrence and per-session Vout on reversed fixed CD1 targets.

    References come from one source-model simulation per session. P2/A2 reuse
    the P1/A1 observed prefixes, frozen initializer, input vectors, and saved
    noise draw. CD1 is the unnormalized difference of mean delay activities.
    None for learning_rate_readout calibrates it separately for each session;
    a positive number overrides calibration for all sessions. Rates stay fixed.
    The existing middle-third recurrent-gradient rule and low rank are retained.

    By default every session receives n_steps updates. With optional stopping,
    each session stops before its next update when loss < 0.01 * abs(target[1]
    - target[0]). n_steps is then ignored: train until all sessions converge,
    or until the optional max_steps cap. The strict criterion requires a
    positive target gap. Initially converged sessions skip calibration and
    optimizer creation; their automatic readout rates/optimizer states are None.

    Histories record pre-update losses/readouts and, in tolerance mode, the
    terminal evaluation. Stopped sessions retain their last loss in aggregate
    histories. Balance probes run at the requested interval and on the last
    scheduled update (0 disables them). The source file is never overwritten.
    """
    if isinstance(n_steps, (bool, np.bool_)) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError('n_steps must be a positive integer.')
    if (
        isinstance(balance_check_interval, (bool, np.bool_))
        or not isinstance(balance_check_interval, (int, np.integer))
        or balance_check_interval < 0
    ):
        raise ValueError('balance_check_interval must be a nonnegative integer.')
    if not isinstance(stop_on_readout_tolerance, (bool, np.bool_)):
        raise ValueError('stop_on_readout_tolerance must be a boolean.')
    stop_on_readout_tolerance = bool(stop_on_readout_tolerance)
    if max_steps is not None:
        if not stop_on_readout_tolerance:
            raise ValueError('max_steps requires stop_on_readout_tolerance=True.')
        if isinstance(max_steps, (bool, np.bool_)) or not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
            raise ValueError('max_steps must be a positive integer or None.')
        max_steps = int(max_steps)
    n_steps, balance_check_interval = int(n_steps), int(balance_check_interval)
    lr_recurrent = _positive_learning_rate(learning_rate_recurrent, 'learning_rate_recurrent')
    if learning_rate_readout is not None:
        learning_rate_readout = _positive_learning_rate(learning_rate_readout, 'learning_rate_readout')
    if checkpoint_path is not None and Path(checkpoint_path).resolve() == Path(source_checkpoint).resolve():
        raise ValueError('The readout checkpoint must differ from the source checkpoint.')
    loaded = load_latent_network(source_checkpoint, training_path=training_path, device=device)
    rnn, dataset = loaded['rnn'], loaded['dataset']
    if not {'P1', 'A1'}.issubset(rnn.conditions):
        raise ValueError('Readout retraining requires a source network with P1 and A1 conditions.')
    if any(rnn.session_model(fx).Vout is not None for fx in rnn.session_ids):
        raise ValueError('Select an original latent-network checkpoint without a retrained readout.')
    delay_window = validate_delay_window(delay_window, min(rnn.n_time_by_session.values()))
    start, stop = delay_window
    optimizers, rates, calibrations = {}, {}, {}
    losses, readouts, balances = ({fx: [] for fx in rnn.session_ids} for _ in range(3))
    active_sessions = set(rnn.session_ids)
    thresholds, latest_losses = {}, {}
    converged = {fx: False for fx in rnn.session_ids}
    stop_steps = {fx: None for fx in rnn.session_ids}
    updates = {fx: 0 for fx in rnn.session_ids}
    rnn.train()
    for fx in rnn.session_ids:
        session = rnn.session_model(fx)
        with torch.no_grad():
            initial = session.initial_activity_from_targets(dataset['activity'][fx])
            noise = session.sample_input_timecourses()
            reference = session(initial_activity=initial, input_timecourses=noise)
            r_p1 = reference['P1'][start:stop].mean(dim=0)
            r_a1 = reference['A1'][start:stop].mean(dim=0)
            cd1 = r_p1 - r_a1
            if not torch.isfinite(cd1).all() or cd1.norm().item() == 0:
                raise ValueError(f'Session {fx}: CD1 must be finite and nonzero.')
            targets = torch.stack([cd1 @ r_a1, cd1 @ r_p1])
        session.initialize_readout(cd1, targets, noise, initial, delay_window)
        thresholds[fx] = 0.01 * (session.readout_targets[1] - session.readout_targets[0]).abs().item()
        if stop_on_readout_tolerance:
            if not np.isfinite(thresholds[fx]) or thresholds[fx] <= 0:
                raise ValueError(
                    f'Session {fx}: readout tolerance requires a finite, positive target gap; '
                    'loss cannot be strictly smaller than a zero threshold.'
                )
            with torch.no_grad():
                initial_predictions = session.readout()
                initial_loss = (initial_predictions - session.readout_targets).square().sum().item()
            if not np.isfinite(initial_loss):
                raise ValueError(f'Session {fx}: nonfinite initial readout loss.')
            latest_losses[fx] = initial_loss
            if initial_loss < thresholds[fx]:
                active_sessions.remove(fx)
                converged[fx], stop_steps[fx] = True, 0
                losses[fx].append(initial_loss)
                readouts[fx].append(initial_predictions.cpu().tolist())
                rates[fx] = learning_rate_readout
                print(f'session {fx} | already within tolerance | loss {initial_loss:.8g} '
                      f'< {thresholds[fx]:.8g} | 0 updates')
                continue
        if learning_rate_readout is None:
            try:
                rates[fx], calibrations[fx] = calibrate_readout_learning_rate(session, lr_recurrent)
            except ValueError as error:
                raise ValueError(f'Session {fx}: {error}') from error
        else:
            rates[fx] = learning_rate_readout
        optimizers[fx] = torch.optim.Adam([
            {'params': session.recurrent_parameters(), 'lr': lr_recurrent, 'betas':torch.tensor((0.9,0.999))},
            {'params': [session.Vout], 'lr': rates[fx]},
        ], weight_decay=0.0)
        print(f'session {fx} | recurrent lr {lr_recurrent:.6g} | readout lr {rates[fx]:.6g}')

    loss_history = []
    step_limit = max_steps if stop_on_readout_tolerance else n_steps
    step = 0
    while active_sessions and (step_limit is None or step < step_limit):
        step_loss = 0.0
        measure = bool(balance_check_interval) and (
            step % balance_check_interval == 0
            or (step_limit is not None and step == step_limit - 1)
        )
        for fx in rnn.session_ids:
            if fx not in active_sessions:
                step_loss += latest_losses[fx] / rnn.n_sessions
                continue
            session, optimizer = rnn.session_model(fx), optimizers[fx]
            optimizer.zero_grad(set_to_none=True)
            predictions = session.readout()
            loss = (predictions - session.readout_targets).square().sum()
            if not torch.isfinite(loss):
                raise ValueError(f'Session {fx}, step {step}: nonfinite readout loss.')
            value = loss.detach().cpu().item()
            losses[fx].append(value)
            readouts[fx].append(predictions.detach().cpu().tolist())
            latest_losses[fx] = value
            step_loss += value / rnn.n_sessions
            if stop_on_readout_tolerance and value < thresholds[fx]:
                active_sessions.remove(fx)
                converged[fx], stop_steps[fx] = True, updates[fx]
                print(f'session {fx} | reached tolerance | loss {value:.8g} '
                      f'< {thresholds[fx]:.8g} | {updates[fx]} updates')
                continue
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for group in optimizer.param_groups for p in group['params']):
                raise ValueError(f'Session {fx}, step {step}: nonfinite gradients.')
            effects = _readout_optimizer_step(session, optimizer, predictions, measure)
            updates[fx] += 1
            if effects is not None:
                balances[fx].append({'step': step, **effects})
        loss_history.append(step_loss)
        if step % 50 == 0 or not active_sessions or (step_limit is not None and step == step_limit - 1):
            progress = f' | active sessions {len(active_sessions)}/{rnn.n_sessions}' if stop_on_readout_tolerance else ''
            print(f'retraining step {step:04d} | readout loss {step_loss:.8g}{progress}')
        step += 1
    final_readouts, final_losses = {}, {}
    terminal_evaluation = stop_on_readout_tolerance and (bool(active_sessions) or not loss_history)
    with torch.no_grad():
        for fx in rnn.session_ids:
            session = rnn.session_model(fx)
            predictions = session.readout()
            final_readouts[fx] = predictions.cpu().tolist()
            final_losses[fx] = (predictions - session.readout_targets).square().sum().item()
            converged[fx] = final_losses[fx] < thresholds[fx]
            # A capped run needs a final check after its last permitted update.
            if stop_on_readout_tolerance and fx in active_sessions:
                losses[fx].append(final_losses[fx])
                readouts[fx].append(final_readouts[fx])
                if converged[fx]:
                    active_sessions.remove(fx)
                    stop_steps[fx] = updates[fx]
                session.zero_grad(set_to_none=True)
    if not all(np.isfinite(values).all() for values in final_readouts.values()):
        raise ValueError('Retraining produced nonfinite final readouts; reduce the learning rates.')
    if not all(np.isfinite(value) for value in final_losses.values()):
        raise ValueError('Retraining produced nonfinite final losses; reduce the learning rates.')
    if terminal_evaluation:
        loss_history.append(sum(final_losses[fx] / rnn.n_sessions for fx in rnn.session_ids))
    termination_reason = 'n_steps'
    if stop_on_readout_tolerance:
        termination_reason = 'all_sessions_converged' if all(converged.values()) else 'max_steps'
        print(f'Readout tolerance met by {sum(converged.values())}/{rnn.n_sessions} sessions.')
        if active_sessions:
            print(f'max_steps={max_steps} reached; unconverged sessions: {sorted(active_sessions)}')
    results = {
        'rnn': rnn, 'dataset': dataset, 'training_path': str(training_path),
        'loss_history': loss_history, 'loss_history_by_session': losses,
        'retraining_objective': 'readout_reversal',
        'readout_retraining': {
            'source_checkpoint': str(Path(source_checkpoint).resolve()),
            'delay_window': delay_window,
            'source_conditions': ['P1', 'A1'], 'readout_conditions': ['P2', 'A2'],
            'learning_rate_recurrent': lr_recurrent,
            'learning_rate_readout_by_session': rates,
            'calibration_by_session': calibrations,
            'stop_on_readout_tolerance': stop_on_readout_tolerance,
            'max_steps': max_steps,
            'threshold_by_session': thresholds,
            'converged_by_session': converged,
            'stop_step_by_session': stop_steps,
            'updates_by_session': updates,
            'final_loss_by_session': final_losses,
            'termination_reason': termination_reason,
            'balance_check_interval': balance_check_interval,
            'balance_history_by_session': balances,
            'readout_history_by_session': readouts,
            'final_readouts_by_session': final_readouts,
            'optimizer_states_by_session': {
                fx: optimizers[fx].state_dict() if fx in optimizers else None
                for fx in rnn.session_ids
            },
        },
    }
    save_latent_network(results, save_path=save_path, checkpoint_path=checkpoint_path)
    return results


# Readout checkpoints also contain their fixed simulation inputs and references.
def save_latent_network(results, save_path=None, checkpoint_path=None):
    rnn = results['rnn']
    dataset = results['dataset']
    readout_metadata = results.get('readout_retraining')
    has_readout = [rnn.session_model(fx).Vout is not None for fx in rnn.session_ids]
    if any(has_readout) and (not all(has_readout) or readout_metadata is None):
        raise ValueError('Readout checkpoints require all session readouts and readout_retraining metadata.')
    if readout_metadata is not None and not all(has_readout):
        raise ValueError('Readout retraining metadata requires a readout for every session.')
    if checkpoint_path is None and readout_metadata is not None:
        root = None if save_path is None else Path(save_path) / rnn.recurrent_type
        checkpoint_path = readout_retrained_checkpoint_path(
            readout_metadata['source_checkpoint'], save_path=root,
        )
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
    Readout checkpoints restore Vout, fixed references, noise, and prefixes;
    rnn.readout(fx) reuses these saved inputs to reproduce the trained readouts.
    Versions 2/3 retain their fixed-zero initial states. New learned-state models
    require initial_activity when simulated directly; plotting helpers extract it.
    """
    if checkpoint_path is None:
        checkpoint_path = latent_network_checkpoint_path(training_path=training_path)
    device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get('version') not in (2, 3, CHECKPOINT_VERSION, READOUT_CHECKPOINT_VERSION):
        raise ValueError('Unsupported checkpoint format; train a new session network.')
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
    readout_metadata = checkpoint.get('readout_retraining')
    if checkpoint['version'] == READOUT_CHECKPOINT_VERSION:
        if readout_metadata is None:
            raise ValueError('Readout checkpoint is missing readout_retraining metadata.')
        state = checkpoint['rnn_state_dict']
        for fx in rnn.session_ids:
            prefix = f'sessions.{fx}.'
            rnn.session_model(fx).initialize_readout(
                state[prefix + 'CD1'], state[prefix + 'readout_targets'],
                state[prefix + 'readout_input_timecourses'],
                state.get(prefix + 'readout_initial_activity'),
                readout_metadata['delay_window'],
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
    if readout_metadata is not None:
        results['readout_retraining'] = readout_metadata
        results['retraining_objective'] = 'readout_reversal'
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
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_recurrent', type=float,
                        help='Readout retraining: recurrent rate (default: --learning_rate).')
    parser.add_argument('--learning_rate_readout', type=float,
                        help='Readout retraining: Vout rate (default: calibrate per session).')
    parser.add_argument('--delay_window', nargs=2, type=int, default=DEFAULT_DELAY_WINDOW,
                        metavar=('START', 'STOP'), help='Readout window, stop excluded (default: 12 16).')
    parser.add_argument('--balance_check_interval', type=int, default=50,
                        help='Readout-effect diagnostic interval; 0 disables it (default: 50).')
    parser.add_argument('--stop_on_readout_tolerance', action='store_true',
                        help='Stop each readout-retraining session when loss < 0.01 * abs(target gap); '
                             'continue until all sessions converge, ignoring --n_steps.')
    parser.add_argument('--max_steps', type=int,
                        help='Optional update cap per session with --stop_on_readout_tolerance.')
    mode.add_argument('--retrain_readout', action='store_true',
                      help='Retrain recurrence and Vout against reversed fixed CD1 readouts.')
    parser.add_argument('--source_checkpoint',
                        help='Source for --retrain_readout (default: selected model configuration).')
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
    parser.add_argument('--input_noise_std', type=float, default=DEFAULT_INPUT_NOISE_STD)
    parser.add_argument(
        '--n_initial_timepoints', type=int,
        help='Observed target bins used to learn the initial state (default: 2).',
    )
    parser.add_argument('--input_masks_path', help='Optional numeric .npy masks of shape (types, time).')
    parser.add_argument('--conditions', nargs='+',
                        help='Training targets (default: P1 A1).')
    parser.add_argument('--fx', type=int, help='Session to plot after training (default: first trained session).')
    parser.add_argument('--condition', help='Condition to plot after training (default: first trained condition).')
    parser.add_argument('--dt', type=float, default=DT)
    mode.add_argument('--load_only', action='store_true')
    parser.add_argument('--checkpoint_path',
                        help='Output file for training, or input file with --load_only.')
    args = parser.parse_args()
    if args.source_checkpoint and not args.retrain_readout:
        parser.error('--source_checkpoint requires --retrain_readout.')
    if (args.stop_on_readout_tolerance or args.max_steps is not None) and not args.retrain_readout:
        parser.error('--stop_on_readout_tolerance and --max_steps require --retrain_readout.')
    if args.max_steps is not None and not args.stop_on_readout_tolerance:
        parser.error('--max_steps requires --stop_on_readout_tolerance.')
    if args.list_datasets:
        print('Available dataset IDs:', ', '.join(available_training_datasets(args.training_path)))
    elif args.load_only:
        checkpoint_path = args.checkpoint_path or latent_network_checkpoint_path(
            args.dataset_id, args.n_input_types, args.n_input_components,
            args.input_noise_std, args.save_path,
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
        source_checkpoint = args.source_checkpoint or latent_network_checkpoint_path(
            args.dataset_id, args.n_input_types, args.n_input_components,
            args.input_noise_std,
            recurrent_type=args.recurrent_type, recurrent_rank=args.recurrent_rank,
            training_path=args.training_path,
        )
        retrained_results = retrain_latent_network_readout(
            source_checkpoint=source_checkpoint, n_steps=args.n_steps,
            learning_rate_recurrent=(
                args.learning_rate if args.learning_rate_recurrent is None else args.learning_rate_recurrent
            ),
            learning_rate_readout=args.learning_rate_readout,
            delay_window=args.delay_window, balance_check_interval=args.balance_check_interval,
            stop_on_readout_tolerance=args.stop_on_readout_tolerance, max_steps=args.max_steps,
            training_path=args.training_path, save_path=args.save_path,
            checkpoint_path=args.checkpoint_path, device=args.device,
        )
    else:
        input_masks = np.load(args.input_masks_path) if args.input_masks_path else None
        trained_results = train_latent_network(
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            dataset_id=args.dataset_id,
            n_input_types=args.n_input_types,
            n_input_components=args.n_input_components,
            input_noise_std=args.input_noise_std,
            conditions=args.conditions,
            dt=args.dt,
            input_masks=input_masks,
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
