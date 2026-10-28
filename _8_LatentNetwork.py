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


def save_latent_network(results, save_path=None, checkpoint_path=None):
    rnn = results['rnn']
    dataset = results['dataset']
    if checkpoint_path is None:
        checkpoint_path = latent_network_checkpoint_path(
            dataset['id'], rnn.n_input_types, rnn.n_input_components,
            rnn.input_noise_std, save_path,
            recurrent_type=rnn.recurrent_type,
            recurrent_rank=rnn.recurrent_rank,
            training_path=results.get('training_path', TRAINING_DATA_PATH),
        )
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'version': CHECKPOINT_VERSION,
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
    Versions 2/3 retain their fixed-zero initial states. New learned-state models
    require initial_activity when simulated directly; plotting helpers extract it.
    """
    if checkpoint_path is None:
        checkpoint_path = latent_network_checkpoint_path(training_path=training_path)
    device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get('version') not in (2, 3, CHECKPOINT_VERSION):
        raise ValueError('Unsupported checkpoint format; expected a version 2, 3, or 4 latent-network checkpoint.')
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
    rnn.load_state_dict(checkpoint['rnn_state_dict'])
    results = {
        'rnn': rnn,
        'dataset': dataset,
        'loss_history': checkpoint['loss_history'],
        'loss_history_by_session': checkpoint.get('loss_history_by_session', {}),
        'checkpoint_path': str(checkpoint_path),
        'training_path': str(training_path),
    }
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
    parser.add_argument('--conditions', nargs='+',
                        help='Training targets (default: P1 A1).')
    parser.add_argument('--fx', type=int, help='Session to plot after training (default: first trained session).')
    parser.add_argument('--condition', help='Condition to plot after training (default: first trained condition).')
    parser.add_argument('--dt', type=float, default=DT)
    mode.add_argument('--load_only', action='store_true')
    parser.add_argument('--checkpoint_path',
                        help='Output file for training, or input file with --load_only.')
    args = parser.parse_args()
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
    else:
        trained_results = train_latent_network(
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            dataset_id=args.dataset_id,
            n_input_types=args.n_input_types,
            n_input_components=args.n_input_components,
            input_noise_std=args.input_noise_std,
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
