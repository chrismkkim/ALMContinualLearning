import glob
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Step 1. Define the latent RNN units
# ==============================================================
#
# Unit order:
#   S Group0, S Group1, S Group2, S Group3,
#   D Group0, D Group1, D Group2, D Group3,
#   R Group0, R Group1, R Group2, R Group3
#
# Each type/group pair has N_UNITS_PER_GROUP units.

SAVE_FIGURE_PATH = 'figure/latent_network/'
SAVE_LATENT_PATH = 'data/latent_network/'
TRAINING_DATA_PATH = 'data/training'
NEURON_TYPES = ['S', 'D', 'R']
GROUPS = ['Group0', 'Group1', 'Group2', 'Group3']
# CONDITIONS = ['P1', 'A1', 'P2', 'A2']
CONDITIONS = ['P1', 'A1']
N_UNITS_PER_GROUP = 10 #10
DEFAULT_RECURRENT_RANK = 10
DEFAULT_READOUT_RANK = 5
DEFAULT_USE_RECURRENT_V_SESSION = True
DEFAULT_RECURRENT_V_SESSION_REGULARIZATION = 1e-3
DEFAULT_MAX_RECURRENT_V_SESSION_FRACTION = 0.1
DEFAULT_N_INPUT_COMPONENTS = 1
DEFAULT_USE_INPUT_U_SESSION = True
DEFAULT_INPUT_U_SESSION_REGULARIZATION = 1e-3
DEFAULT_MAX_INPUT_U_SESSION_FRACTION = 0.1
DEFAULT_DEVICE = 'cpu'
DT = 0.1


def resolve_device(device=DEFAULT_DEVICE):
    """Return a torch.device after validating requested CPU/GPU support."""
    if isinstance(device, torch.device):
        requested = str(device)
    else:
        requested = str(device).strip().lower()

    if requested == 'gpu':
        if torch.cuda.is_available():
            requested = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            requested = 'mps'
        else:
            raise ValueError(
                'GPU was requested, but neither CUDA nor MPS is available.'
            )

    resolved = torch.device(requested)

    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError(
            f'CUDA device "{resolved}" was requested, but CUDA is not available.'
        )
    if resolved.type == 'mps':
        mps_available = (
            hasattr(torch.backends, 'mps')
            and torch.backends.mps.is_available()
        )
        if not mps_available:
            raise ValueError(
                f'MPS device "{resolved}" was requested, but MPS is not available.'
            )

    return resolved


def make_unit_slices(n_units_per_group=N_UNITS_PER_GROUP):
    """Map each (neuron type, group) pair to its columns in the RNN state."""
    unit_slices = {}
    col = 0

    for neuron_type in NEURON_TYPES:
        for group in GROUPS:
            unit_slices[(neuron_type, group)] = slice(
                col,
                col + n_units_per_group
            )
            col += n_units_per_group

    return unit_slices, col


UNIT_SLICES, N_RNN_UNITS = make_unit_slices()


class LatentRNN(nn.Module):
    """
    Latent RNN:

        dh/dt = -h + W*g(h) + I

    Here g is ReLU. The external input is a sum of trainable rank-1
    condition-specific components I_k * u_k(t).
    """

    def __init__(
        self,
        n_units,
        n_time,
        n_sessions,
        recurrent_rank=DEFAULT_RECURRENT_RANK,
        use_recurrent_v_session=DEFAULT_USE_RECURRENT_V_SESSION,
        max_recurrent_v_session_fraction=DEFAULT_MAX_RECURRENT_V_SESSION_FRACTION,
        n_input_components=DEFAULT_N_INPUT_COMPONENTS,
        use_input_u_session=DEFAULT_USE_INPUT_U_SESSION,
        max_input_u_session_fraction=DEFAULT_MAX_INPUT_U_SESSION_FRACTION,
        n_conditions=len(CONDITIONS),
        dt=DT,
    ):
        super().__init__()

        self.n_units = n_units
        self.n_time = n_time
        self.n_sessions = n_sessions
        self.recurrent_rank = recurrent_rank
        self.use_recurrent_v_session = use_recurrent_v_session
        self.max_recurrent_v_session_fraction = max_recurrent_v_session_fraction
        self.n_input_components = n_input_components
        self.use_input_u_session = use_input_u_session
        self.max_input_u_session_fraction = max_input_u_session_fraction
        self.dt = dt

        if recurrent_rank < 1 or recurrent_rank > n_units:
            raise ValueError(
                f'recurrent_rank must be between 1 and {n_units}, '
                f'got {recurrent_rank}.'
            )
        if max_recurrent_v_session_fraction <= 0:
            raise ValueError(
                f'max_recurrent_v_session_fraction must be positive, '
                f'got {max_recurrent_v_session_fraction}.'
            )
        if n_input_components < 1:
            raise ValueError(
                f'n_input_components must be at least 1, '
                f'got {n_input_components}.'
            )
        if max_input_u_session_fraction <= 0:
            raise ValueError(
                f'max_input_u_session_fraction must be positive, '
                f'got {max_input_u_session_fraction}.'
            )

        # Sessions share U and the dominant V factor. Optionally, each session
        # gets a capped additive perturbation to V.
        factor_scale = np.sqrt(0.05 / np.sqrt(recurrent_rank))
        self.U = nn.Parameter(
            factor_scale * torch.randn(n_units, recurrent_rank)
        )
        self.V_shared = nn.Parameter(
            factor_scale * torch.randn(n_units, recurrent_rank)
        )
        if self.use_recurrent_v_session:
            self.V_session = nn.Parameter(
                0.01
                * factor_scale
                * torch.randn(n_sessions, n_units, recurrent_rank)
            )
        self.register_buffer('h0', torch.zeros(n_conditions, n_units))

        input_factor_scale = np.sqrt(0.01 / n_input_components)
        self.input_vectors = nn.Parameter(
            input_factor_scale
            * torch.randn(n_conditions, n_input_components, n_units)
        )
        self.u_shared = nn.Parameter(
            input_factor_scale
            * torch.randn(n_conditions, n_input_components, n_time)
        )
        if self.use_input_u_session:
            self.u_session = nn.Parameter(
                0.01
                * input_factor_scale
                * torch.randn(
                    n_sessions,
                    n_conditions,
                    n_input_components,
                    n_time
                )
            )

        # The external input is trainable only in the first and last thirds.
        # Multiplying by zero in the middle third blocks gradients to
        # the factorized input over input_mask == 0.
        input_mask = torch.zeros(n_time)
        input_mask[: n_time // 3] = 1.0
        input_mask[2 * n_time // 3:] = 1.0
        self.register_buffer('input_mask', input_mask)

    @property
    def V(self):
        """Backward-compatible alias for the shared recurrent V factor."""
        return self.V_shared

    @property
    def use_session_v(self):
        """Backward-compatible alias for recurrent V-session usage."""
        return self.use_recurrent_v_session

    def effective_recurrent_v_session(self, fx):
        """
        Return the capped session-specific V perturbation.

        The cap enforces:
            ||V_session_fx|| <=
                max_recurrent_v_session_fraction * ||V_shared||
        """
        if not self.use_recurrent_v_session:
            return torch.zeros_like(self.V_shared)

        raw_delta = self.V_session[int(fx)]
        raw_norm = raw_delta.norm()
        max_norm = (
            self.max_recurrent_v_session_fraction
            * self.V_shared.norm().detach()
        )
        max_norm = max_norm.clamp_min(1e-8)
        scale = torch.clamp(max_norm / raw_norm.clamp_min(1e-8), max=1.0)

        return raw_delta * scale

    def recurrent_v(self, fx=None):
        """Return the shared V plus the optional session-specific perturbation."""
        if not self.use_recurrent_v_session:
            return self.V_shared
        if fx is None:
            raise ValueError('fx is required when session-specific V is enabled.')

        return self.V_shared + self.effective_recurrent_v_session(fx)

    def recurrent_v_session_norm_ratios(self):
        """
        Return ||V_session_fx|| / ||V_shared|| for each session.

        The uncapped perturbation is used so the regularizer discourages the
        learned session term from growing past the hard cap.
        """
        if not self.use_recurrent_v_session:
            return self.V_shared.new_zeros(0)

        shared_norm = self.V_shared.norm().detach().clamp_min(1e-8)
        return self.V_session.flatten(start_dim=1).norm(dim=1) / shared_norm

    def recurrent_v_session_regularization_loss(self):
        """Penalize the relative size of the session-specific V component."""
        ratios = self.recurrent_v_session_norm_ratios()
        if ratios.numel() == 0:
            return self.V_shared.new_tensor(0.0)

        return ratios.pow(2).mean()

    def project_recurrent_v_session_(self):
        """Project session-specific V parameters to satisfy the norm cap."""
        if not self.use_recurrent_v_session:
            return

        with torch.no_grad():
            max_norm = (
                self.max_recurrent_v_session_fraction
                * self.V_shared.norm().detach().clamp_min(1e-8)
            )
            raw_norms = self.V_session.flatten(start_dim=1).norm(dim=1)
            scales = torch.clamp(
                max_norm / raw_norms.clamp_min(1e-8),
                max=1.0
            )
            self.V_session.mul_(scales.view(-1, 1, 1))

    def input_u(self, fx, icond):
        """Return condition time courses u_shared + optional u_session."""
        u = self.u_shared[icond]
        if self.use_input_u_session:
            u = u + self.effective_input_u_session(fx, icond)

        return u

    def effective_input_u_session(self, fx, icond):
        """
        Return the capped session-specific input time-course perturbation.

        The cap enforces:
            ||u_session_fx|| <=
                max_input_u_session_fraction * ||u_shared||
        """
        if not self.use_input_u_session:
            return torch.zeros_like(self.u_shared[icond])

        raw_delta = self.u_session[int(fx), icond]
        raw_norm = raw_delta.norm()
        max_norm = (
            self.max_input_u_session_fraction
            * self.u_shared.norm().detach()
        )
        max_norm = max_norm.clamp_min(1e-8)
        scale = torch.clamp(max_norm / raw_norm.clamp_min(1e-8), max=1.0)

        return raw_delta * scale

    def condition_input(self, fx, icond):
        """
        Return masked external input over time for one session and condition.

        Shape:
            (n_time, n_units)
        """
        input_over_time = torch.einsum(
            'ku,kt->tu',
            self.input_vectors[icond],
            self.input_u(fx, icond)
        )

        return input_over_time * self.input_mask[:, None]

    def input_u_session_norm_ratios(self):
        """
        Return ||u_session_fx|| / ||u_shared|| for each session.

        The uncapped perturbation is used so the regularizer discourages the
        learned session term from growing past the hard cap.
        """
        if not self.use_input_u_session:
            return self.u_shared.new_zeros(0)

        shared_norm = self.u_shared.norm().detach().clamp_min(1e-8)
        return self.u_session.flatten(start_dim=1).norm(dim=1) / shared_norm

    def input_u_session_regularization_loss(self):
        """Penalize the relative size of session-specific input time courses."""
        ratios = self.input_u_session_norm_ratios()
        if ratios.numel() == 0:
            return self.u_shared.new_tensor(0.0)

        return ratios.pow(2).mean()

    def project_input_u_session_(self):
        """Project session-specific input time courses to satisfy the norm cap."""
        if not self.use_input_u_session:
            return

        with torch.no_grad():
            max_norm = (
                self.max_input_u_session_fraction
                * self.u_shared.norm().detach().clamp_min(1e-8)
            )
            raw_norms = self.u_session.flatten(start_dim=1).norm(dim=1)
            scales = torch.clamp(
                max_norm / raw_norms.clamp_min(1e-8),
                max=1.0
            )
            self.u_session.mul_(scales.view(-1, 1, 1, 1))

    def recurrent_weight(self, fx=None):
        """
        Return the rank-constrained recurrent matrix for one session.
        """
        V = self.recurrent_v(fx)
        return self.U @ V.T
        
    def forward(self, fx):
        """
        Return rectified latent activity g(h) for one session.

        Shape:
            g_by_condition[condition] = (n_time, n_units)
        """
        W = self.recurrent_weight(fx)
        middle_start = self.n_time // 3
        middle_end = 2 * self.n_time // 3
        g_by_condition = {}

        for icond, condition in enumerate(CONDITIONS):
            h = self.h0[icond]
            input_over_time = self.condition_input(fx, icond)
            h_over_time = []

            for t in range(self.n_time):
                # if t == middle_start or t == middle_end:
                #     h = h.detach()

                h_over_time.append(h)
                g_h = F.relu(h)
                I_t = input_over_time[t]

                if middle_start <= t < middle_end:
                    W_t = W
                else:
                    W_t = W.detach()

                dh = -h + g_h @ W_t.T + I_t
                h = h + self.dt * dh

            h_over_time = torch.stack(h_over_time, dim=0)
            g_by_condition[condition] = F.relu(h_over_time)

        return g_by_condition


# ==============================================================
# Step 2. Session-specific total readout matrices
# ==============================================================


class SessionReadouts(nn.Module):
    """
    One low-rank total readout matrix per session.

    total_readout(fx) has shape:
        (number of original neurons in session fx, number of RNN units)

    For each session, the full matrix is reconstructed by concatenating
    three population-specific low-rank blocks:

        [U_S_fx @ V_S, U_D_fx @ V_D, U_R_fx @ V_R]

    U_S_fx, U_D_fx, and U_R_fx are session-specific. V_S, V_D, and V_R
    are shared across sessions. Each factor is passed through softplus
    before multiplication, so every reconstructed readout entry is positive.

    Most rows/columns are never used directly in a training loss. The
    helper build_actual_readout selects top-cell rows and masks columns.
    """

    def __init__(self, cells, readout_rank=DEFAULT_READOUT_RANK):
        super().__init__()

        self.readout_rank = readout_rank
        self.n_units_per_population = len(GROUPS) * N_UNITS_PER_GROUP

        if readout_rank < 1:
            raise ValueError(f'readout_rank must be at least 1, got {readout_rank}.')

        # Initialize positive factors so each population block starts near
        # the previous dense readout scale of 0.01.
        init_factor = np.sqrt(0.01 / readout_rank)
        raw_init_factor = torch.log(torch.expm1(torch.tensor(init_factor)))

        self.raw_V_by_type = nn.ParameterDict({
            neuron_type: nn.Parameter(
                raw_init_factor
                + 0.01 * torch.randn(readout_rank, self.n_units_per_population)
            )
            for neuron_type in NEURON_TYPES
        })

        self.raw_U_by_session_type = nn.ModuleDict()

        for fx in sorted(cells):
            n_og_cells = get_num_original_cells(cells[fx])
            raw_U_by_type_fx = nn.ParameterDict({
                neuron_type: nn.Parameter(
                    raw_init_factor
                    + 0.01 * torch.randn(n_og_cells, readout_rank)
                )
                for neuron_type in NEURON_TYPES
            })
            self.raw_U_by_session_type[str(fx)] = raw_U_by_type_fx

    def total_readout(self, fx):
        readout_blocks = []

        for neuron_type in NEURON_TYPES:
            U = F.softplus(
                self.raw_U_by_session_type[str(fx)][neuron_type]
            )
            V = F.softplus(self.raw_V_by_type[neuron_type])
            readout_blocks.append(U @ V)

        return torch.cat(readout_blocks, dim=1)


def get_num_original_cells(cells_fx):
    """
    Read the number of original neurons in a session.

    This value sets the number of rows in the session's total readout
    matrix. Top-cell nonoutlier indices are converted into these original
    neuron-row indices later in build_actual_readout.
    """
    return int(np.asarray(cells_fx['num_og_cells']).item())


# ==============================================================
# Step 3-4. Build actual masked readout matrices from top cells
# ==============================================================


def top_cells_in_training_order(cells_fx):
    """
    Return top cells in the same order as activity*.npy.

    The activity files are built as:
        S Group0..3, then D Group0..3, then R Group0..3.

    Each returned item contains:
        nonoutlier_idx : index into nonoutlier-normalized data
        original_idx   : row index in the total readout matrix
        neuron_type    : S, D, or R
        group          : Group0..Group3
    """
    nonoutlier = np.asarray(cells_fx['nonoutlier'], dtype=int)
    rows = []

    for neuron_type in NEURON_TYPES:
        for group in GROUPS:
            for nonoutlier_idx in np.asarray(
                cells_fx[neuron_type][group],
                dtype=int
            ):
                rows.append({
                    'nonoutlier_idx': int(nonoutlier_idx),
                    'original_idx': int(nonoutlier[nonoutlier_idx]),
                    'neuron_type': neuron_type,
                    'group': group,
                })

    return rows


def build_readout_mask(cell_rows, device):
    """
    Make a binary mask for the actual readout matrix.

    For a top cell belonging to (S, Group1), only the columns for
    (S, Group1) units are trainable; all other columns are masked to zero.
    """
    mask = torch.zeros(len(cell_rows), N_RNN_UNITS, device=device)

    for irow, row in enumerate(cell_rows):
        unit_slice = UNIT_SLICES[(row['neuron_type'], row['group'])]
        mask[irow, unit_slice] = 1.0

    return mask


def build_actual_readout(total_readout_fx, cells_fx):
    """
    Select top-cell rows from a session's total readout matrix and mask
    each row to its allowed (neuron type, group) unit block.

    This function is called during every training step. Because the actual
    readout is built from total_readout_fx using differentiable indexing
    and multiplication by a binary mask, backprop updates only the selected
    rows and only the unmasked columns. total_readout_fx is already positive
    because SessionReadouts.total_readout applies softplus; masked columns
    remain exactly zero.
    """
    device = total_readout_fx.device
    cell_rows = top_cells_in_training_order(cells_fx)

    original_rows = torch.tensor(
        [row['original_idx'] for row in cell_rows],
        dtype=torch.long,
        device=device
    )
    mask = build_readout_mask(cell_rows, device)

    actual_readout = total_readout_fx[original_rows] * mask
    return actual_readout, cell_rows


# ==============================================================
# Training data utilities
# ==============================================================


def load_training_data(training_path=TRAINING_DATA_PATH):
    """Load all cells*.npy and matching activity*.npy files."""
    cell_files = sorted(glob.glob(os.path.join(training_path, 'cells*.npy')))
    datasets = []

    for cell_file in cell_files:
        data_id = os.path.basename(cell_file).replace('cells', '').replace('.npy', '')
        activity_file = os.path.join(training_path, f'activity{data_id}.npy')

        if not os.path.exists(activity_file):
            continue

        datasets.append({
            'cells': np.load(cell_file, allow_pickle=True).item(),
            'activity': np.load(activity_file, allow_pickle=True).item(),
            'id': data_id,
        })

    if len(datasets) == 0:
        raise FileNotFoundError(f'No cells/activity files found in {training_path}')

    return datasets


def infer_n_time(datasets):
    """Use the first session/condition to infer number of time bins."""
    first_activity = datasets[0]['activity']
    first_fx = sorted(first_activity)[0]
    return first_activity[first_fx][CONDITIONS[0]].shape[1]


def check_activity_matches_cells(cells_fx, activity_fx):
    """Sanity check: rows in activity should match selected top-cell rows."""
    n_cells = len(top_cells_in_training_order(cells_fx))

    for condition in CONDITIONS:
        if activity_fx[condition].shape[0] != n_cells:
            raise ValueError(
                f'{condition}: activity has {activity_fx[condition].shape[0]} '
                f'rows, but cells specify {n_cells} top-cell rows.'
            )


# ==============================================================
# Step 5. Training loop
# ==============================================================


def train_latent_network(
    n_steps=200,
    learning_rate=1e-3,
    recurrent_rank=DEFAULT_RECURRENT_RANK,
    use_recurrent_v_session=DEFAULT_USE_RECURRENT_V_SESSION,
    recurrent_v_session_regularization=DEFAULT_RECURRENT_V_SESSION_REGULARIZATION,
    max_recurrent_v_session_fraction=DEFAULT_MAX_RECURRENT_V_SESSION_FRACTION,
    n_input_components=DEFAULT_N_INPUT_COMPONENTS,
    use_input_u_session=DEFAULT_USE_INPUT_U_SESSION,
    input_u_session_regularization=DEFAULT_INPUT_U_SESSION_REGULARIZATION,
    max_input_u_session_fraction=DEFAULT_MAX_INPUT_U_SESSION_FRACTION,
    readout_rank=DEFAULT_READOUT_RANK,
    training_path=TRAINING_DATA_PATH,
    save_path=SAVE_LATENT_PATH,
    device=DEFAULT_DEVICE,
):
    """
    Train low-rank recurrent matrices and the session readouts.

    At every optimizer step:
        1. Simulate the latent RNN for each session to get g(h).
        2. For each dataset/session, rebuild the actual readout from the
           current total readout matrix and the current cells*.npy metadata.
        3. Predict neural activity: actual_readout @ g(h).T
        4. Backpropagate MSE loss.

    Updating the actual readout automatically updates the relevant rows and
    columns of the total readout matrix because actual_readout is a masked
    view/computation from total_readout.
    """
    datasets = load_training_data(training_path)
    n_time = infer_n_time(datasets)
    n_sessions = len(datasets[0]['cells'])
    device = resolve_device(device)

    rnn = LatentRNN(
        N_RNN_UNITS,
        n_time,
        n_sessions,
        recurrent_rank=recurrent_rank,
        use_recurrent_v_session=use_recurrent_v_session,
        max_recurrent_v_session_fraction=max_recurrent_v_session_fraction,
        n_input_components=n_input_components,
        use_input_u_session=use_input_u_session,
        max_input_u_session_fraction=max_input_u_session_fraction
    ).to(device)

    # One readout model per independently resampled training dataset.
    # Each readout model contains session-specific U factors and shared
    # S/D/R V factors.
    readouts_by_dataset = nn.ModuleList([
        SessionReadouts(dataset['cells'], readout_rank=readout_rank)
        for dataset in datasets
    ]).to(device)

    optimizer = torch.optim.Adam(
        list(rnn.parameters()) + list(readouts_by_dataset.parameters()),
        lr=learning_rate
    )

    loss_history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        loss = torch.tensor(0.0, device=device)
        n_terms = 0

        for idata, dataset in enumerate(datasets):
            cells = dataset['cells']
            activity = dataset['activity']
            readouts = readouts_by_dataset[idata]

            for fx in sorted(cells):
                g_by_condition = rnn(fx)
                check_activity_matches_cells(cells[fx], activity[fx])

                actual_readout, _ = build_actual_readout(
                    readouts.total_readout(fx),
                    cells[fx]
                )

                for condition in CONDITIONS:
                    target = torch.tensor(
                        activity[fx][condition],
                        dtype=torch.float32,
                        device=device
                    )

                    # actual_readout: (top cells, RNN units)
                    # g_by_condition[condition].T: (RNN units, time)
                    # prediction: (top cells, time)
                    prediction = actual_readout @ g_by_condition[condition].T
                    loss = loss + F.mse_loss(prediction, target)
                    n_terms += 1

        loss = loss / n_terms
        if use_recurrent_v_session and recurrent_v_session_regularization > 0:
            loss = (
                loss
                + recurrent_v_session_regularization
                * rnn.recurrent_v_session_regularization_loss()
            )
        if use_input_u_session and input_u_session_regularization > 0:
            loss = (
                loss
                + input_u_session_regularization
                * rnn.input_u_session_regularization_loss()
            )
        loss.backward()
        optimizer.step()
        rnn.project_recurrent_v_session_()
        rnn.project_input_u_session_()

        loss_history.append(float(loss.detach().cpu()))

        if step % 50 == 0:
            print(f'step {step:04d} | loss {loss_history[-1]:.6f}')

    results = {
        'rnn': rnn,
        'readouts_by_dataset': readouts_by_dataset,
        'loss_history': loss_history,
        'datasets': datasets,
        'recurrent_v_session_regularization': recurrent_v_session_regularization,
        'input_u_session_regularization': input_u_session_regularization,
    }
    save_latent_network(results, save_path=save_path)
    return results


# ==============================================================
# Step 6. Save and load trained latent-network parameters
# ==============================================================


def save_latent_network(results, save_path=SAVE_LATENT_PATH):
    """
    Save all trained network parameters needed to restore the model.

    This includes:
        - RNN parameters: low-rank W factors and factorized external inputs
        - low-rank readout factors for every dataset/session
        - loss history and small model metadata
    """
    os.makedirs(save_path, exist_ok=True)
    checkpoint_path = os.path.join(save_path, 'latent_network.pt')

    checkpoint = {
        'rnn_state_dict': results['rnn'].state_dict(),
        'readouts_state_dict': results['readouts_by_dataset'].state_dict(),
        'loss_history': results['loss_history'],
        'dataset_ids': [dataset['id'] for dataset in results['datasets']],
        'config': {
            'n_rnn_units': N_RNN_UNITS,
            'n_units_per_group': N_UNITS_PER_GROUP,
            'n_time': results['rnn'].n_time,
            'n_sessions': results['rnn'].n_sessions,
            'recurrent_rank': results['rnn'].recurrent_rank,
            'use_recurrent_v_session': results['rnn'].use_recurrent_v_session,
            'recurrent_v_session_regularization': results.get(
                'recurrent_v_session_regularization',
                DEFAULT_RECURRENT_V_SESSION_REGULARIZATION
            ),
            'max_recurrent_v_session_fraction': (
                results['rnn'].max_recurrent_v_session_fraction
            ),
            'n_input_components': results['rnn'].n_input_components,
            'use_input_u_session': results['rnn'].use_input_u_session,
            'input_u_session_regularization': results.get(
                'input_u_session_regularization',
                DEFAULT_INPUT_U_SESSION_REGULARIZATION
            ),
            'max_input_u_session_fraction': (
                results['rnn'].max_input_u_session_fraction
            ),
            'readout_rank': results['readouts_by_dataset'][0].readout_rank,
            'dt': results['rnn'].dt,
            'neuron_types': NEURON_TYPES,
            'groups': GROUPS,
            'conditions': CONDITIONS,
            'shared_recurrent_u': True,
            'shared_recurrent_weight': (
                not results['rnn'].use_recurrent_v_session
            ),
        },
    }

    torch.save(checkpoint, checkpoint_path)
    results['checkpoint_path'] = checkpoint_path
    print(f'Saved latent network to {checkpoint_path}')


def convert_legacy_rnn_state_dict(rnn_state_dict, rnn):
    """
    Convert older checkpoints that stored session-specific recurrent weights.

    Older checkpoints may have full W_by_session matrices or low-rank
    U_by_session/V_by_session factors. Those session-specific recurrent
    matrices are averaged into one shared W, then projected into the requested
    low-rank form using a truncated SVD:
        W ~= U @ V.T
    """
    converted = {
        key: value
        for key, value in rnn_state_dict.items()
        if (
            not key.startswith('W_by_session.')
            and not key.startswith('U_by_session.')
            and not key.startswith('V_by_session.')
        )
    }
    if 'V' in converted and 'V_shared' not in converted:
        converted['V_shared'] = converted.pop('V')

    W_list = []

    for fx in range(rnn.n_sessions):
        legacy_key = f'W_by_session.{fx}'
        legacy_u_key = f'U_by_session.{fx}'
        legacy_v_key = f'V_by_session.{fx}'

        if legacy_key in rnn_state_dict:
            W_list.append(rnn_state_dict[legacy_key])
        elif legacy_u_key in rnn_state_dict and legacy_v_key in rnn_state_dict:
            W_list.append(
                rnn_state_dict[legacy_u_key] @ rnn_state_dict[legacy_v_key].T
            )

    if len(W_list) > 0:
        W = torch.stack(W_list, dim=0).mean(dim=0)
        U_svd, S_svd, Vh_svd = torch.linalg.svd(W, full_matrices=False)
        rank = rnn.recurrent_rank
        sqrt_s = torch.sqrt(S_svd[:rank])
        converted['U'] = U_svd[:, :rank] * sqrt_s
        converted['V_shared'] = Vh_svd[:rank, :].T * sqrt_s

    return converted


def rename_legacy_shared_v(rnn_state_dict):
    """Rename old shared recurrent V key to the current V_shared key."""
    if 'V' not in rnn_state_dict or 'V_shared' in rnn_state_dict:
        return rnn_state_dict

    renamed = dict(rnn_state_dict)
    renamed['V_shared'] = renamed.pop('V')
    return renamed


def prepare_rnn_state_dict_for_load(rnn_state_dict):
    """
    Drop obsolete trainable initial/input tensors before loading.

    h0 is now a fixed zero buffer, and old dense input_by_condition tensors
    do not map onto the factorized input parameterization.
    """
    prepared = dict(rnn_state_dict)
    prepared.pop('h0', None)

    if 'input_by_condition' in prepared:
        print(
            'Skipping legacy dense input_by_condition; factorized input '
            'parameters are initialized from scratch.'
        )
        prepared.pop('input_by_condition')

    return prepared


def load_readouts_state_dict(readouts_by_dataset, readouts_state_dict):
    """
    Load readout parameters.

    Older checkpoints stored dense raw_total_readouts. Those cannot be loaded
    into the new shared-V low-rank readout parameterization, so they are
    skipped and the new low-rank readouts keep their initialization.
    """
    has_dense_readouts = any(
        'raw_total_readouts' in key
        for key in readouts_state_dict
    )

    if has_dense_readouts:
        print(
            'Skipping legacy dense raw_total_readouts; low-rank readout '
            'factors are initialized from scratch.'
        )
        filtered_state_dict = {
            key: value
            for key, value in readouts_state_dict.items()
            if 'raw_total_readouts' not in key
        }
        readouts_by_dataset.load_state_dict(filtered_state_dict, strict=False)
    else:
        readouts_by_dataset.load_state_dict(readouts_state_dict)


def load_latent_network(
    checkpoint_path=os.path.join(SAVE_LATENT_PATH, 'latent_network.pt'),
    training_path=TRAINING_DATA_PATH,
    device=DEFAULT_DEVICE,
):
    """
    Load saved network parameters and recreate the model objects.

    The training data are loaded again only to reconstruct the readout
    objects with the correct per-session matrix shapes.
    """
    datasets = load_training_data(training_path)
    device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    n_time = checkpoint['config']['n_time']
    n_sessions = checkpoint['config']['n_sessions']
    recurrent_rank = checkpoint['config'].get(
        'recurrent_rank',
        N_RNN_UNITS
    )
    use_recurrent_v_session = checkpoint['config'].get(
        'use_recurrent_v_session',
        checkpoint['config'].get('use_session_v', False)
    )
    max_recurrent_v_session_fraction = checkpoint['config'].get(
        'max_recurrent_v_session_fraction',
        checkpoint['config'].get(
            'max_v_session_fraction',
            DEFAULT_MAX_RECURRENT_V_SESSION_FRACTION
        )
    )
    n_input_components = checkpoint['config'].get(
        'n_input_components',
        DEFAULT_N_INPUT_COMPONENTS
    )
    use_input_u_session = checkpoint['config'].get(
        'use_input_u_session',
        False
    )
    max_input_u_session_fraction = checkpoint['config'].get(
        'max_input_u_session_fraction',
        DEFAULT_MAX_INPUT_U_SESSION_FRACTION
    )
    readout_rank = checkpoint['config'].get(
        'readout_rank',
        DEFAULT_READOUT_RANK
    )

    rnn = LatentRNN(
        N_RNN_UNITS,
        n_time,
        n_sessions,
        recurrent_rank=recurrent_rank,
        use_recurrent_v_session=use_recurrent_v_session,
        max_recurrent_v_session_fraction=max_recurrent_v_session_fraction,
        n_input_components=n_input_components,
        use_input_u_session=use_input_u_session,
        max_input_u_session_fraction=max_input_u_session_fraction
    ).to(device)
    rnn_state_dict = checkpoint['rnn_state_dict']
    has_session_recurrent_weights = any(
        key.startswith('W_by_session.')
        or key.startswith('U_by_session.')
        or key.startswith('V_by_session.')
        for key in rnn_state_dict
    )
    if has_session_recurrent_weights:
        print(
            'Converting session-specific recurrent checkpoint to one shared '
            f'low-rank recurrent matrix with rank={recurrent_rank}.'
        )
        rnn_state_dict = convert_legacy_rnn_state_dict(rnn_state_dict, rnn)
    else:
        rnn_state_dict = rename_legacy_shared_v(rnn_state_dict)
    rnn_state_dict = prepare_rnn_state_dict_for_load(rnn_state_dict)
    rnn.load_state_dict(rnn_state_dict, strict=False)

    readouts_by_dataset = nn.ModuleList([
        SessionReadouts(dataset['cells'], readout_rank=readout_rank)
        for dataset in datasets
    ]).to(device)
    load_readouts_state_dict(
        readouts_by_dataset,
        checkpoint['readouts_state_dict']
    )

    return {
        'rnn': rnn,
        'readouts_by_dataset': readouts_by_dataset,
        'loss_history': checkpoint['loss_history'],
        'datasets': datasets,
        'checkpoint_path': checkpoint_path,
    }


# ==============================================================
# Step 7. Compare target and prediction
# ==============================================================


import matplotlib.pyplot as plt


def plot_prediction_vs_target(
    results,
    dataset_idx=0,
    fx=0,
    condition='P1',
):
    """
    Compare model prediction and target neural activity for one dataset,
    one session, and one condition.

    prediction shape: (num_top_neurons, num_timesteps)
    target shape:     (num_top_neurons, num_timesteps)
    """
    os.makedirs(SAVE_FIGURE_PATH, exist_ok=True)

    rnn = results['rnn']
    readouts_by_dataset = results['readouts_by_dataset']
    datasets = results['datasets']

    dataset = datasets[dataset_idx]
    cells_fx = dataset['cells'][fx]
    target = dataset['activity'][fx][condition]

    rnn.eval()
    with torch.no_grad():
        g_by_condition = rnn(fx)

        total_readout_fx = readouts_by_dataset[dataset_idx].total_readout(fx)
        actual_readout, _ = build_actual_readout(
            total_readout_fx,
            cells_fx
        )

        prediction = (
            actual_readout
            @ g_by_condition[condition].T
        ).cpu().numpy()

    # vmax = np.nanmax([np.nanmax(target), np.nanmax(prediction)])
    # vmin = np.nanmin([np.nanmin(target), np.nanmin(prediction)])
    vmin = 0
    vmax = 0.2

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(
        target,
        aspect='auto',
        cmap='jet',
        vmin=vmin,
        vmax=vmax
    )
    plt.title(f'Target: dataset {dataset_idx}, fx {fx}, {condition}')
    plt.xlabel('time')
    plt.ylabel('neurons')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 2, 2)
    plt.imshow(
        prediction,
        aspect='auto',
        cmap='jet',
        vmin=vmin,
        vmax=vmax
    )
    plt.title('Prediction')
    plt.xlabel('time')
    plt.ylabel('neurons')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(SAVE_FIGURE_PATH + 'compare_pred_target.png',dpi=300)


# ==============================================================
# Step 8. Plot saved latent activity and recurrent weights
# ==============================================================


def _draw_unit_boundaries(ax):
    """Mark group/population boundaries for the 120 ordered latent units."""
    for i in range(1, len(NEURON_TYPES) * len(GROUPS)):
        ax.axhline(i * N_UNITS_PER_GROUP - 0.5, color='w', lw=0.5)
        ax.axvline(i * N_UNITS_PER_GROUP - 0.5, color='w', lw=0.5)

    for i in range(1, len(NEURON_TYPES)):
        ax.axhline(i * len(GROUPS) * N_UNITS_PER_GROUP - 0.5, color='w', lw=1.5)
        ax.axvline(i * len(GROUPS) * N_UNITS_PER_GROUP - 0.5, color='w', lw=1.5)


def plot_latent_activity_and_weights(
    results,
    fx=0,
    condition='P1',
    savefig=True,
):
    """
    Simulate the loaded latent model and plot:
        1. rectified latent unit activity g(h), units x time
        2. recurrent weight matrix W, units x units
    """
    os.makedirs(SAVE_FIGURE_PATH, exist_ok=True)

    rnn = results['rnn']
    rnn.eval()

    with torch.no_grad():
        g_by_condition = rnn(fx)
        latent_activity = g_by_condition[condition].T.cpu().numpy()
        W = rnn.recurrent_weight(fx).cpu().numpy()

    wlim = np.max(np.abs(W))/4
    if wlim == 0:
        wlim = 1

    plt.figure(figsize=(5, 4))

    ax1 = plt.subplot(1, 1, 1)
    plt.imshow(
        latent_activity,
        aspect='auto',
        cmap='jet',
        vmin=0,
        vmax=10
    )
    plt.title(f'Latent activity g(h): fx {fx}, {condition}')
    plt.xlabel('time')
    plt.ylabel('latent units')
    plt.colorbar(fraction=0.046, pad=0.04)
    for i in range(1, len(NEURON_TYPES) * len(GROUPS)):
        ax1.axhline(i * N_UNITS_PER_GROUP - 0.5, color='w', lw=0.5)
    for i in range(1, len(NEURON_TYPES)):
        ax1.axhline(i * len(GROUPS) * N_UNITS_PER_GROUP - 0.5, color='w', lw=1.5)

    if savefig:
        plt.savefig(
            SAVE_FIGURE_PATH + f'latent_activity_fx{fx}_{condition}.png',
            dpi=300
        )

    plt.figure(figsize=(5, 4))
    
    ax2 = plt.subplot(1, 1, 1)
    plt.imshow(
        W,
        aspect='auto',
        cmap='bwr',
        vmin=-wlim,
        vmax=wlim
    )
    if rnn.use_recurrent_v_session:
        plt.title(f'Session recurrent weight W: fx {fx}')
    else:
        plt.title('Shared recurrent weight W')
    plt.xlabel('from unit')
    plt.ylabel('to unit')
    plt.colorbar(fraction=0.046, pad=0.04)
    _draw_unit_boundaries(ax2)

    plt.tight_layout()

    if savefig:
        plt.savefig(
            SAVE_FIGURE_PATH + f'W_fx{fx}_{condition}.png',
            dpi=300
        )


# Run this file directly to train, or use --load_only to simulate a saved model.
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument(
        '--device',
        type=str,
        default=DEFAULT_DEVICE,
        help='Device for model tensors: cpu, cuda, cuda:0, mps, or gpu.'
    )
    parser.add_argument('--recurrent_rank', type=int, default=DEFAULT_RECURRENT_RANK)
    parser.add_argument(
        '--drop_recurrent_v_session',
        '--drop_session_v',
        dest='drop_recurrent_v_session',
        action='store_true'
    )
    parser.add_argument(
        '--recurrent_v_session_regularization',
        '--v_session_regularization',
        dest='recurrent_v_session_regularization',
        type=float,
        default=DEFAULT_RECURRENT_V_SESSION_REGULARIZATION
    )
    parser.add_argument(
        '--max_recurrent_v_session_fraction',
        '--max_v_session_fraction',
        dest='max_recurrent_v_session_fraction',
        type=float,
        default=DEFAULT_MAX_RECURRENT_V_SESSION_FRACTION
    )
    parser.add_argument(
        '--n_input_components',
        type=int,
        default=DEFAULT_N_INPUT_COMPONENTS
    )
    parser.add_argument('--drop_input_u_session', action='store_true')
    parser.add_argument(
        '--input_u_session_regularization',
        type=float,
        default=DEFAULT_INPUT_U_SESSION_REGULARIZATION
    )
    parser.add_argument(
        '--max_input_u_session_fraction',
        type=float,
        default=DEFAULT_MAX_INPUT_U_SESSION_FRACTION
    )
    parser.add_argument('--readout_rank', type=int, default=DEFAULT_READOUT_RANK)
    parser.add_argument('--condition', type=str, default='P1', choices=CONDITIONS)
    parser.add_argument('--load_only', action='store_true')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=os.path.join(SAVE_LATENT_PATH, 'latent_network.pt')
    )
    parser.add_argument('--dataset_idx', type=int, default=0)
    parser.add_argument('--fx', type=int, default=0)
    args = parser.parse_args()

    if args.load_only:
        loaded_results = load_latent_network(
            checkpoint_path=args.checkpoint_path,
            device=args.device
        )
    else:
        train_latent_network(
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            recurrent_rank=args.recurrent_rank,
            use_recurrent_v_session=not args.drop_recurrent_v_session,
            recurrent_v_session_regularization=(
                args.recurrent_v_session_regularization
            ),
            max_recurrent_v_session_fraction=(
                args.max_recurrent_v_session_fraction
            ),
            n_input_components=args.n_input_components,
            use_input_u_session=not args.drop_input_u_session,
            input_u_session_regularization=args.input_u_session_regularization,
            max_input_u_session_fraction=args.max_input_u_session_fraction,
            readout_rank=args.readout_rank,
            save_path=SAVE_LATENT_PATH,
            device=args.device
        )
        loaded_results = load_latent_network(
            checkpoint_path=args.checkpoint_path,
            device=args.device
        )

    plot_prediction_vs_target(
        loaded_results,
        dataset_idx=args.dataset_idx,
        fx=args.fx,
        condition=args.condition,
    )

    plot_latent_activity_and_weights(
        loaded_results,
        fx=args.fx,
        condition=args.condition,
    )
