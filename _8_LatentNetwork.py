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
DT = 0.1


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

    Here g is ReLU. I is implemented as a trainable time-dependent input
    for each trial condition. If you later have measured inputs, replace
    self.input_by_condition with those inputs.
    """

    def __init__(
        self,
        n_units,
        n_time,
        n_sessions,
        recurrent_rank=DEFAULT_RECURRENT_RANK,
        n_conditions=len(CONDITIONS),
        dt=DT,
    ):
        super().__init__()

        self.n_units = n_units
        self.n_time = n_time
        self.n_sessions = n_sessions
        self.recurrent_rank = recurrent_rank
        self.dt = dt

        if recurrent_rank < 1 or recurrent_rank > n_units:
            raise ValueError(
                f'recurrent_rank must be between 1 and {n_units}, '
                f'got {recurrent_rank}.'
            )

        # Each session has a low-rank recurrent matrix:
        #     W = U @ V.T
        # with rank at most recurrent_rank. All sessions start from the exact
        # same U and V factors, then each session's factors train independently.
        factor_scale = np.sqrt(0.05 / np.sqrt(recurrent_rank))
        U_init = factor_scale * torch.randn(n_units, recurrent_rank)
        V_init = factor_scale * torch.randn(n_units, recurrent_rank)
        self.U_by_session = nn.ParameterList([
            nn.Parameter(U_init.clone())
            for _ in range(n_sessions)
        ])
        self.V_by_session = nn.ParameterList([
            nn.Parameter(V_init.clone())
            for _ in range(n_sessions)
        ])
        self.h0 = nn.Parameter(torch.zeros(n_conditions, n_units))
        self.input_by_condition = nn.Parameter(
            0.01 * torch.randn(n_conditions, n_time, n_units)
        )

        # The external input is trainable only in the first and last thirds.
        # Multiplying by zero in the middle third blocks gradients to
        # input_by_condition[:, middle_start:middle_end].
        input_mask = torch.zeros(n_time)
        input_mask[: n_time // 3] = 1.0
        input_mask[2 * n_time // 3:] = 1.0
        self.register_buffer('input_mask', input_mask)

    def recurrent_weight(self, fx):
        """Return the rank-constrained recurrent matrix for one session."""
        return self.U_by_session[fx] @ self.V_by_session[fx].T
        
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
            h_over_time = []

            for t in range(self.n_time):
                # if t == middle_start or t == middle_end:
                #     h = h.detach()

                h_over_time.append(h)
                g_h = F.relu(h)
                I_t = self.input_by_condition[icond, t] * self.input_mask[t]

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
    One total readout matrix per session.

    total_readout(fx) has shape:
        (number of original neurons in session fx, number of RNN units)

    The trainable parameters are raw_total_readouts. The total readout used
    in the model is softplus(raw_total_readouts), so every usable readout
    entry is positive after initialization and after every optimizer update.

    Most rows/columns are never used directly in a training loss. The
    helper build_actual_readout selects top-cell rows and masks columns.
    """

    def __init__(self, cells):
        super().__init__()

        self.raw_total_readouts = nn.ParameterList()

        for fx in sorted(cells):
            n_og_cells = get_num_original_cells(cells[fx])
            init_total_readout = torch.full(
                (n_og_cells, N_RNN_UNITS),
                0.01
            )
            raw_readout_fx = nn.Parameter(
                torch.log(torch.expm1(init_total_readout))
            )
            self.raw_total_readouts.append(raw_readout_fx)

    def total_readout(self, fx):
        return F.softplus(self.raw_total_readouts[fx])


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
    training_path=TRAINING_DATA_PATH,
    save_path=SAVE_LATENT_PATH,
    device='cpu',
):
    """
    Train session-specific low-rank W matrices and the session readouts.

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

    rnn = LatentRNN(
        N_RNN_UNITS,
        n_time,
        n_sessions,
        recurrent_rank=recurrent_rank
    ).to(device)

    # One readout model per independently resampled training dataset.
    # Each readout model contains 33 session-specific total readout matrices.
    readouts_by_dataset = nn.ModuleList([
        SessionReadouts(dataset['cells'])
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
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.detach().cpu()))

        if step % 50 == 0:
            print(f'step {step:04d} | loss {loss_history[-1]:.6f}')

    results = {
        'rnn': rnn,
        'readouts_by_dataset': readouts_by_dataset,
        'loss_history': loss_history,
        'datasets': datasets,
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
        - RNN parameters: low-rank W factors, h0, and
          condition-dependent input I
        - raw_total_readouts for every dataset/session
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
            'dt': results['rnn'].dt,
            'neuron_types': NEURON_TYPES,
            'groups': GROUPS,
            'conditions': CONDITIONS,
        },
    }

    torch.save(checkpoint, checkpoint_path)
    results['checkpoint_path'] = checkpoint_path
    print(f'Saved latent network to {checkpoint_path}')


def convert_legacy_rnn_state_dict(rnn_state_dict, rnn):
    """
    Convert older checkpoints that stored full W_by_session matrices.

    The full recurrent matrix is projected into the requested low-rank form
    using a truncated SVD:
        W ~= U @ V.T
    """
    converted = {
        key: value
        for key, value in rnn_state_dict.items()
        if not key.startswith('W_by_session.')
    }

    for fx in range(rnn.n_sessions):
        legacy_key = f'W_by_session.{fx}'
        if legacy_key not in rnn_state_dict:
            continue

        W = rnn_state_dict[legacy_key]
        U_svd, S_svd, Vh_svd = torch.linalg.svd(W, full_matrices=False)
        rank = rnn.recurrent_rank
        sqrt_s = torch.sqrt(S_svd[:rank])
        converted[f'U_by_session.{fx}'] = U_svd[:, :rank] * sqrt_s
        converted[f'V_by_session.{fx}'] = Vh_svd[:rank, :].T * sqrt_s

    return converted


def load_latent_network(
    checkpoint_path=os.path.join(SAVE_LATENT_PATH, 'latent_network.pt'),
    training_path=TRAINING_DATA_PATH,
    device='cpu',
):
    """
    Load saved network parameters and recreate the model objects.

    The training data are loaded again only to reconstruct the readout
    objects with the correct per-session matrix shapes.
    """
    datasets = load_training_data(training_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    n_time = checkpoint['config']['n_time']
    n_sessions = checkpoint['config']['n_sessions']
    recurrent_rank = checkpoint['config'].get(
        'recurrent_rank',
        N_RNN_UNITS
    )

    rnn = LatentRNN(
        N_RNN_UNITS,
        n_time,
        n_sessions,
        recurrent_rank=recurrent_rank
    ).to(device)
    rnn_state_dict = checkpoint['rnn_state_dict']
    if any(key.startswith('W_by_session.') for key in rnn_state_dict):
        print(
            'Converting legacy W_by_session checkpoint to '
            f'low-rank factors with rank={recurrent_rank}.'
        )
        rnn_state_dict = convert_legacy_rnn_state_dict(rnn_state_dict, rnn)
    rnn.load_state_dict(rnn_state_dict)

    readouts_by_dataset = nn.ModuleList([
        SessionReadouts(dataset['cells'])
        for dataset in datasets
    ]).to(device)
    readouts_by_dataset.load_state_dict(checkpoint['readouts_state_dict'])

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
    plt.title(f'Recurrent weight W: fx {fx}')
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
    parser.add_argument('--recurrent_rank', type=int, default=DEFAULT_RECURRENT_RANK)
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
            checkpoint_path=args.checkpoint_path
        )
    else:
        train_latent_network(
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            recurrent_rank=args.recurrent_rank,
            save_path=SAVE_LATENT_PATH
        )
        loaded_results = load_latent_network(
            checkpoint_path=args.checkpoint_path
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
