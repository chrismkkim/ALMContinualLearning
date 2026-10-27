"""Behavior checks for direct fitting of variable-sized session networks."""

import json
from pathlib import Path
import runpy
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _8_LatentNetwork as ln


@pytest.fixture(params=['fullconn', 'lowrank'])
def recurrent_config(request):
    return {
        'recurrent_type': request.param,
        'recurrent_rank': 2 if request.param == 'lowrank' else None,
    }


def recurrent_parameters(model):
    if model.recurrent_type == 'fullconn':
        return (model.W,)
    return (model.U, model.V)


def fill_recurrent_weight(model, value):
    """Set either parameterization to the same constant effective matrix."""
    with torch.no_grad():
        if model.recurrent_type == 'fullconn':
            model.W.fill_(value)
        else:
            model.U.fill_(1)
            model.V.fill_(value / (model.recurrent_scale * model.recurrent_rank))


@pytest.fixture
def training_data(tmp_path):
    """Provide activity files only, with noncontiguous session IDs."""
    root = tmp_path / 'training'
    root.mkdir()
    rng = np.random.default_rng(12)
    for dataset_id, sizes in (('0', (4, 7)), ('2', (5, 8)), ('10', (6, 9))):
        activity = {}
        for fx, n_units, n_time in ((3, sizes[0], 12), (8, sizes[1], 15)):
            activity[fx] = {
                condition: rng.uniform(0, 0.2, (n_units, n_time))
                for condition in ('P1', 'A1', 'P2', 'A2')
            }
        np.save(root / f'activity{dataset_id}.npy', activity)
    return root


def test_dataset_choice_and_activity_shapes(training_data):
    assert not list(training_data.glob('dict*.npy'))
    assert ln.available_training_datasets(training_data) == ['0', '2', '10']
    dataset = ln.load_training_data(training_data, dataset_id=2)
    assert dataset['id'] == '2'
    assert ln.dataset_session_shapes(dataset) == {3: (5, 12), 8: (8, 15)}
    assert set(dataset) == {'id', 'activity'}
    with pytest.raises(FileNotFoundError, match='Available dataset IDs'):
        ln.load_training_data(training_data, dataset_id=1)


def test_variable_sizes_and_independent_session_parameters(recurrent_config):
    model = ln.LatentRNN({3: (4, 12), 8: (7, 15)}, n_input_components=3, **recurrent_config, n_initial_timepoints=None)
    assert model.session_ids == (3, 8)
    assert model.n_units_by_session == {3: 4, 8: 7}
    for fx, n_units, n_time in ((3, 4, 12), (8, 7, 15)):
        assert model.recurrent_weight(fx).shape == (n_units, n_units)
        assert model.session_model(fx).input_vectors.shape == (2, 2, 3, n_units)
        predictions = model(fx)
        for values in predictions.values():
            assert values.shape == (n_time, n_units)
            assert torch.count_nonzero(values[0]) == 0
    for first, second in zip(recurrent_parameters(model.session_model(3)), recurrent_parameters(model.session_model(8))):
        assert first.data_ptr() != second.data_ptr()
    if model.recurrent_type == 'fullconn':
        assert set(dict(model.session_model(3).named_parameters())) == {'W', 'input_vectors'}
    else:
        assert set(dict(model.session_model(3).named_parameters())) == {'U', 'V', 'input_vectors'}
        for fx, n_units in ((3, 4), (8, 7)):
            assert model.session_model(fx).U.shape == (n_units, 2)
            assert model.session_model(fx).V.shape == (n_units, 2)
        other_weights = model.recurrent_weight(8).detach().clone()
        with torch.no_grad():
            model.session_model(3).U.add_(1)
        torch.testing.assert_close(model.recurrent_weight(8), other_weights, rtol=0, atol=0)


def test_recurrent_initialization_matches_requested_gaussian():
    torch.manual_seed(17)
    expected = (0.1 / np.sqrt(20)) * torch.randn(20, 20)
    torch.manual_seed(17)
    model = ln.SessionRNN(20, 12, n_initial_timepoints=None)
    torch.testing.assert_close(model.W, expected, rtol=0, atol=0)


def test_lowrank_initialization_has_matching_entry_scale():
    torch.manual_seed(47)
    model = ln.SessionRNN(256, 12, recurrent_type='lowrank', recurrent_rank=4, n_initial_timepoints=None)
    weights = model.recurrent_weight().detach()
    expected_std = 0.1 / np.sqrt(256)
    assert weights.std().item() == pytest.approx(expected_std, rel=0.15)
    assert abs(weights.mean().item()) < 0.1 * expected_std


@pytest.mark.parametrize('rank', [1, 3])
def test_lowrank_drive_matches_materialized_weight_and_gradients(rank):
    torch.manual_seed(31)
    model = ln.SessionRNN(7, 12, recurrent_type='lowrank', recurrent_rank=rank, n_initial_timepoints=None).double()
    rates = torch.rand(3, 7, dtype=torch.float64, requires_grad=True)
    weights = model.recurrent_weight()
    assert weights.shape == (7, 7)
    assert torch.linalg.matrix_rank(weights).item() == rank
    factorized = model.recurrent_drive(rates, train_recurrent=True)
    materialized = rates @ weights.T
    torch.testing.assert_close(factorized, materialized, rtol=1e-12, atol=1e-12)
    factor_gradients = torch.autograd.grad(factorized.square().sum(), (model.U, model.V, rates))
    dense_gradients = torch.autograd.grad(materialized.square().sum(), (model.U, model.V, rates))
    for actual, expected in zip(factor_gradients, dense_gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    detached = model.recurrent_drive(rates, train_recurrent=False)
    torch.testing.assert_close(detached, materialized, rtol=1e-12, atol=1e-12)
    u_grad, v_grad, state_grad = torch.autograd.grad(
        detached.sum(), (model.U, model.V, rates), allow_unused=True,
    )
    assert u_grad is None and v_grad is None
    assert torch.count_nonzero(state_grad) > 0


@pytest.mark.parametrize('recurrent_type, rank, message', [
    ('other', None, 'recurrent_type'),
    ('lowrank', None, 'positive integer'),
    ('lowrank', 0, 'positive integer'),
    ('lowrank', -1, 'positive integer'),
    ('lowrank', 2.5, 'positive integer'),
    ('lowrank', True, 'positive integer'),
    ('lowrank', 5, 'session size'),
])
def test_recurrent_configuration_is_validated_at_construction(recurrent_type, rank, message):
    with pytest.raises(ValueError, match=message):
        ln.SessionRNN(4, 12, recurrent_type=recurrent_type, recurrent_rank=rank, n_initial_timepoints=None)


def test_lowrank_rank_must_fit_smallest_session():
    with pytest.raises(ValueError, match='session size of 3'):
        ln.LatentRNN({3: (3, 12), 8: (7, 15)}, recurrent_type='lowrank', recurrent_rank=4, n_initial_timepoints=None)


def test_fixed_zero_initial_states_and_fresh_noise_in_eval(recurrent_config):
    model = ln.SessionRNN(4, 12, **recurrent_config, n_initial_timepoints=None)
    model.eval()
    fill_recurrent_weight(model, 0.01)
    with torch.no_grad():
        model.input_vectors.fill_(0.1)
    first = model()
    second = model()
    assert not torch.equal(first['P1'][1:], second['P1'][1:])
    for condition in model.conditions:
        assert torch.count_nonzero(first[condition][0]) == 0
        assert torch.count_nonzero(second[condition][0]) == 0
    assert torch.count_nonzero(model.h0) == 0
    assert not model.h0.requires_grad
    assert model.h0.grad is None


def test_input_noise_is_untrained_and_independent_across_dimensions():
    model = ln.SessionRNN(4, 12, n_input_components=3, n_initial_timepoints=None)
    torch.manual_seed(19)
    expected = 1 + 0.1 * torch.randn(2, 2, 3, 12)
    torch.manual_seed(19)
    samples = model.sample_input_timecourses()
    torch.testing.assert_close(samples, expected, rtol=0, atol=0)
    assert not samples.requires_grad
    assert not torch.equal(samples, model.sample_input_timecourses())
    assert not torch.equal(samples[0], samples[1])
    assert not torch.equal(samples[:, 0], samples[:, 1])
    assert not torch.equal(samples[:, :, 0], samples[:, :, 1])
    assert all('timecourses' not in key for key in model.state_dict())


@pytest.mark.parametrize('input_type, active_slice', [(0, slice(0, 4)), (1, slice(8, 12))])
def test_sensory_and_response_have_separate_temporal_masks(input_type, active_slice):
    model = ln.SessionRNN(4, 12, n_input_components=3, input_noise_std=0, n_initial_timepoints=None)
    with torch.no_grad():
        model.input_vectors.zero_()
        model.input_vectors[input_type, 0].fill_(1)
    inputs = model.condition_input(0)
    expected = torch.zeros(12, 4)
    expected[active_slice] = 3
    torch.testing.assert_close(inputs, expected)
    assert torch.count_nonzero(model.condition_input(1)) == 0


@pytest.mark.parametrize('n_time', [10, 12])
@pytest.mark.parametrize('transition', ['before_middle', 'first_middle', 'last_middle', 'after_middle'])
def test_recurrent_weights_receive_gradients_only_in_middle_third(n_time, transition, recurrent_config):
    start, stop = n_time // 3, 2 * n_time // 3
    step = {
        'before_middle': start - 1,
        'first_middle': start,
        'last_middle': stop - 1,
        'after_middle': stop,
    }[transition]
    # A pulse immediately before this transition makes it the first recurrence
    # with nonzero activity, isolating its W gradient from earlier transitions.
    masks = torch.zeros(2, n_time)
    masks[0, step - 1] = 1
    model = ln.SessionRNN(4, n_time, input_noise_std=0, input_masks=masks, **recurrent_config, n_initial_timepoints=None)
    fill_recurrent_weight(model, 0.01)
    with torch.no_grad():
        model.input_vectors.zero_()
        model.input_vectors[0].fill_(0.1)
    model()['P1'][step + 1].sum().backward()
    for parameter in recurrent_parameters(model):
        if transition in ('first_middle', 'last_middle'):
            assert parameter.grad is not None
            assert torch.all(parameter.grad > 0)
        else:
            assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
    assert torch.all(model.input_vectors.grad[0, 0] > 0)
    assert model.h0.grad is None


def test_later_losses_train_middle_third_through_state_history(recurrent_config):
    model = ln.SessionRNN(4, 12, input_noise_std=0, **recurrent_config, n_initial_timepoints=None)
    fill_recurrent_weight(model, 0.01)
    with torch.no_grad():
        model.input_vectors.zero_()
        model.input_vectors[0].fill_(0.1)
    # A late loss still reaches W through earlier middle-third transitions.
    model()['P1'][-1].sum().backward()
    for parameter in recurrent_parameters(model):
        assert parameter.grad is not None
        assert torch.all(parameter.grad > 0)
    assert torch.all(model.input_vectors.grad[0, 0] > 0)
    assert model.h0.grad is None


def test_recurrent_dynamics_remain_active_after_middle_third(recurrent_config):
    config = dict(recurrent_config)
    if config['recurrent_type'] == 'lowrank':
        config['recurrent_rank'] = 1
    model = ln.SessionRNN(1, 12, input_noise_std=0, dt=1, **config, n_initial_timepoints=None)
    fill_recurrent_weight(model, 0.5)
    with torch.no_grad():
        model.input_vectors.zero_()
        model.input_vectors[1].fill_(1 / model.n_input_components)
    # Response-only drive: r9=1, r10=0.5*r9+1, r11=0.5*r10+1.
    expected = torch.zeros(12, 1)
    expected[9:, 0] = torch.tensor([1.0, 1.5, 1.75])
    torch.testing.assert_close(model()['P1'], expected)


def test_custom_input_type_count_and_component_count():
    masks = torch.zeros(3, 12)
    for i in range(3):
        masks[i, 4 * i:4 * (i + 1)] = 1
    model = ln.SessionRNN(4, 12, n_input_types=3, n_input_components=2, input_masks=masks, n_initial_timepoints=None)
    assert model.input_vectors.shape == (3, 2, 2, 4)
    assert model.sample_input_timecourses().shape == (3, 2, 2, 12)
    assert model()['P1'].shape == (12, 4)
    with pytest.raises(ValueError, match='Supply input_masks'):
        ln.SessionRNN(4, 12, n_input_types=3, n_initial_timepoints=None)


def test_direct_training_checkpoint_and_reload(training_data, tmp_path, recurrent_config):
    torch.manual_seed(22)
    path = tmp_path / 'selected.pt'
    results = ln.train_latent_network(
        n_steps=3, dataset_id='2', n_input_components=2,
        training_path=training_data, checkpoint_path=path,
        **recurrent_config,
    )
    assert results['dataset']['id'] == '2'
    assert results['rnn'].conditions == ('P1', 'A1')
    assert len(results['loss_history']) == 3
    assert np.isfinite(results['loss_history']).all()
    assert set(results['loss_history_by_session']) == {3, 8}
    assert all(len(history) == 3 for history in results['loss_history_by_session'].values())
    np.testing.assert_allclose(
        results['loss_history'],
        np.mean(list(results['loss_history_by_session'].values()), axis=0),
    )
    assert results['checkpoint_path'] == str(path)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint['version'] == 4
    assert checkpoint['config']['recurrent_type'] == recurrent_config['recurrent_type']
    assert checkpoint['config']['recurrent_rank'] == recurrent_config['recurrent_rank']
    assert 'neuron_ids_by_session' not in checkpoint
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for session in results['rnn'].sessions.values()
        for parameter in recurrent_parameters(session)
    )
    loaded = ln.load_latent_network(path, training_path=training_data)
    assert loaded['dataset']['id'] == '2'
    assert loaded['rnn'].n_input_components == 2
    assert loaded['rnn'].recurrent_type == recurrent_config['recurrent_type']
    assert loaded['rnn'].recurrent_rank == recurrent_config['recurrent_rank']
    assert loaded['loss_history'] == results['loss_history']
    assert loaded['loss_history_by_session'] == results['loss_history_by_session']
    for key, value in results['rnn'].state_dict().items():
        torch.testing.assert_close(loaded['rnn'].state_dict()[key], value, rtol=0, atol=0)
    # Resetting the RNG explicitly lets us compare the restored model using
    # the same fresh noise draw; normal consecutive calls use different noise.
    prefix = results['rnn'].initial_activity_from_targets(3, results['dataset']['activity'][3])
    torch.manual_seed(23)
    expected = results['rnn'](3, initial_activity=prefix)
    torch.manual_seed(23)
    actual = loaded['rnn'](3, initial_activity=prefix)
    for condition in expected:
        torch.testing.assert_close(actual[condition], expected[condition], rtol=0, atol=0)


def test_custom_masks_survive_checkpoint(training_data, tmp_path, recurrent_config):
    masks = {}
    for fx, n_time in ((3, 12), (8, 15)):
        masks[fx] = torch.zeros(3, n_time)
        for i in range(3):
            masks[fx][i, i * (n_time // 3):(i + 1) * (n_time // 3)] = 1
    results = ln.train_latent_network(
        n_steps=1, n_input_types=3, input_masks=masks,
        training_path=training_data, save_path=tmp_path,
        **recurrent_config,
    )
    loaded = ln.load_latent_network(results['checkpoint_path'], training_path=training_data)
    for fx in masks:
        torch.testing.assert_close(loaded['rnn'].session_model(fx).input_masks, masks[fx])
    assert Path(results['checkpoint_path']).parent == tmp_path / recurrent_config['recurrent_type']


def test_checkpoint_types_and_ranks_use_distinct_paths(training_data):
    paths = set()
    for recurrent_type, rank in (('fullconn', None), ('lowrank', 1), ('lowrank', 2)):
        results = ln.train_latent_network(
            n_steps=1, training_path=training_data,
            recurrent_type=recurrent_type, recurrent_rank=rank,
        )
        rank_suffix = f'_Rank{rank}' if rank is not None else ''
        expected = training_data / recurrent_type / (
            f'latent_network_Dataset0_Types2_In1{rank_suffix}_{recurrent_type}.pt'
        )
        assert results['checkpoint_path'] == str(expected)
        assert ln.latent_network_checkpoint_path(
            training_path=training_data, recurrent_type=recurrent_type, recurrent_rank=rank,
        ) == str(expected)
        paths.add(expected)
    assert len(paths) == 3
    assert all(path.is_file() for path in paths)
    assert ln.available_training_datasets(training_data) == ['0', '2', '10']
    # Default reload finds fullconn under the selected training directory.
    assert ln.load_latent_network(training_path=training_data)['rnn'].recurrent_type == 'fullconn'


@pytest.mark.parametrize('axis', [0, 1])
def test_changed_activity_shape_is_rejected(training_data, tmp_path, axis):
    results = ln.train_latent_network(n_steps=1, training_path=training_data, save_path=tmp_path)
    path = training_data / 'activity0.npy'
    activity = np.load(path, allow_pickle=True).item()
    for condition, values in activity[3].items():
        activity[3][condition] = np.concatenate((values, np.take(values, [0], axis=axis)), axis=axis)
    np.save(path, activity)
    with pytest.raises(ValueError, match='Training data shapes differ'):
        ln.load_latent_network(results['checkpoint_path'], training_path=training_data)


def test_legacy_checkpoint_loads_without_metadata_file(training_data, tmp_path):
    results = ln.train_latent_network(n_steps=1, training_path=training_data, save_path=tmp_path)
    checkpoint = torch.load(results['checkpoint_path'], weights_only=True)
    checkpoint['version'] = 2
    checkpoint['config'].pop('recurrent_type')
    checkpoint['config'].pop('recurrent_rank')
    checkpoint['config'].pop('n_initial_timepoints')
    checkpoint.pop('loss_history_by_session')
    for fx, (n_units, _) in checkpoint['config']['session_shapes'].items():
        checkpoint['rnn_state_dict'].pop(f'sessions.{fx}.initial_state_layer.weight')
        checkpoint['rnn_state_dict'].pop(f'sessions.{fx}.initial_state_layer.bias')
        checkpoint['rnn_state_dict'][f'sessions.{fx}.h0'] = torch.zeros(2, n_units)
    checkpoint['neuron_ids_by_session'] = {3: [100, 101, 102, 103], 8: list(range(100, 107))}
    torch.save(checkpoint, results['checkpoint_path'])
    loaded = ln.load_latent_network(results['checkpoint_path'], training_path=training_data)
    assert loaded['loss_history_by_session'] == {}
    assert loaded['rnn'].recurrent_type == 'fullconn'
    assert loaded['rnn'].recurrent_rank is None
    assert loaded['rnn'].session_shapes == results['rnn'].session_shapes
    for key, value in checkpoint['rnn_state_dict'].items():
        torch.testing.assert_close(loaded['rnn'].state_dict()[key], value, rtol=0, atol=0)


def test_plotting_uses_direct_unit_activity(training_data, recurrent_config):
    dataset = ln.load_training_data(training_data)
    model = ln.LatentRNN(ln.dataset_session_shapes(dataset), **recurrent_config, n_initial_timepoints=None)
    results = {'rnn': model, 'dataset': dataset, 'loss_history': []}
    target = dataset['activity'][3]['P1']
    row_order = np.argsort(np.argmax(target, axis=1))
    torch.manual_seed(24)
    with torch.no_grad():
        expected_prediction = model(3)['P1'].T.cpu().numpy()
    torch.manual_seed(24)
    fig = ln.plot_prediction_vs_target(results, savefig=False)
    assert fig.axes[0].images[0].get_array().shape == (4, 12)
    assert fig.axes[1].images[0].get_array().shape == (4, 12)
    np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), target[row_order])
    np.testing.assert_array_equal(fig.axes[1].images[0].get_array(), expected_prediction[row_order])
    target_image, prediction_image = (fig.axes[i].images[0] for i in (0, 1))
    assert target_image.get_clim() == prediction_image.get_clim()
    assert model.training
    activity_fig, weight_fig = ln.plot_latent_activity_and_weights(results, fx=8, savefig=False)
    assert activity_fig.axes[0].images[0].get_array().shape == (7, 15)
    assert weight_fig.axes[0].images[0].get_array().shape == (7, 7)
    plt.close('all')


def test_command_line_dataset_selection(training_data, tmp_path, monkeypatch, capsys, recurrent_config):
    module_path = Path(ln.__file__)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, 'show', lambda: None)
    # Run the actual command-line entry point without forking an initialized
    # OpenMP runtime, which is unsupported in some macOS environments.
    def run_cli(*args):
        monkeypatch.setattr(sys, 'argv', [str(module_path), *args])
        return runpy.run_path(str(module_path), run_name='__main__')
    run_cli('--training_path', str(training_data), '--list_datasets')
    assert '0, 2, 10' in capsys.readouterr().out
    recurrent_args = ['--recurrent_type', recurrent_config['recurrent_type']]
    if recurrent_config['recurrent_rank'] is not None:
        recurrent_args.extend(['--recurrent_rank', str(recurrent_config['recurrent_rank'])])
    trained = run_cli(
        '--training_path', str(training_data), '--dataset_id', '10',
        '--n_steps', '1', '--n_initial_timepoints', '3', *recurrent_args,
        '--fx', '8', '--condition', 'A1',
    )
    checkpoint = Path(trained['trained_results']['checkpoint_path'])
    assert trained['trained_results']['rnn'].n_initial_timepoints == 3
    assert checkpoint.parent == training_data / recurrent_config['recurrent_type']
    assert checkpoint.is_file()
    assert (tmp_path / 'figure/latent_network/compare_dataset10_fx8_A1.png').is_file()
    fig = trained['prediction_figure']
    target = trained['trained_results']['dataset']['activity'][8]['A1']
    row_order = np.argsort(np.argmax(target, axis=1))
    np.testing.assert_array_equal(
        fig.axes[0].images[0].get_array(),
        target[row_order],
    )
    assert fig.axes[1].images[0].get_array().shape == (9, 15)
    plt.close(fig)
    loaded = run_cli(
        '--training_path', str(training_data), '--load_only',
        '--dataset_id', '10', *recurrent_args,
    )
    assert loaded['loaded_results']['dataset']['id'] == '10'
    assert loaded['loaded_results']['rnn'].recurrent_type == recurrent_config['recurrent_type']
    assert 'Loaded dataset 10; 2 sessions.' in capsys.readouterr().out


def test_notebook_analysis_cells_execute(training_data, tmp_path, monkeypatch, recurrent_config):
    from sklearn.decomposition import PCA
    # Include the notebook's selected session in the synthetic activity file.
    activity_path = training_data / 'activity0.npy'
    activity = np.load(activity_path, allow_pickle=True).item()
    activity[10] = {condition: values.copy() for condition, values in activity[3].items()}
    np.save(activity_path, activity)
    results = ln.train_latent_network(
        n_steps=1, training_path=training_data, save_path=tmp_path,
        **recurrent_config,
    )
    monkeypatch.setattr(ln, 'SAVE_FIGURE_PATH', str(tmp_path / 'figures'))
    monkeypatch.setattr(plt, 'show', lambda: None)
    namespace = {
        'ln': ln, 'np': np, 'torch': torch, 'plt': plt, 'Path': Path, 'PCA': PCA,
        'checkpoint_path': results['checkpoint_path'], 'device': 'cpu',
    }
    loaded = ln.load_latent_network(results['checkpoint_path'], training_path=training_data)
    monkeypatch.setattr(ln, 'load_latent_network', lambda **kwargs: loaded)
    book = json.loads(Path(ln.__file__).with_name('_8_LatentNet_Notebook.ipynb').read_text())
    for index, cell in enumerate(book['cells']):
        if cell['cell_type'] == 'code' and index != 1:
            exec(compile(''.join(cell['source']), f'cell{index}', 'exec'), namespace)
            plt.close('all')
    assert set(namespace['components']) == {3, 8, 10}
    assert namespace['components'][3].shape == (4, 4)
    assert namespace['components'][8].shape == (7, 7)
    assert namespace['components'][10].shape == (4, 4)


def test_notebook_sources_are_valid_and_use_variable_session_sizes():
    path = Path(ln.__file__).with_name('_8_LatentNet_Notebook.ipynb')
    book = json.loads(path.read_text())
    sources = [''.join(cell['source']) for cell in book['cells']]
    assert all('readout' not in source.lower() for source in sources)
    assert 'components = {}' in '\n'.join(sources)
    assert 'rnn.n_units_by_session' in '\n'.join(sources)
    for index, cell in enumerate(book['cells']):
        if cell['cell_type'] == 'code':
            compile(sources[index].replace('%matplotlib inline\n', ''), f'cell{index}', 'exec')
