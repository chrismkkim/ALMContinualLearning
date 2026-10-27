"""Learned initial states, observed windows, checkpoint loading, and prediction."""

from pathlib import Path
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
    return dict(recurrent_type=request.param, recurrent_rank=2 if request.param == 'lowrank' else None)


@pytest.fixture
def activity_data(tmp_path):
    rng = np.random.default_rng(21)
    activity = {
        fx: {condition: rng.uniform(0.05, 0.2, (n, t)).astype(np.float32)
             for condition in ('P1', 'A1', 'P2', 'A2')}
        for fx, n, t in ((3, 4, 9), (11, 7, 12))
    }
    np.save(tmp_path / 'activity9.npy', activity)
    return tmp_path, ln.load_training_data(tmp_path, '9')


def test_default_layer_averages_each_neuron_and_can_mix_neurons():
    model = ln.SessionRNN(3, 9, conditions=('A1', 'P1')).double()
    assert model.n_initial_timepoints == 2
    assert model.initial_state_layer.weight.shape == (3, 6)
    assert not hasattr(model, 'h0')
    prefix = torch.arange(1., 13., dtype=torch.float64).reshape(2, 3, 2)
    torch.testing.assert_close(model.initial_state(prefix), prefix.mean(dim=2))
    with torch.no_grad():
        model.initial_state_layer.weight[1, 0] += 0.25
        model.initial_state_layer.bias[2] = 0.5
    expected = prefix.mean(dim=2)
    expected[:, 1] += 0.25 * prefix[:, 0, 0]
    expected[:, 2] += 0.5
    torch.testing.assert_close(model.initial_state(prefix), expected)
    # The internal state is converted to rates with the existing ReLU.
    with torch.no_grad():
        model.initial_state_layer.bias[0] = -100
    prediction = model(initial_activity=prefix)
    assert prediction['A1'].shape == (9, 3)
    assert prediction['A1'][0, 0] == 0


def test_windows_follow_condition_names_and_ignore_later_targets():
    model = ln.LatentRNN({3: (3, 9), 11: (5, 12)}, conditions=('A2', 'P2'))
    targets = {
        'P2': torch.arange(27.).reshape(3, 9),
        'A2': torch.arange(27.).reshape(3, 9) + 100,
    }
    window = model.initial_activity_from_targets(3, targets)
    torch.testing.assert_close(window[0], targets['A2'][:, :2])
    torch.testing.assert_close(window[1], targets['P2'][:, :2])
    for values in targets.values():
        values[:, 2:] = float('nan')
    torch.testing.assert_close(model.initial_activity_from_targets(3, targets), window)
    assert model.session_model(11).initial_state_layer.weight.shape == (5, 10)
    assert (model.session_model(11).initial_state_layer.weight.data_ptr()
            != model.session_model(3).initial_state_layer.weight.data_ptr())


@pytest.mark.parametrize('loss_bin', [1, 8])
def test_initializer_receives_early_and_late_gradients(recurrent_config, loss_bin):
    model = ln.SessionRNN(3, 9, input_noise_std=0, **recurrent_config).double()
    with torch.no_grad():
        model.input_vectors.fill_(0.1)
        if model.recurrent_type == 'fullconn':
            model.W.fill_(0.02)
        else:
            model.U.fill_(0.3)
            model.V.fill_(0.4)
    prefix = torch.full((2, 3, 2), 0.2, dtype=torch.float64, requires_grad=True)
    model(initial_activity=prefix)['P1'][loss_bin].sum().backward()
    assert torch.all(model.initial_state_layer.weight.grad > 0)
    assert torch.all(model.initial_state_layer.bias.grad > 0)
    assert prefix.grad is None  # Observations are data, not optimized inputs.
    recurrent = (model.W,) if model.recurrent_type == 'fullconn' else (model.U, model.V)
    if loss_bin == 1:
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in recurrent)
    else:
        assert all(p.grad is not None and torch.all(p.grad > 0) for p in recurrent)


@pytest.mark.parametrize('n', [0, -1, 1.5, True, 10])
def test_invalid_initial_window_length(n):
    with pytest.raises(ValueError, match='n_initial_timepoints'):
        ln.SessionRNN(3, 9, n_initial_timepoints=n)


def test_initial_observations_are_required_and_validated():
    model = ln.SessionRNN(3, 9)
    with pytest.raises(ValueError, match='initial_activity is required'):
        model()
    for bad in (torch.ones(2, 3, 1), torch.ones(3, 2), torch.full((2, 3, 2), float('nan'))):
        with pytest.raises(ValueError, match='initial_activity'):
            model(initial_activity=bad)
    with pytest.raises(ValueError, match='at least 2'):
        model.initial_activity_from_targets({'P1': np.ones((3, 1)), 'A1': np.ones((3, 1))})


def test_initializer_training_roundtrip_and_plotting(activity_data, recurrent_config):
    root, dataset = activity_data
    torch.manual_seed(19)
    results = ln.train_latent_network(
        n_steps=3, dataset_id='9', training_path=root,
        conditions=('A1', 'P1'), n_initial_timepoints=3, **recurrent_config,
    )
    model = results['rnn']
    for fx in model.session_ids:
        layer = model.session_model(fx).initial_state_layer
        initial_weights = torch.eye(model.n_units_by_session[fx]).repeat_interleave(3, dim=1) / 3
        assert not torch.equal(layer.weight, initial_weights)
        assert torch.isfinite(layer.weight.grad).all()
        assert torch.count_nonzero(layer.weight.grad) > 0
    checkpoint = torch.load(results['checkpoint_path'], weights_only=True)
    assert checkpoint['version'] == 4
    assert checkpoint['config']['n_initial_timepoints'] == 3
    assert not any(key.endswith('.h0') for key in checkpoint['rnn_state_dict'])
    restored = ln.load_latent_network(results['checkpoint_path'], training_path=root)
    loaded = restored['rnn'].eval()
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], tensor, rtol=0, atol=0)

    fx, condition = 3, 'A1'
    prefix = loaded.initial_activity_from_targets(fx, dataset['activity'][fx])
    noise = loaded.session_model(fx).sample_input_timecourses()
    with torch.no_grad():
        original = loaded(fx, initial_activity=prefix, input_timecourses=noise)[condition]
        expected = loaded(fx, initial_activity=prefix, input_timecourses=noise, truncation_rank=1)[condition]
        full_rank = loaded(fx, initial_activity=prefix, input_timecourses=noise, truncation_rank=4)[condition]
    torch.testing.assert_close(original[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(original, full_rank)
    try:
        fig = ln.plot_prediction_vs_target(restored, fx=fx, condition=condition,
                                          truncation_rank=1, input_timecourses=noise, savefig=False)
        order = np.argsort(dataset['activity'][fx][condition].argmax(axis=1))
        np.testing.assert_allclose(fig.axes[1].images[0].get_array(), expected.T.numpy()[order])
        fig_activity, _ = ln.plot_latent_activity_and_weights(
            restored, fx=fx, condition=condition, truncation_rank=1,
            input_timecourses=noise, savefig=False,
        )
        np.testing.assert_allclose(fig_activity.axes[0].images[0].get_array(), expected.T.numpy())
    finally:
        plt.close('all')

@pytest.mark.parametrize('version, recurrent_type', [(2, 'fullconn'), (3, 'fullconn'), (3, 'lowrank')])
def test_legacy_checkpoints_keep_zero_start(activity_data, version, recurrent_type):
    root, dataset = activity_data
    model = ln.LatentRNN(
        ln.dataset_session_shapes(dataset), conditions=('A1', 'P1'),
        n_initial_timepoints=None, recurrent_type=recurrent_type,
        recurrent_rank=2 if recurrent_type == 'lowrank' else None,
    )
    source = root / 'legacy.pt'
    ln.save_latent_network({'rnn': model, 'dataset': dataset, 'loss_history': []}, checkpoint_path=source)
    checkpoint = torch.load(source, weights_only=True)
    checkpoint['version'] = version
    checkpoint['config'].pop('n_initial_timepoints')
    if version == 2:
        checkpoint['config'].pop('recurrent_type')
        checkpoint['config'].pop('recurrent_rank')
    torch.save(checkpoint, source)
    original_bytes = source.read_bytes()
    loaded = ln.load_latent_network(source, training_path=root)['rnn']
    assert loaded.n_initial_timepoints is None
    noise = model.session_model(3).sample_input_timecourses()
    original = model(3, input_timecourses=noise)
    restored = loaded(3, input_timecourses=noise)
    for condition in original:
        torch.testing.assert_close(original[condition], restored[condition], rtol=0, atol=0)
        assert torch.count_nonzero(restored[condition][0]) == 0
    assert source.read_bytes() == original_bytes
