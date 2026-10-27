"""Check notebook simulations with resampled recurrent and input weights."""

import json
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


@pytest.mark.parametrize('recurrent_type', ['fullconn', 'lowrank'])
@pytest.mark.parametrize('change_weights', [False, True])
@pytest.mark.parametrize('n_initial_timepoints', [None, 2])
def test_resampled_simulation_and_heatmaps(recurrent_type, change_weights, n_initial_timepoints, monkeypatch):
    model = ln.LatentRNN(
        {3: (4, 9), 8: (201, 12)}, conditions=('A1', 'P1'),
        n_input_components=2, dt=0.2,
        recurrent_type=recurrent_type,
        recurrent_rank=2 if recurrent_type == 'lowrank' else None,
        n_initial_timepoints=n_initial_timepoints,
    ).double()
    model.session_model(8).eval()  # Preserve distinct modes for the two sessions.
    dataset = {'activity': {
        fx: {condition: np.full(model.session_shapes[fx], 0.02 * (icond + 1))
             for icond, condition in enumerate(model.conditions)}
        for fx in model.session_ids
    }}
    prefixes = {
        fx: model.initial_activity_from_targets(fx, dataset['activity'][fx])
        for fx in model.session_ids
    }
    sampled, shared_noise, noise_calls, expected_models = {}, {}, {}, {}
    for fx in model.session_ids:
        session = model.session_model(fx)
        n = session.n_units
        u = np.linspace(0.1, 0.2, n)
        v = np.linspace(0.1, 0.3, n)
        singular_value = 0.2
        W = torch.from_numpy(singular_value * np.outer(u, v))
        with torch.no_grad():
            session.input_vectors.fill_(0.05)
            session.input_vectors[0, 1, 1].fill_(0.1)
            session.input_vectors[1, 1, 1].fill_(0.2)
            if recurrent_type == 'fullconn':
                session.W.copy_(W)
            else:
                session.U.zero_()
                session.V.zero_()
                session.U[:, 0].copy_(torch.from_numpy(u))
                session.V[:, 0].copy_(torch.from_numpy(v) * singular_value / session.recurrent_scale)
        shared_noise[fx] = session.sample_input_timecourses()
        noise_calls[fx] = 0

        def sample_once(fx=fx):
            noise_calls[fx] += 1
            return shared_noise[fx]

        monkeypatch.setattr(session, 'sample_input_timecourses', sample_once)
        sampled[fx] = {
            'U': u + (0.02 if change_weights else 0),
            'V': v * (0.8 if change_weights else 1),
            'sensory': np.full(n, 0.15 if change_weights else 0.1),
            'response': np.full(n, 0.4 if change_weights else 0.2),
            'singular_value': singular_value, 'condition': 'P1', 'input_component': 1,
        }
        # A separate dense model is an independent reference for either parameterization.
        reference = ln.SessionRNN(
            n, session.n_time, conditions=session.conditions, n_input_components=2,
            dt=session.dt, input_masks=session.input_masks,
            n_initial_timepoints=n_initial_timepoints,
        ).double().eval()
        with torch.no_grad():
            reference.input_vectors.copy_(session.input_vectors)
            reference.input_vectors[0, 1, 1].copy_(torch.from_numpy(sampled[fx]['sensory']))
            reference.input_vectors[1, 1, 1].copy_(torch.from_numpy(sampled[fx]['response']))
            reference.W.copy_(torch.from_numpy(singular_value * np.outer(sampled[fx]['U'], sampled[fx]['V'])))
        expected_models[fx] = reference

    before = {key: value.clone() for key, value in model.state_dict().items()}
    modes = {fx: model.session_model(fx).training for fx in model.session_ids}
    effdim = np.zeros(max(model.session_ids) + 1, dtype=int)
    effdim[list(model.session_ids)] = 2
    namespace = {'torch': torch, 'np': np, 'plt': plt, 'dataset': dataset, 'effdim': effdim,
                 'bgmm_rnn': model, 'resampled_connectivity': sampled}
    book = json.loads(Path(ln.__file__).with_name('_9_Lowdim_Dynamics.ipynb').read_text())
    source = next(''.join(c['source']) for c in book['cells'] if c.get('id') == 'resampled-network-activity')
    figures = []
    monkeypatch.setattr(plt, 'show', lambda: figures.append(plt.gcf()))
    try:
        exec(source, namespace)
        comparisons = namespace['resampled_activity_comparison']
        assert set(comparisons) == {3, 8}
        assert noise_calls == {3: 1, 8: 1}
        for fx, fig in zip(model.session_ids, figures):
            values = comparisons[fx]
            with torch.no_grad():
                original = model(fx, input_timecourses=shared_noise[fx], initial_activity=prefixes[fx])['P1'].T.numpy()
                expected = expected_models[fx](input_timecourses=shared_noise[fx], initial_activity=prefixes[fx])['P1'].T.numpy()
            np.testing.assert_allclose(values['original'], original, atol=1e-12)
            np.testing.assert_allclose(values['resampled'], expected, atol=1e-12)
            if change_weights:
                assert not np.allclose(original, values['resampled'])
            else:
                np.testing.assert_allclose(values['resampled'], original, atol=1e-12)
            if n_initial_timepoints is None:
                assert np.count_nonzero(values['resampled'][:, 0]) == 0
            else:
                np.testing.assert_allclose(values['resampled'][:, 0], 0.04)
            assert values['original'].shape == model.session_shapes[fx]
            order = values['neuron_order']
            np.testing.assert_array_equal(order, np.argsort(np.argmax(original, axis=1)))
            for ax, activity in zip(fig.axes[:2], [original, expected]):
                np.testing.assert_allclose(ax.images[0].get_array(), activity[order], atol=1e-12)
            assert fig.axes[0].images[0].get_clim() == fig.axes[1].images[0].get_clim()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        assert model.training
        assert {fx: model.session_model(fx).training for fx in model.session_ids} == modes
    finally:
        plt.close('all')
