"""Check the notebook's neuron features and Bayesian mixture model selection."""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _8_LatentNetwork as ln


def notebook_sources():
    book = json.loads(Path(ln.__file__).with_name('_9_Lowdim_Dynamics.ipynb').read_text())
    return {cell.get('id'): ''.join(cell['source']) for cell in book['cells']}


@pytest.fixture(scope='module')
def mixture_results():
    rng = np.random.default_rng(25)
    features, true_labels = {}, {}
    centers = np.array([[-4, -4, -4, -4], [4, 4, -4, -4], [0, 0, 5, 5]])
    for fx, k in [(3, 1), (8, 2), (17, 3)]:
        labels = np.arange(90 * k) % k
        X = rng.normal(scale=0.4, size=(len(labels), 4))
        if k > 1:
            X += centers[labels]
        features[fx], true_labels[fx] = X, labels
    model = ln.LatentRNN(
        {fx: (len(X), 12) for fx, X in features.items()},
        conditions=('A1', 'P1'), n_input_components=2,
        recurrent_type='lowrank', recurrent_rank=1,
    ).double()
    with torch.no_grad():
        for fx, X in features.items():
            session = model.session_model(fx)
            session.U[:, 0].copy_(torch.from_numpy(X[:, 0]))
            session.V[:, 0].copy_(torch.from_numpy(X[:, 1]) / session.recurrent_scale)
            session.input_vectors[0, 0, 1].copy_(torch.from_numpy(X[:, 2]))
            session.input_vectors[1, 0, 1].copy_(torch.from_numpy(X[:, 3]))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    sources = notebook_sources()
    namespace = {'np': np, 'plt': plt, 'rnn_lr': model, 'cond': 'A1'}
    exec(sources['bgmm-session-settings'], namespace)
    namespace['bgmm_input_component'] = 1
    # Record scaler inputs to check that CV never standardizes using test rows.
    base_scaler = namespace['StandardScaler']
    scaler_inputs = []

    class TrackingScaler(base_scaler):
        def fit(self, X, y=None, sample_weight=None):
            scaler_inputs.append(np.asarray(X).copy())
            return super().fit(X, y, sample_weight=sample_weight)

    namespace['StandardScaler'] = TrackingScaler
    with threadpool_limits(limits=1):
        exec(sources['bgmm-session-fit'], namespace)
    return namespace, features, true_labels, before, scaler_inputs


def test_features_preserve_neuron_condition_and_component_alignment(mixture_results):
    namespace, raw_features, _, before, _ = mixture_results
    results = namespace['bgmm_results']
    assert set(results) == {3, 8, 17}
    for fx, original in raw_features.items():
        result = results[fx]
        X = result['features']
        assert X.shape == original.shape
        assert result['condition'] == 'A1'
        assert result['input_component'] == 1
        np.testing.assert_array_equal(result['neuron_indices'], np.arange(len(X)))
        np.testing.assert_array_equal(X[:, 2:], original[:, 2:])
        # For an outer product, singular vectors equal the normalized factors
        # up to a single shared sign; this checks all neuron entries in order.
        for index in (0, 1):
            expected = original[:, index] / np.linalg.norm(original[:, index])
            sign = np.sign(X[:, index] @ expected)
            np.testing.assert_allclose(X[:, index], sign * expected, atol=1e-12)
    model = namespace['rnn_lr']
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_cross_validation_selects_known_mixtures_and_recovers_groups(mixture_results):
    from sklearn.metrics import adjusted_rand_score

    namespace, _, true_labels, _, _ = mixture_results
    summary = namespace['bgmm_summary']
    assert len(summary) == 9
    for fx, k in [(3, 1), (8, 2), (17, 3)]:
        result = namespace['bgmm_results'][fx]
        assert result['best_n_components'] == k
        assert set(result['fits']) == {1, 2, 3}
        assert adjusted_rand_score(true_labels[fx], result['labels']) > 0.95
        assert result['probabilities'].shape == (len(true_labels[fx]), k)
        np.testing.assert_allclose(result['probabilities'].sum(axis=1), 1)
        rows = summary[summary['session'] == fx]
        assert rows['selected'].sum() == 1
        assert rows.loc[rows['selected'], 'n_components'].iloc[0] == k
        assert np.isfinite(rows['cv_mean_log_likelihood']).all()
        assert rows['cv_converged'].all() and rows['converged'].all()
        assert all(len(fit['fold_log_likelihoods']) == 5 for fit in result['fits'].values())


def test_cross_validation_scaling_uses_only_training_neurons(mixture_results):
    from sklearn.model_selection import KFold

    namespace, _, _, _, scaler_inputs = mixture_results
    assert len(scaler_inputs) == 6 * 3  # One refit scaler and five fold scalers per session.
    for index, result in enumerate(namespace['bgmm_results'].values()):
        X = result['features']
        np.testing.assert_array_equal(scaler_inputs[6 * index], X)
        splits = KFold(n_splits=5, shuffle=True, random_state=0).split(X)
        for fold, (train, _) in enumerate(splits):
            np.testing.assert_array_equal(scaler_inputs[6 * index + fold + 1], X[train])


def test_goodness_of_fit_plot_contains_each_session(mixture_results, monkeypatch):
    namespace = mixture_results[0]
    monkeypatch.setattr(plt, 'show', lambda: None)
    try:
        exec(notebook_sources()['bgmm-session-plots'], namespace)
        axes = [ax for ax in namespace['fig'].axes if ax.get_visible()]
        assert len(axes) == 3
        for ax, fx in zip(axes, [3, 8, 17]):
            assert f'Session {fx};' in ax.get_title()
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 3])
    finally:
        plt.close('all')


def test_small_sessions_are_reported_without_invalid_fits(monkeypatch):
    model = ln.LatentRNN(
        {2: (1, 12), 7: (2, 12), 11: (3, 12)},
        recurrent_type='lowrank', recurrent_rank=1,
    )
    namespace = {'np': np, 'plt': plt, 'rnn_lr': model, 'cond': 'P1'}
    sources = notebook_sources()
    exec(sources['bgmm-session-settings'], namespace)
    with threadpool_limits(limits=1):
        exec(sources['bgmm-session-fit'], namespace)
    results = namespace['bgmm_results']
    assert results[2]['fits'] == {}
    assert results[2]['best_n_components'] is None
    assert set(results[7]['fits']) == {1, 2}
    assert results[7]['best_n_components'] is None
    assert results[11]['fits'][3]['metrics']['status'] == 'too few neurons for cross-validation'
    assert not results[11]['fits'][3]['metrics']['selected']
    monkeypatch.setattr(plt, 'show', lambda: None)
    try:
        exec(sources['bgmm-session-plots'], namespace)
    finally:
        plt.close('all')
