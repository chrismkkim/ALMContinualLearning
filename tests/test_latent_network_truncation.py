"""Behavior checks for SVD truncation during simulation of trained networks."""

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
def spectral_rnn(request):
    rnn = ln.LatentRNN(
        {3: (3, 12), 8: (5, 15)}, dt=1,
        recurrent_type=request.param,
        recurrent_rank=2 if request.param == 'lowrank' else None,
        n_initial_timepoints=None,
    ).double()
    session = rnn.session_model(3)
    # Orthogonal modes of sizes 3 and 1, with the SMALLER factor stored first.
    with torch.no_grad():
        session.input_vectors.fill_(0.2)
        if session.recurrent_type == 'fullconn':
            session.W.zero_()
            session.W[0, 1] = 3
            session.W[2, 0] = 1
        else:
            session.U.zero_()
            session.V.zero_()
            session.U[2, 0] = 1
            session.V[0, 0] = 1 / session.recurrent_scale
            session.U[0, 1] = 3
            session.V[1, 1] = 1 / session.recurrent_scale
    return rnn


def test_truncation_keeps_largest_singular_mode_and_preserves_parameters(spectral_rnn):
    rnn = spectral_rnn
    before = {key: value.clone() for key, value in rnn.state_dict().items()}
    expected = torch.zeros(3, 3, dtype=torch.float64)
    expected[0, 1] = 3
    truncated = rnn.truncated_recurrent_weight(3, np.int64(1))
    torch.testing.assert_close(truncated, expected)
    assert not truncated.requires_grad
    for rank in (2, 3):
        torch.testing.assert_close(
            rnn.truncated_recurrent_weight(3, rank), rnn.recurrent_weight(3),
        )
    for key, value in rnn.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_truncated_simulation_uses_one_svd_and_shared_noise_throughout(spectral_rnn, monkeypatch):
    rnn = spectral_rnn.eval()
    session = rnn.session_model(3)
    before = {key: value.clone() for key, value in rnn.state_dict().items()}
    noise = session.sample_input_timecourses()
    calls = []
    original_svd = torch.linalg.svd

    def counted_svd(*args, **kwargs):
        calls.append(1)
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, 'svd', counted_svd)
    rng_state = torch.random.get_rng_state()
    with torch.no_grad():
        original = rnn(3, input_timecourses=noise)
        truncated = rnn(3, truncation_rank=1, input_timecourses=noise)
    assert len(calls) == 1
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    for icond, condition in enumerate(rnn.conditions):
        inputs = session.condition_input(icond, noise).detach()
        # With dt=1 and W[0,1]=3, unit 0 receives unit 1's previous input.
        expected = torch.zeros(12, 3, dtype=torch.float64)
        expected[1:] = inputs[:-1]
        expected[2:, 0] += 3 * inputs[:-2, 1]
        torch.testing.assert_close(truncated[condition], expected)
        assert not torch.allclose(original[condition], truncated[condition])
    for key, value in rnn.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    with torch.no_grad():
        full_rank = rnn(3, truncation_rank=3, input_timecourses=noise)
        original_again = rnn(3, input_timecourses=noise)
        other_session = rnn(8, truncation_rank=4)
    for condition in rnn.conditions:
        torch.testing.assert_close(full_rank[condition], original[condition])
        torch.testing.assert_close(original_again[condition], original[condition], rtol=0, atol=0)
        assert other_session[condition].shape == (15, 5)
    # Omitted input_timecourses still samples fresh noise in truncated eval.
    with torch.no_grad():
        first = rnn(3, truncation_rank=1)['P1']
        second = rnn(3, truncation_rank=1)['P1']
    assert not torch.equal(first, second)


@pytest.mark.parametrize('rank', [0, -1, 4, 1.5, True, np.bool_(True)])
def test_invalid_truncation_rank_is_rejected(rank):
    session = ln.SessionRNN(3, 12, n_initial_timepoints=None).eval()
    with pytest.raises(ValueError, match='truncation_rank'):
        session(truncation_rank=rank)


def test_truncation_requires_eval_and_supplied_noise_remains_untrained(spectral_rnn):
    rnn = spectral_rnn
    with pytest.raises(RuntimeError, match=r'eval\(\)'):
        rnn(3, truncation_rank=1)
    noise = rnn.session_model(3).sample_input_timecourses().requires_grad_()
    rnn(3, input_timecourses=noise)['P1'].sum().backward()
    assert noise.grad is None
    assert rnn.session_model(3).input_vectors.grad is not None
    for invalid in (noise[..., :-1], torch.full_like(noise, float('nan'))):
        with pytest.raises(ValueError, match='input_timecourses'):
            rnn(3, input_timecourses=invalid)


def test_plots_use_truncated_dynamics_and_weights(spectral_rnn, tmp_path, monkeypatch):
    rnn = spectral_rnn
    monkeypatch.setattr(ln, 'SAVE_FIGURE_PATH', str(tmp_path))
    target = np.zeros((3, 12))
    target[np.arange(3), [7, 2, 4]] = 0.2
    results = {
        'rnn': rnn,
        'dataset': {'id': '0', 'activity': {3: {'P1': target}}},
    }
    noise = rnn.session_model(3).sample_input_timecourses()
    rnn.eval()
    with torch.no_grad():
        expected_activity = rnn(3, truncation_rank=1, input_timecourses=noise)['P1'].T.numpy()
    rnn.train()
    try:
        original_fig = ln.plot_prediction_vs_target(results, input_timecourses=noise)
        fig = ln.plot_prediction_vs_target(results, truncation_rank=1, input_timecourses=noise)
        assert rnn.training
        row_order = np.argsort(np.argmax(target, axis=1))
        np.testing.assert_allclose(fig.axes[1].images[0].get_array(), expected_activity[row_order])
        assert 'SVD rank 1' in fig._suptitle.get_text()
        assert 'SVD rank' not in original_fig._suptitle.get_text()
        activity_fig, weight_fig = ln.plot_latent_activity_and_weights(
            results, fx=3, truncation_rank=1, input_timecourses=noise,
        )
        assert rnn.training
        np.testing.assert_allclose(activity_fig.axes[0].images[0].get_array(), expected_activity)
        expected_weight = np.zeros((3, 3))
        expected_weight[0, 1] = 3
        np.testing.assert_allclose(weight_fig.axes[0].images[0].get_array(), expected_weight)
        assert 'SVD rank 1' in weight_fig.axes[0].get_title()
        for filename in (
            'compare_dataset0_fx3_P1.png',
            'compare_dataset0_fx3_P1_TruncRank1.png',
            'activity_dataset0_fx3_P1_TruncRank1.png',
            'W_dataset0_fx3_TruncRank1.png',
        ):
            assert (tmp_path / filename).is_file()
        with pytest.raises(ValueError, match='truncation_rank'):
            ln.plot_prediction_vs_target(results, truncation_rank=4, savefig=False)
        assert rnn.training
    finally:
        plt.close('all')
