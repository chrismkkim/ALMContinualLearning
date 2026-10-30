"""Centered readout reversal with optional left-factor training and fresh noise."""

from pathlib import Path
import runpy
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _8_LatentNetwork as ln


@pytest.fixture(params=['fullconn', 'lowrank'])
def source_model(request, tmp_path):
    torch.manual_seed(61)
    root = tmp_path / 'training'
    root.mkdir()
    rng = np.random.default_rng(19)
    activity = {
        fx: {condition: rng.uniform(0.1, 0.3, (n_units, 24)) for condition in ('P1', 'A1')}
        for fx, n_units in ((3, 4), (8, 6))
    }
    np.save(root / 'activity0.npy', activity)
    source = tmp_path / 'source.pt'
    results = ln.train_latent_network(
        n_steps=1, training_path=root, checkpoint_path=source,
        recurrent_type=request.param,
        recurrent_rank=2 if request.param == 'lowrank' else None,
        n_input_components=2, input_noise_std=0.3,
    )
    return root, source, results


def initialize_session(n_units=4):
    torch.manual_seed(24)
    session = ln.SessionRNN(n_units, 24, n_initial_timepoints=None).double()
    reference = ln.compute_context1_reference(session)
    session.initialize_readout_retraining(reference)
    return session


def test_context1_reference_uses_exact_four_bins_and_no_training(monkeypatch):
    session = ln.SessionRNN(3, 24, n_initial_timepoints=None).double()
    before = {name: value.clone() for name, value in session.state_dict().items()}
    p = torch.arange(24, dtype=torch.float64)[:, None] * torch.tensor([1., 2., 3.])
    a = torch.ones_like(p)
    p[16] = 1e6
    monkeypatch.setattr(session, 'forward', lambda **kwargs: {'P1': p, 'A1': a})
    ref = ln.compute_context1_reference(session)
    torch.testing.assert_close(ref['rP1'], torch.tensor([13.5, 27., 40.5], dtype=torch.float64))
    torch.testing.assert_close(ref['rA1'], torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(ref['zP1'], ref['cd1'].norm() / 2)
    torch.testing.assert_close(ref['zA1'], -ref['cd1'].norm() / 2)
    assert all(not value.requires_grad for value in ref.values())
    for name, value in before.items():
        torch.testing.assert_close(session.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize('train_left_vectors', [False, True])
def test_svd_initialization_reconstructs_source_and_selects_left_training(source_model, train_left_vectors):
    _, _, results = source_model
    for fx in results['rnn'].session_ids:
        session = results['rnn'].session_model(fx)
        W0 = session.recurrent_weight().detach().clone()
        initial = session.initial_activity_from_targets(results['dataset']['activity'][fx])
        ref = ln.compute_context1_reference(session, initial)
        session.initialize_readout_retraining(ref, initial, train_left_vectors=train_left_vectors)
        torch.testing.assert_close(session.recurrent_weight(), W0, rtol=1e-5, atol=1e-7)
        expected_trainable = {'alpha', 'u', 'svd_values', 'svd_right'}
        if train_left_vectors:
            expected_trainable.add('svd_left')
        assert {name for name, p in session.named_parameters() if p.requires_grad} == expected_trainable
        assert session.svd_left.requires_grad == train_left_vectors
        k = session.n_units if session.recurrent_type == 'fullconn' else session.recurrent_rank
        assert session.svd_left.shape == session.svd_right.shape == (session.n_units, k)
        torch.testing.assert_close(session.svd_values, torch.linalg.svdvals(W0)[:k])
        left = session.svd_left.clone()
        with torch.no_grad():
            session.svd_values.neg_()
            session.svd_right.add_(0.37)
        assert (session.svd_values < 0).all()
        assert not torch.allclose(session.svd_right.T @ session.svd_right, torch.eye(k))
        torch.testing.assert_close(session.svd_left, left, rtol=0, atol=0)
        rates = torch.rand(2, session.n_units, requires_grad=True)
        drive = session.recurrent_drive(rates, train_recurrent=True)
        torch.testing.assert_close(drive, rates @ session.recurrent_weight().T)
        recurrent_parameters = [session.svd_values, session.svd_right]
        if train_left_vectors:
            recurrent_parameters.append(session.svd_left)
        grads = torch.autograd.grad(drive.sum(), recurrent_parameters)
        assert all(torch.isfinite(grad).all() and torch.count_nonzero(grad) for grad in grads)
        detached = session.recurrent_drive(rates, train_recurrent=False)
        grads = torch.autograd.grad(
            detached.sum(), (*recurrent_parameters, rates), allow_unused=True,
        )
        assert all(grad is None for grad in grads[:-1])
        assert grads[-1] is not None


def test_context2_projection_and_normalization_gradients():
    session = initialize_session()
    with torch.no_grad():
        session.alpha.fill_(0.7)
        session.u.copy_(torch.tensor([0.4, -0.1, 0.2, 0.3]))
    d = torch.tensor([0.3, -0.2, 0.1, 0.5], dtype=torch.float64, requires_grad=True)
    p = (torch.ones_like(d) + d / 2).repeat(24, 1)
    a = (torch.ones_like(d) - d / 2).repeat(24, 1)
    # Bin 16 must not contribute to any context 2 quantity.
    p = torch.cat((p[:16], p[16:] + 1000))
    out = session.context2_readout({'P1': p, 'A1': a})
    torch.testing.assert_close(out['cd2'], d)
    torch.testing.assert_close(out['zP2'], -out['zA2'])
    c = session.reference_cd1
    torch.testing.assert_close(torch.dot(c, out['u_ortho']), c.new_tensor(0.), atol=1e-15, rtol=0)
    grad_u, grad_d = torch.autograd.grad(out['zP2'], (session.u, d), retain_graph=True)
    P = torch.eye(4, dtype=torch.float64) - torch.outer(c, c) / c.square().sum()
    torch.testing.assert_close(grad_u, P @ d / (2 * d.norm()))
    w = out['w_o']
    expected_grad_d = (w / d.norm() - torch.dot(w, d) * d / d.norm().pow(3)) / 2
    torch.testing.assert_close(grad_d, expected_grad_d)
    assert torch.count_nonzero(grad_u) > 0
    losses = ln.readout_reversal_loss(session, out, lambda_readout=3, lambda_coding=7)
    expected_reverse = (
        (session.reference_zP1 - out['zA2']).square()
        + (session.reference_zA1 - out['zP2']).square()
    )
    torch.testing.assert_close(losses['loss_reverse'], expected_reverse)
    torch.testing.assert_close(
        losses['loss'], expected_reverse + 3 * losses['readout_residual'].square()
        + 7 * losses['coding_residual'].square(),
    )


def test_zero_cd2_is_guarded_and_constraint_uses_actual_norm():
    session = initialize_session()
    p = torch.ones(24, 4, dtype=torch.float64, requires_grad=True)
    a = torch.ones_like(p, requires_grad=True)
    out = session.context2_readout({'P1': p, 'A1': a})
    assert out['cd2_clamped']
    assert out['norm_cd2'] == 0
    torch.testing.assert_close(
        out['norm_cd2_safe'], session.min_cd2_norm_ratio * session.reference_norm_cd1,
    )
    losses = ln.readout_reversal_loss(session, out)
    assert losses['coding_residual'] == -1
    assert torch.isfinite(losses['loss'])
    grads = torch.autograd.grad(losses['loss'], (p, a, session.u, session.alpha))
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_reversed_readout_satisfies_both_constraints():
    session = initialize_session()
    with torch.no_grad():
        session.alpha.fill_(-1)
    out = session.context2_readout({
        'P1': session.reference_rP1.repeat(24, 1),
        'A1': session.reference_rA1.repeat(24, 1),
    })
    losses = ln.readout_reversal_loss(session, out)
    torch.testing.assert_close(out['zP2'], session.reference_zA1)
    torch.testing.assert_close(out['zA2'], session.reference_zP1)
    torch.testing.assert_close(losses['loss'], session.alpha.new_tensor(0.), atol=1e-25, rtol=0)


@pytest.mark.parametrize('train_left_vectors', [False, True])
def test_retraining_updates_selected_parameters_and_restores_checkpoint(source_model, monkeypatch, train_left_vectors):
    root, source, original = source_model
    before_bytes = source.read_bytes()
    calls = []
    initial_states = {}
    sample = ln.SessionRNN.sample_input_timecourses
    initialize = ln.SessionRNN.initialize_readout_retraining

    def tracked_initialize(session, *args, **kwargs):
        source_weight = session.recurrent_weight().detach().clone()
        initialize(session, *args, **kwargs)
        initial_states[session.n_units] = {
            'source_weight': source_weight,
            'svd_left': session.svd_left.detach().clone(),
            'svd_values': session.svd_values.detach().clone(),
            'svd_right': session.svd_right.detach().clone(),
            'references': {name: getattr(session, 'reference_' + name).clone()
                           for name in ln.CONTEXT1_REFERENCE_KEYS},
        }

    def tracked_sample(session):
        values = sample(session)
        calls.append((session.n_units, values.detach().clone()))
        return values

    monkeypatch.setattr(ln.SessionRNN, 'sample_input_timecourses', tracked_sample)
    monkeypatch.setattr(ln.SessionRNN, 'initialize_readout_retraining', tracked_initialize)
    torch.manual_seed(94)
    result = ln.retrain_latent_network_readout(
        source, training_path=root, n_steps=3,
        learning_rate_alpha=0.01, learning_rate_u=0.02,
        learning_rate_singular_values=0.003, learning_rate_right_vectors=0.004,
        train_left_vectors=train_left_vectors, learning_rate_left_vectors=0.005,
        lambda_readout=2, lambda_coding=3,
    )
    assert source.read_bytes() == before_bytes
    assert result['rnn'].input_noise_std == 0.3  # Inherit the source noise level.
    assert result['checkpoint_path'] == str(source.with_name('source_retrained_readout_svd.pt'))
    assert len(calls) == 2 * (1 + 3)  # One reference and three training draws per session.
    for n_units in (4, 6):
        draws = [values for n, values in calls if n == n_units]
        assert len(draws) == 4
        assert all(not torch.equal(first, second) for first, second in zip(draws, draws[1:]))
        assert all(not torch.equal(draw[..., 0], draw[..., 1]) for draw in draws)
        assert all(not torch.equal(draw[:, 0], draw[:, 1]) for draw in draws)
        assert all(not torch.equal(draw[:, :, 0], draw[:, :, 1]) for draw in draws)
    assert not torch.equal(calls[0][1], calls[1][1])  # Different sessions draw independently.
    assert np.isfinite(result['loss_history']).all()
    np.testing.assert_allclose(result['loss_history'], np.mean(list(result['loss_history_by_session'].values()), axis=0))
    for fx in result['rnn'].session_ids:
        session = result['rnn'].session_model(fx)
        source_session = original['rnn'].session_model(fx)
        for name, parameter in source_session.named_parameters():
            torch.testing.assert_close(dict(session.named_parameters())[name], parameter, rtol=0, atol=0)
            assert not dict(session.named_parameters())[name].requires_grad
        W0 = source_session.recurrent_weight().detach()
        initial = initial_states[session.n_units]
        torch.testing.assert_close(initial['source_weight'], W0, rtol=0, atol=0)
        if train_left_vectors:
            assert not torch.allclose(session.svd_left, initial['svd_left'])
            assert session.svd_left.grad is not None
            assert torch.count_nonzero(session.svd_left.grad) > 0
        else:
            torch.testing.assert_close(session.svd_left, initial['svd_left'], rtol=0, atol=0)
            assert session.svd_left.grad is None
        for name, value in initial['references'].items():
            torch.testing.assert_close(getattr(session, 'reference_' + name), value, rtol=0, atol=0)
        assert not torch.allclose(session.svd_values, initial['svd_values'])
        assert not torch.allclose(session.svd_right, initial['svd_right'])
        assert not torch.allclose(session.recurrent_weight(), W0)
        assert session.alpha.item() != 1
        assert torch.count_nonzero(session.u) > 0
        assert len(result['readout_history_by_session'][fx]) == 3
    saved = torch.load(result['checkpoint_path'], weights_only=True)
    assert saved['version'] == 6
    assert saved['readout_retraining']['readout_window'] == [12, 16]
    assert saved['readout_retraining']['train_left_vectors'] is train_left_vectors
    assert saved['readout_retraining']['method'] == (
        'unconstrained_svd' if train_left_vectors else 'fixed_left_unconstrained_svd'
    )
    expected_rates = {
        'alpha': 0.01, 'u': 0.02, 'svd_values': 0.003, 'svd_right': 0.004,
    }
    if train_left_vectors:
        expected_rates['svd_left'] = 0.005
    assert saved['readout_retraining']['learning_rates'] == expected_rates
    calls.clear()
    restored = ln.load_latent_network(result['checkpoint_path'], training_path=root)
    assert not calls  # Loading must neither regenerate references nor draw noise.
    assert restored['readout_history_by_session'] == result['readout_history_by_session']
    for name, value in result['rnn'].state_dict().items():
        torch.testing.assert_close(restored['rnn'].state_dict()[name], value, rtol=0, atol=0)
    for fx in result['rnn'].session_ids:
        actual_session = restored['rnn'].session_model(fx)
        expected_trainable = {'alpha', 'u', 'svd_values', 'svd_right'}
        if train_left_vectors:
            expected_trainable.add('svd_left')
        assert {name for name, p in actual_session.named_parameters() if p.requires_grad} == expected_trainable
        torch.manual_seed(13)
        expected = result['rnn'].readout(fx)
        torch.manual_seed(13)
        actual = restored['rnn'].readout(fx)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        first, second = restored['rnn'].readout(fx), restored['rnn'].readout(fx)
        assert not torch.equal(first['cd2'], second['cd2'])


def test_readout_checkpoint_without_left_training_flag_loads_frozen(source_model):
    root, source, _ = source_model
    result = ln.retrain_latent_network_readout(source, training_path=root, n_steps=1)
    checkpoint = torch.load(result['checkpoint_path'], weights_only=True)
    checkpoint['readout_retraining'].pop('train_left_vectors')
    torch.save(checkpoint, result['checkpoint_path'])
    restored = ln.load_latent_network(result['checkpoint_path'], training_path=root)
    for fx in restored['rnn'].session_ids:
        session = restored['rnn'].session_model(fx)
        assert not session.svd_left.requires_grad
        torch.testing.assert_close(
            session.svd_left, result['rnn'].session_model(fx).svd_left, rtol=0, atol=0,
        )


@pytest.mark.parametrize('train_layer, train_sensory', [(True, False), (False, True), (True, True)])
def test_retraining_existing_layer_and_sensory_vectors(source_model, train_layer, train_sensory):
    root, source, original = source_model
    before_bytes = source.read_bytes()
    torch.manual_seed(94)
    result = ln.retrain_latent_network_readout(
        source, training_path=root, n_steps=3,
        train_initial_state_layer=train_layer, train_sensory_input=train_sensory,
        learning_rate_initial_state_layer=0.0001, learning_rate_sensory_input=0.002,
    )
    restored = ln.load_latent_network(result['checkpoint_path'], training_path=root)
    assert source.read_bytes() == before_bytes
    metadata = restored['readout_retraining']
    assert metadata['train_initial_state_layer'] == train_layer
    assert metadata['train_sensory_input'] == train_sensory
    if train_layer:
        assert metadata['learning_rates']['initial_state_layer'] == 0.0001
    if train_sensory:
        assert metadata['learning_rates']['input_vectors'] == 0.002
    for fx in result['rnn'].session_ids:
        session = result['rnn'].session_model(fx)
        source_session = original['rnn'].session_model(fx)
        saved_session = restored['rnn'].session_model(fx)
        expected_trainable = {'alpha', 'u', 'svd_values', 'svd_right'}
        if train_layer:
            expected_trainable.update({'initial_state_layer.weight', 'initial_state_layer.bias'})
        if train_sensory:
            expected_trainable.add('input_vectors')
        assert {name for name, p in session.named_parameters() if p.requires_grad} == expected_trainable
        assert {name for name, p in saved_session.named_parameters() if p.requires_grad} == expected_trainable
        for name, parameter in source_session.named_parameters():
            actual = dict(session.named_parameters())[name]
            if name.startswith('initial_state_layer.') and train_layer:
                assert actual.grad is not None and torch.isfinite(actual.grad).all()
                if name.endswith('weight'):
                    assert not torch.equal(actual, parameter)
            elif name == 'input_vectors' and train_sensory:
                assert not torch.equal(actual[0], parameter[0])
                torch.testing.assert_close(actual[1:], parameter[1:], rtol=0, atol=0)
                assert torch.count_nonzero(actual.grad[1:]) == 0
            else:
                torch.testing.assert_close(actual, parameter, rtol=0, atol=0)
        initial = source_session.initial_activity_from_targets(original['dataset']['activity'][fx])
        torch.testing.assert_close(session.readout_initial_activity, initial, rtol=0, atol=0)
        assert not session.readout_initial_activity.requires_grad
        assert torch.equal(session.initial_state(initial), source_session.initial_state(initial)) != train_layer
        for name, value in session.state_dict().items():
            torch.testing.assert_close(saved_session.state_dict()[name], value, rtol=0, atol=0)


def test_response_input_stays_fixed_even_with_nonzero_response_gradients(source_model, monkeypatch):
    root, source, original = source_model
    checkpoint = torch.load(source, weights_only=True)
    # Exercise the mask when response inputs affect the readout window, too.
    for fx in original['rnn'].session_ids:
        checkpoint['rnn_state_dict'][f'sessions.{fx}.input_masks'][1, :] = 1
    active_response_source = root / 'active_response.pt'
    torch.save(checkpoint, active_response_source)
    response_gradients = []
    initialize = ln.SessionRNN.initialize_readout_retraining

    def tracked_initialize(session, *args, **kwargs):
        initialize(session, *args, **kwargs)
        session.input_vectors.register_hook(
            lambda grad: response_gradients.append(grad[1:].detach().clone())
        )

    monkeypatch.setattr(ln.SessionRNN, 'initialize_readout_retraining', tracked_initialize)
    result = ln.retrain_latent_network_readout(
        active_response_source, training_path=root, n_steps=3, train_sensory_input=True,
    )
    assert len(response_gradients) == 3 * original['rnn'].n_sessions
    assert all(torch.count_nonzero(grad) > 0 for grad in response_gradients)
    for fx in result['rnn'].session_ids:
        session = result['rnn'].session_model(fx)
        source_session = original['rnn'].session_model(fx)
        torch.testing.assert_close(session.input_vectors[1:], source_session.input_vectors[1:], rtol=0, atol=0)
        assert not torch.equal(session.input_vectors[0], source_session.input_vectors[0])


def test_retraining_rejects_source_overwrite_before_sampling(source_model, monkeypatch):
    root, source, _ = source_model
    before = source.read_bytes()
    def unexpected_sample(*args):
        pytest.fail('Overwrite must be rejected before reference simulation.')
    monkeypatch.setattr(ln.SessionRNN, 'sample_input_timecourses', unexpected_sample)
    with pytest.raises(ValueError, match='must differ'):
        ln.retrain_latent_network_readout(source, training_path=root, checkpoint_path=source, n_steps=1)
    assert source.read_bytes() == before


@pytest.mark.parametrize('noise_std', [0, -1, float('nan')])
def test_retraining_requires_positive_fresh_noise(source_model, noise_std):
    root, source, _ = source_model
    with pytest.raises(ValueError, match='input_noise_std'):
        ln.retrain_latent_network_readout(source, training_path=root, n_steps=1, input_noise_std=noise_std)


def test_legacy_zero_initial_state_retrains_and_restores(source_model):
    root, source, _ = source_model
    checkpoint = torch.load(source, weights_only=True)
    checkpoint['version'] = 3
    checkpoint['config'].pop('n_initial_timepoints')
    for fx, (n_units, _) in checkpoint['config']['session_shapes'].items():
        prefix = f'sessions.{fx}.'
        checkpoint['rnn_state_dict'].pop(prefix + 'initial_state_layer.weight')
        checkpoint['rnn_state_dict'].pop(prefix + 'initial_state_layer.bias')
        checkpoint['rnn_state_dict'][prefix + 'h0'] = torch.zeros(2, n_units)
    legacy_source = root / 'legacy.pt'
    torch.save(checkpoint, legacy_source)
    result = ln.retrain_latent_network_readout(legacy_source, training_path=root, n_steps=1)
    restored = ln.load_latent_network(result['checkpoint_path'], training_path=root)
    assert restored['rnn'].n_initial_timepoints is None
    for fx in restored['rnn'].session_ids:
        session = restored['rnn'].session_model(fx)
        assert session.readout_initial_activity is None
        assert session.initial_state_layer is None
        assert torch.isfinite(session.readout()['zP2'])
    with pytest.raises(ValueError, match='no initial-state layer'):
        ln.retrain_latent_network_readout(
            legacy_source, training_path=root, n_steps=1, train_initial_state_layer=True,
        )


def test_cli_retrains_existing_checkpoint(source_model, monkeypatch, capsys):
    root, source, _ = source_model
    output = root / 'new_readout.pt'
    monkeypatch.setattr(sys, 'argv', [
        ln.__file__, '--retrain_readout', '--source_checkpoint', str(source),
        '--training_path', str(root), '--checkpoint_path', str(output),
        '--n_steps', '2', '--learning_rate', '0.005',
        '--learning_rate_alpha', '0.01', '--learning_rate_u', '0.02',
        '--learning_rate_singular_values', '0.003', '--learning_rate_right_vectors', '0.004',
        '--train_left_vectors', '--learning_rate_left_vectors', '0.006',
        '--train_initial_state_layer', '--train_sensory_input',
        '--learning_rate_initial_state_layer', '0.0001', '--learning_rate_sensory_input', '0.002',
        '--lambda_readout', '2', '--lambda_coding', '3',
        '--min_cd2_norm_ratio', '0.0001', '--input_noise_std', '0.2',
    ])
    namespace = runpy.run_path(ln.__file__, run_name='__main__')
    result = namespace['retrained_results']
    assert result['checkpoint_path'] == str(output)
    assert result['rnn'].input_noise_std == 0.2
    assert all(session.input_noise_std == 0.2 for session in result['rnn'].sessions.values())
    assert result['readout_retraining']['min_cd2_norm_ratio'] == 0.0001
    assert result['readout_retraining']['lambda_readout'] == 2
    assert result['readout_retraining']['lambda_coding'] == 3
    assert result['readout_retraining']['train_initial_state_layer'] is True
    assert result['readout_retraining']['train_sensory_input'] is True
    assert result['readout_retraining']['train_left_vectors'] is True
    assert result['readout_retraining']['learning_rates']['svd_left'] == 0.006
    assert result['readout_retraining']['learning_rates']['initial_state_layer'] == 0.0001
    assert result['readout_retraining']['learning_rates']['input_vectors'] == 0.002
    assert 'cd2 clamps' in capsys.readouterr().out
    restored = ln.load_latent_network(output, training_path=root)
    assert restored['rnn'].input_noise_std == 0.2
    assert all(session.svd_left.requires_grad for session in restored['rnn'].sessions.values())


def test_retraining_entry_point_and_cli_help(monkeypatch, capsys):
    assert callable(getattr(ln, 'retrain_latent_network_readout', None))
    monkeypatch.setattr(sys, 'argv', [ln.__file__, '--help'])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(ln.__file__, run_name='__main__')
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    for flag in (
        '--retrain_readout', '--source_checkpoint', '--learning_rate_alpha',
        '--learning_rate_u', '--learning_rate_singular_values',
        '--learning_rate_right_vectors', '--lambda_readout', '--lambda_coding',
        '--train_left_vectors', '--learning_rate_left_vectors',
        '--min_cd2_norm_ratio', '--train_initial_state_layer', '--train_sensory_input',
        '--learning_rate_initial_state_layer', '--learning_rate_sensory_input',
    ):
        assert flag in help_text


@pytest.mark.parametrize('train_left_vectors', [False, True])
def test_cli_shared_learning_rate_and_inherited_noise(source_model, monkeypatch, train_left_vectors):
    root, source, _ = source_model
    arguments = [
        ln.__file__, '--retrain_readout', '--source_checkpoint', str(source),
        '--training_path', str(root), '--n_steps', '1',
        '--learning_rate', '0.007', '--learning_rate_u', '0.009',
    ]
    if train_left_vectors:
        arguments.append('--train_left_vectors')
    monkeypatch.setattr(sys, 'argv', arguments)
    result = runpy.run_path(ln.__file__, run_name='__main__')['retrained_results']
    assert result['rnn'].input_noise_std == 0.3
    expected_rates = {
        'alpha': 0.007, 'u': 0.009, 'svd_values': 0.007, 'svd_right': 0.007,
    }
    if train_left_vectors:
        expected_rates['svd_left'] = 0.007
    assert result['readout_retraining']['learning_rates'] == expected_rates


@pytest.mark.parametrize('arguments, message', [
    (['--retrain_readout'], 'requires --source_checkpoint'),
    (['--source_checkpoint', 'unused.pt'], 'requires --retrain_readout'),
    (['--learning_rate_u', '0.01'], 'require --retrain_readout'),
    (['--train_initial_state_layer'], 'require --retrain_readout'),
    (['--train_sensory_input'], 'require --retrain_readout'),
    (['--train_left_vectors'], 'requires --retrain_readout'),
    (['--learning_rate_left_vectors', '0.001'], 'require --retrain_readout'),
    (['--retrain_readout', '--source_checkpoint', 'unused.pt',
      '--learning_rate_left_vectors', '0.001'], 'requires --train_left_vectors'),
    (['--retrain_readout', '--source_checkpoint', 'unused.pt',
      '--learning_rate_initial_state_layer', '0.001'], 'requires --train_initial_state_layer'),
    (['--retrain_readout', '--source_checkpoint', 'unused.pt',
      '--learning_rate_sensory_input', '0.001'], 'requires --train_sensory_input'),
])
def test_cli_requires_retraining_mode_and_source(arguments, message, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', [ln.__file__, *arguments])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(ln.__file__, run_name='__main__')
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('kwargs', [
    {'n_steps': 0}, {'learning_rate_alpha': float('nan')},
    {'learning_rate_u': -1}, {'learning_rate_singular_values': 0},
    {'learning_rate_right_vectors': float('inf')}, {'lambda_readout': -1},
    {'lambda_coding': -1}, {'min_cd2_norm_ratio': 0},
    {'train_initial_state_layer': True, 'learning_rate_initial_state_layer': 0},
    {'train_sensory_input': True, 'learning_rate_sensory_input': float('nan')},
    {'train_left_vectors': True, 'learning_rate_left_vectors': 0},
    {'train_left_vectors': True, 'learning_rate_left_vectors': -1},
    {'train_left_vectors': True, 'learning_rate_left_vectors': float('nan')},
])
def test_invalid_retraining_options_fail_before_loading(kwargs):
    with pytest.raises(ValueError):
        ln.retrain_latent_network_readout('nonexistent.pt', **kwargs)


def test_short_sessions_are_rejected():
    session = ln.SessionRNN(3, 15, n_initial_timepoints=None)
    with pytest.raises(ValueError, match='at least 16'):
        ln.compute_context1_reference(session)
