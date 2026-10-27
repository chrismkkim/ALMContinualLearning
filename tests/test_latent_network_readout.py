"""Readout reversal, frozen drives, optimizer calibration, and checkpoint reuse."""

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
def source_network(tmp_path, request):
    """Two differently sized sessions with only P1/A1 observations."""
    root = tmp_path / 'recurrent'
    data_root = root / 'training_data'
    data_root.mkdir(parents=True)
    rng = np.random.default_rng(23)
    activity = {
        fx: {condition: rng.uniform(0.1, 0.3, (n, t)).astype(np.float32)
             for condition in ('P1', 'A1')}
        for fx, n, t in ((3, 4, 24), (11, 7, 27))
    }
    np.save(data_root / 'activity0.npy', activity)
    dataset = ln.load_training_data(root)
    torch.manual_seed(29)
    rnn = ln.LatentRNN(
        ln.dataset_session_shapes(dataset), conditions=('A1', 'P1'),
        recurrent_type=request.param, recurrent_rank=2 if request.param == 'lowrank' else None,
    )
    with torch.no_grad():
        for fx in rnn.session_ids:
            session = rnn.session_model(fx)
            for parameter in session.recurrent_parameters():
                parameter.uniform_(0.01, 0.15)
            for index, condition in enumerate(session.conditions):
                pattern = torch.linspace(0.05, 0.3, session.n_units)
                if condition == 'A1':
                    pattern = pattern.flip(0) + 0.03
                session.input_vectors[:, index].copy_(pattern.expand(2, 1, -1))
    source = root / request.param / 'original.pt'
    ln.save_latent_network({'rnn': rnn, 'dataset': dataset, 'loss_history': []}, checkpoint_path=source)
    return root, source, rnn


def initialize_session_readout(session, targets):
    with torch.no_grad():
        initial = session.initial_activity_from_targets(targets)
        noise = session.sample_input_timecourses()
        original = session(initial_activity=initial, input_timecourses=noise)
        p = original['P1'][12:16].mean(0)
        a = original['A1'][12:16].mean(0)
        cd = p - a
    session.initialize_readout(cd, torch.stack([cd @ a, cd @ p]), noise, initial)


def test_reference_targets_and_only_requested_parameters_change(source_network):
    root, source, original = source_network
    source_bytes = source.read_bytes()
    results = ln.retrain_latent_network_readout(
        source, n_steps=1, learning_rate_recurrent=1e-3, learning_rate_readout=1e-4,
        training_path=root,
    )
    assert source.read_bytes() == source_bytes
    assert Path(results['checkpoint_path']) == source.with_name('original_retrained_readout.pt')
    assert results['rnn'].conditions == ('A1', 'P1')
    metadata = results['readout_retraining']
    for fx in original.session_ids:
        session = results['rnn'].session_model(fx)
        with torch.no_grad():
            reference = original(
                fx, input_timecourses=session.readout_input_timecourses,
                initial_activity=session.readout_initial_activity,
            )
            p = reference['P1'][12:16].mean(0)
            a = reference['A1'][12:16].mean(0)
            expected_cd = p - a
            expected_targets = torch.stack([expected_cd @ a, expected_cd @ p])
        torch.testing.assert_close(session.CD1, expected_cd, rtol=0, atol=0)
        torch.testing.assert_close(session.readout_targets, expected_targets, rtol=0, atol=0)
        assert session.Vout.shape == (session.n_units,)
        assert not torch.equal(session.Vout, session.CD1)
        torch.testing.assert_close(
            torch.tensor(metadata['readout_history_by_session'][fx][0]), expected_targets.flip(0),
        )
        # At initialization the two reversed residuals are +/- ||CD1||^2.
        assert results['loss_history_by_session'][fx][0] == pytest.approx(
            2 * expected_cd.square().sum().item() ** 2, rel=1e-5,
        )
        recurrent_names = {'W'} if session.recurrent_type == 'fullconn' else {'U', 'V'}
        original_parameters = dict(original.session_model(fx).named_parameters())
        for name, parameter in session.named_parameters():
            if name in recurrent_names or name == 'Vout':
                assert parameter.requires_grad
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
                if name in recurrent_names:
                    assert not torch.equal(parameter, original_parameters[name])
            else:
                assert not parameter.requires_grad
                assert parameter.grad is None
                torch.testing.assert_close(parameter, original_parameters[name], rtol=0, atol=0)
        for buffer in (session.CD1, session.readout_targets, session.readout_input_timecourses,
                       session.readout_initial_activity):
            assert not buffer.requires_grad


def test_calibration_restores_model_and_matches_actual_adam_effects(source_network):
    root, _, original = source_network
    session = original.session_model(3)
    initialize_session_readout(session, ln.load_training_data(root)['activity'][3])
    assert torch.equal(session.Vout, session.CD1)
    before = {key: value.clone() for key, value in session.state_dict().items()}
    rng_before = torch.random.get_rng_state().clone()
    rate, diagnostics = ln.calibrate_readout_learning_rate(session, 1e-3)
    for key, value in session.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert all(parameter.grad is None for parameter in session.parameters())
    assert rate > 0 and np.isfinite(rate)
    assert diagnostics['recurrent_change_rms'] > 0
    assert diagnostics['readout_change_rms'] > 0
    repeated_rate, _ = ln.calibrate_readout_learning_rate(session, 1e-3)
    assert repeated_rate == rate

    optimizer = torch.optim.Adam([
        {'params': session.recurrent_parameters(), 'lr': 1e-3},
        {'params': [session.Vout], 'lr': rate},
    ])
    predictions = session.readout()
    (predictions - session.readout_targets).square().sum().backward()
    effects = ln._readout_optimizer_step(session, optimizer, predictions, measure_effects=True)
    assert effects['readout_to_recurrent_ratio'] == pytest.approx(1, rel=0.005)
    assert all(state['step'].item() == 1 for state in optimizer.state.values())


def test_retraining_reduces_loss_and_roundtrips_all_readout_state(source_network, tmp_path):
    root, source, original = source_network
    results = ln.retrain_latent_network_readout(source, n_steps=30, training_path=root, balance_check_interval=7)
    metadata = results['readout_retraining']
    assert metadata['delay_window'] == (12, 16)
    assert set(metadata['learning_rate_readout_by_session']) == {3, 11}
    assert set(metadata['calibration_by_session']) == {3, 11}
    assert len(results['loss_history']) == 30
    for fx in original.session_ids:
        assert len(results['loss_history_by_session'][fx]) == 30
        assert len(metadata['readout_history_by_session'][fx]) == 30
        assert [item['step'] for item in metadata['balance_history_by_session'][fx]] == [0, 7, 14, 21, 28, 29]
        assert metadata['balance_history_by_session'][fx][0]['readout_to_recurrent_ratio'] == pytest.approx(1, rel=0.01)
        session = results['rnn'].session_model(fx)
        final_loss = (session.readout() - session.readout_targets).square().sum().item()
        assert final_loss < results['loss_history_by_session'][fx][0]
        assert all(state['step'].item() == 30
                   for state in metadata['optimizer_states_by_session'][fx]['state'].values())
    np.testing.assert_allclose(
        results['loss_history'],
        np.mean(list(results['loss_history_by_session'].values()), axis=0),
    )
    checkpoint = torch.load(results['checkpoint_path'], weights_only=True)
    assert checkpoint['version'] == ln.READOUT_CHECKPOINT_VERSION
    restored = ln.load_latent_network(results['checkpoint_path'], training_path=root)
    assert restored['retraining_objective'] == 'readout_reversal'
    restored['rnn'].eval()
    for fx in original.session_ids:
        session = restored['rnn'].session_model(fx)
        assert session.readout_delay_window == (12, 16)
        assert not session.input_vectors.requires_grad
        assert not session.initial_state_layer.weight.requires_grad
        with torch.no_grad():
            expected = results['rnn'].readout(fx)
            actual = restored['rnn'].readout(fx)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(restored['rnn'].readout(fx), actual, rtol=0, atol=0)
        torch.testing.assert_close(actual, torch.tensor(metadata['final_readouts_by_session'][fx]))
    for key, value in results['rnn'].state_dict().items():
        torch.testing.assert_close(restored['rnn'].state_dict()[key], value, rtol=0, atol=0)
    ln.save_latent_network(restored, checkpoint_path=tmp_path / 'resaved.pt')
    assert (tmp_path / 'resaved.pt').is_file()


def test_manual_rate_and_diagnostics_do_not_change_training(source_network, tmp_path, monkeypatch):
    root, source, _ = source_network

    def unexpected_calibration(*args, **kwargs):
        pytest.fail('Explicit readout rates must bypass calibration.')

    monkeypatch.setattr(ln, 'calibrate_readout_learning_rate', unexpected_calibration)
    runs = []
    for interval in (0, 1):
        torch.manual_seed(42)
        runs.append(ln.retrain_latent_network_readout(
            source, n_steps=3, training_path=root, learning_rate_recurrent=3e-4,
            learning_rate_readout=2e-4, balance_check_interval=interval,
            save_path=tmp_path / f'output_{interval}',
        ))
    for results in runs:
        assert Path(results['checkpoint_path']).parent.name == results['rnn'].recurrent_type
        metadata = results['readout_retraining']
        assert metadata['learning_rate_recurrent'] == 3e-4
        assert metadata['learning_rate_readout_by_session'] == {3: 2e-4, 11: 2e-4}
        assert metadata['calibration_by_session'] == {}
    assert runs[0]['loss_history'] == runs[1]['loss_history']
    assert runs[0]['readout_retraining']['balance_history_by_session'] == {3: [], 11: []}
    for key, value in runs[0]['rnn'].state_dict().items():
        torch.testing.assert_close(runs[1]['rnn'].state_dict()[key], value, rtol=0, atol=0)


def test_invalid_window_and_source_overwrite_are_rejected(source_network):
    root, source, _ = source_network
    before = source.read_bytes()
    with pytest.raises(ValueError, match='delay_window'):
        ln.retrain_latent_network_readout(source, training_path=root, delay_window=(12, 25))
    with pytest.raises(ValueError, match='must differ'):
        ln.retrain_latent_network_readout(source, training_path=root, checkpoint_path=source)
    assert source.read_bytes() == before


def test_archived_source_saves_readout_in_main_model_folder(source_network):
    root, source, _ = source_network
    source_bytes = source.read_bytes()
    archived_source = source.parent / 'previous' / source.name
    archived_source.parent.mkdir()
    archived_source.write_bytes(source_bytes)
    results = ln.retrain_latent_network_readout(
        archived_source, n_steps=1, training_path=root, learning_rate_readout=1e-4,
    )
    expected = source.with_name('original_retrained_readout.pt')
    assert Path(results['checkpoint_path']) == expected
    assert expected.is_file()
    assert not (archived_source.parent / expected.name).exists()
    assert archived_source.read_bytes() == source_bytes
    assert source.read_bytes() == source_bytes
    restored = ln.load_latent_network(expected, training_path=root)
    ln.save_latent_network(restored)
    assert Path(restored['checkpoint_path']) == expected


def test_zero_coding_direction_and_unmeasurable_calibration_are_reported(source_network):
    root, source, original = source_network
    session = original.session_model(3)
    initialize_session_readout(session, ln.load_training_data(root)['activity'][3])
    with torch.no_grad():
        session.readout_targets.copy_(session.readout())
    before = {key: value.clone() for key, value in session.state_dict().items()}
    with pytest.raises(ValueError, match='no measurable'):
        ln.calibrate_readout_learning_rate(session)
    for key, value in session.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    # Make a separate source whose two responses are identically zero.
    loaded = ln.load_latent_network(source, training_path=root)
    with torch.no_grad():
        for model in loaded['rnn'].sessions.values():
            model.input_vectors.zero_()
            model.initial_state_layer.weight.zero_()
            model.initial_state_layer.bias.zero_()
    zero_source = source.with_name('zero_cd.pt')
    ln.save_latent_network(loaded, checkpoint_path=zero_source)
    with pytest.raises(ValueError, match='CD1 must be finite and nonzero'):
        ln.retrain_latent_network_readout(zero_source, training_path=root)


def test_legacy_zero_initial_state_supports_readout_roundtrip(source_network):
    root, source, _ = source_network
    loaded = ln.load_latent_network(source, training_path=root)
    rnn = loaded['rnn']
    rnn.n_initial_timepoints = None
    for session in rnn.sessions.values():
        session.n_initial_timepoints = None
        session.initial_state_layer = None
        session.register_buffer('h0', torch.zeros(session.n_conditions, session.n_units))
    legacy_source = source.with_name('legacy.pt')
    ln.save_latent_network(loaded, checkpoint_path=legacy_source)
    checkpoint = torch.load(legacy_source, weights_only=True)
    checkpoint['version'] = 3
    checkpoint['config'].pop('n_initial_timepoints')
    torch.save(checkpoint, legacy_source)
    results = ln.retrain_latent_network_readout(legacy_source, n_steps=2, training_path=root)
    restored = ln.load_latent_network(results['checkpoint_path'], training_path=root)
    for fx in rnn.session_ids:
        assert restored['rnn'].session_model(fx).readout_initial_activity is None
        torch.testing.assert_close(restored['rnn'].readout(fx), results['rnn'].readout(fx), rtol=0, atol=0)


def test_command_line_readout_retraining_and_loading(source_network, tmp_path, monkeypatch):
    root, source, _ = source_network
    output = tmp_path / 'cli_readout.pt'
    module = Path(ln.__file__)
    monkeypatch.setattr(sys, 'argv', [
        str(module), '--retrain_readout', '--source_checkpoint', str(source),
        '--checkpoint_path', str(output), '--training_path', str(root), '--n_steps', '2',
        '--learning_rate_recurrent', '0.0003', '--learning_rate_readout', '0.0002',
        '--delay_window', '12', '16', '--balance_check_interval', '1',
    ])
    namespace = runpy.run_path(str(module), run_name='__main__')
    results = namespace['retrained_results']
    assert results['checkpoint_path'] == str(output)
    assert results['readout_retraining']['learning_rate_readout_by_session'] == {3: 0.0002, 11: 0.0002}
    monkeypatch.setattr(sys, 'argv', [
        str(module), '--load_only', '--checkpoint_path', str(output), '--training_path', str(root),
    ])
    namespace = runpy.run_path(str(module), run_name='__main__')
    assert namespace['loaded_results']['retraining_objective'] == 'readout_reversal'


def scaled_source_checkpoint(source_network, session_ids=None, scale=0.01):
    """Scale chosen sessions' responses while retaining their recurrent weights."""
    root, source, _ = source_network
    loaded = ln.load_latent_network(source, training_path=root)
    selected = loaded['rnn'].session_ids if session_ids is None else session_ids
    with torch.no_grad():
        for fx in selected:
            session = loaded['rnn'].session_model(fx)
            session.input_vectors.mul_(scale)
            session.initial_state_layer.weight.mul_(scale)
            session.initial_state_layer.bias.mul_(scale)
    scaled = source.with_name('scaled_responses.pt')
    ln.save_latent_network(loaded, checkpoint_path=scaled)
    return scaled, loaded['rnn']


def test_tolerance_stops_sessions_independently_without_later_updates(source_network, monkeypatch):
    root, _, _ = source_network
    # Give the second session a larger reversal to learn so it finishes later.
    source, original = scaled_source_checkpoint(source_network, session_ids=(11,), scale=2.0)
    real_step = ln._readout_optimizer_step
    fx_by_units = {original.n_units_by_session[fx]: fx for fx in original.session_ids}
    sessions, frozen, step_calls = {}, {}, []

    def snapshot(session):
        return {name: value.detach().clone() for name, value in session.named_parameters()}

    def tracked_step(session, optimizer, predictions, measure_effects=False):
        fx = fx_by_units[session.n_units]
        sessions[fx] = session
        assert len(step_calls) < 2000, 'Tolerance training did not finish.'
        for stopped_fx, saved in frozen.items():
            for name, value in sessions[stopped_fx].named_parameters():
                torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
        threshold = 0.01 * (session.readout_targets[1] - session.readout_targets[0]).abs().item()
        assert (predictions - session.readout_targets).square().sum().item() >= threshold
        step_calls.append(fx)
        effects = real_step(session, optimizer, predictions, measure_effects)
        with torch.no_grad():
            if (session.readout() - session.readout_targets).square().sum().item() < threshold:
                frozen[fx] = snapshot(session)
        return effects

    monkeypatch.setattr(ln, '_readout_optimizer_step', tracked_step)
    torch.manual_seed(41)
    results = ln.retrain_latent_network_readout(
        source, n_steps=1, training_path=root, learning_rate_readout=0.01,
        stop_on_readout_tolerance=True, balance_check_interval=3,
    )
    metadata = results['readout_retraining']
    assert metadata['termination_reason'] == 'all_sessions_converged'
    assert metadata['max_steps'] is None
    assert all(metadata['converged_by_session'].values())
    assert len(set(metadata['updates_by_session'].values())) == 2
    assert min(metadata['updates_by_session'].values()) > 1  # n_steps is ignored in this mode.
    for fx in original.session_ids:
        count = metadata['updates_by_session'][fx]
        assert count == step_calls.count(fx) == metadata['stop_step_by_session'][fx]
        assert len(results['loss_history_by_session'][fx]) == count + 1
        assert len(metadata['readout_history_by_session'][fx]) == count + 1
        assert metadata['final_loss_by_session'][fx] < metadata['threshold_by_session'][fx]
        assert all(state['step'].item() == count
                   for state in metadata['optimizer_states_by_session'][fx]['state'].values())
        for name, value in results['rnn'].session_model(fx).named_parameters():
            torch.testing.assert_close(value, frozen[fx][name], rtol=0, atol=0)
    expected_aggregate = [
        sum(history[min(index, len(history) - 1)] / original.n_sessions
            for history in results['loss_history_by_session'].values())
        for index in range(len(results['loss_history']))
    ]
    np.testing.assert_allclose(results['loss_history'], expected_aggregate)
    restored = ln.load_latent_network(results['checkpoint_path'], training_path=root)
    for key in ('threshold_by_session', 'converged_by_session', 'stop_step_by_session',
                'updates_by_session', 'final_loss_by_session', 'termination_reason'):
        assert restored['readout_retraining'][key] == metadata[key]


def test_initially_converged_sessions_skip_calibration_and_optimizers(source_network, monkeypatch):
    root, _, _ = source_network
    source, original = scaled_source_checkpoint(source_network)

    def unexpected_optimizer(*args, **kwargs):
        pytest.fail('Initially converged sessions must not create optimizers or calibrate rates.')

    monkeypatch.setattr(torch.optim, 'Adam', unexpected_optimizer)
    results = ln.retrain_latent_network_readout(
        source, training_path=root, stop_on_readout_tolerance=True,
    )
    metadata = results['readout_retraining']
    assert metadata['termination_reason'] == 'all_sessions_converged'
    assert metadata['updates_by_session'] == {3: 0, 11: 0}
    assert metadata['stop_step_by_session'] == {3: 0, 11: 0}
    assert metadata['learning_rate_readout_by_session'] == {3: None, 11: None}
    assert metadata['optimizer_states_by_session'] == {3: None, 11: None}
    assert metadata['calibration_by_session'] == {}
    assert len(results['loss_history']) == 1
    for fx in original.session_ids:
        session = results['rnn'].session_model(fx)
        assert len(results['loss_history_by_session'][fx]) == 1
        assert metadata['final_loss_by_session'][fx] < metadata['threshold_by_session'][fx]
        torch.testing.assert_close(session.Vout, session.CD1, rtol=0, atol=0)
        for name, value in original.session_model(fx).named_parameters():
            torch.testing.assert_close(dict(session.named_parameters())[name], value, rtol=0, atol=0)
    restored = ln.load_latent_network(results['checkpoint_path'], training_path=root)
    assert restored['readout_retraining']['optimizer_states_by_session'] == {3: None, 11: None}


def test_initially_converged_session_does_not_block_calibration_of_other_sessions(source_network, monkeypatch):
    root, _, _ = source_network
    source, _ = scaled_source_checkpoint(source_network, session_ids=(3,))
    real_calibration = ln.calibrate_readout_learning_rate
    calibrated_units = []

    def track_calibration(session, learning_rate_recurrent):
        calibrated_units.append(session.n_units)
        return real_calibration(session, learning_rate_recurrent)

    monkeypatch.setattr(ln, 'calibrate_readout_learning_rate', track_calibration)
    torch.manual_seed(41)
    results = ln.retrain_latent_network_readout(
        source, training_path=root, stop_on_readout_tolerance=True, max_steps=2,
    )
    metadata = results['readout_retraining']
    assert calibrated_units == [7]
    assert metadata['updates_by_session'] == {3: 0, 11: 2}
    assert metadata['learning_rate_readout_by_session'][3] is None
    assert metadata['learning_rate_readout_by_session'][11] > 0
    assert set(metadata['calibration_by_session']) == {11}
    assert len(results['loss_history_by_session'][3]) == 1
    assert len(results['loss_history_by_session'][11]) == 3


def test_fixed_duration_still_updates_sessions_already_within_tolerance(source_network):
    root, _, _ = source_network
    source, original = scaled_source_checkpoint(source_network)
    results = ln.retrain_latent_network_readout(
        source, training_path=root, n_steps=4, learning_rate_readout=1e-5,
    )
    metadata = results['readout_retraining']
    assert metadata['stop_on_readout_tolerance'] is False
    assert metadata['termination_reason'] == 'n_steps'
    assert metadata['updates_by_session'] == {3: 4, 11: 4}
    assert metadata['stop_step_by_session'] == {3: None, 11: None}
    assert len(results['loss_history']) == 4
    for fx in original.session_ids:
        assert results['loss_history_by_session'][fx][0] < metadata['threshold_by_session'][fx]
        assert len(results['loss_history_by_session'][fx]) == 4
        assert all(state['step'].item() == 4
                   for state in metadata['optimizer_states_by_session'][fx]['state'].values())


def test_max_steps_caps_updates_and_reports_unconverged_sessions(source_network):
    root, source, _ = source_network
    torch.manual_seed(41)
    results = ln.retrain_latent_network_readout(
        source, n_steps=1, training_path=root, learning_rate_recurrent=1e-8,
        learning_rate_readout=1e-8, stop_on_readout_tolerance=True, max_steps=2,
    )
    metadata = results['readout_retraining']
    assert metadata['termination_reason'] == 'max_steps'
    assert metadata['updates_by_session'] == {3: 2, 11: 2}
    assert metadata['converged_by_session'] == {3: False, 11: False}
    assert metadata['stop_step_by_session'] == {3: None, 11: None}
    assert len(results['loss_history']) == 3  # Two pre-update points plus the final check.
    for fx in (3, 11):
        assert len(results['loss_history_by_session'][fx]) == 3
        assert results['loss_history_by_session'][fx][-1] == metadata['final_loss_by_session'][fx]
        assert metadata['final_loss_by_session'][fx] >= metadata['threshold_by_session'][fx]


def test_convergence_on_last_capped_update_is_recorded(source_network, monkeypatch):
    root, source, _ = source_network
    real_step = ln._readout_optimizer_step

    def solve_readout_on_update(session, optimizer, predictions, measure_effects=False):
        effects = real_step(session, optimizer, predictions, measure_effects)
        # Solve the two readout constraints exactly on this final permitted step.
        with torch.no_grad():
            activity = session(initial_activity=session.readout_initial_activity,
                               input_timecourses=session.readout_input_timecourses)
            matrix = torch.stack([activity[condition][12:16].mean(0) for condition in ('P1', 'A1')])
            solution = torch.linalg.lstsq(matrix.double(), session.readout_targets.double()).solution
            session.Vout.copy_(solution.to(session.Vout.dtype))
        return effects

    monkeypatch.setattr(ln, '_readout_optimizer_step', solve_readout_on_update)
    torch.manual_seed(41)
    results = ln.retrain_latent_network_readout(
        source, training_path=root, learning_rate_readout=1e-4,
        stop_on_readout_tolerance=True, max_steps=1, balance_check_interval=0,
    )
    metadata = results['readout_retraining']
    assert metadata['termination_reason'] == 'all_sessions_converged'
    assert metadata['converged_by_session'] == {3: True, 11: True}
    assert metadata['stop_step_by_session'] == {3: 1, 11: 1}
    assert metadata['updates_by_session'] == {3: 1, 11: 1}
    assert len(results['loss_history']) == 2


def test_zero_target_gap_is_rejected_in_tolerance_mode(source_network, monkeypatch):
    root, source, _ = source_network
    real_initialize = ln.SessionRNN.initialize_readout

    def initialize_equal_targets(session, *args, **kwargs):
        real_initialize(session, *args, **kwargs)
        session.readout_targets.fill_(1)

    monkeypatch.setattr(ln.SessionRNN, 'initialize_readout', initialize_equal_targets)
    with pytest.raises(ValueError, match='positive target gap'):
        ln.retrain_latent_network_readout(source, training_path=root, stop_on_readout_tolerance=True)


def test_command_line_tolerance_mode(source_network, tmp_path, monkeypatch):
    root, source, _ = source_network
    output = tmp_path / 'cli_tolerance.pt'
    monkeypatch.setattr(sys, 'argv', [
        ln.__file__, '--retrain_readout', '--source_checkpoint', str(source),
        '--checkpoint_path', str(output), '--training_path', str(root),
        '--stop_on_readout_tolerance', '--max_steps', '2', '--n_steps', '1',
        '--learning_rate_recurrent', '1e-8', '--learning_rate_readout', '1e-8',
    ])
    namespace = runpy.run_path(ln.__file__, run_name='__main__')
    metadata = namespace['retrained_results']['readout_retraining']
    assert metadata['stop_on_readout_tolerance'] is True
    assert metadata['max_steps'] == 2
    assert metadata['updates_by_session'] == {3: 2, 11: 2}
    assert metadata['termination_reason'] == 'max_steps'


@pytest.mark.parametrize('arguments', [
    ['--stop_on_readout_tolerance'],
    ['--retrain_readout', '--max_steps', '2'],
])
def test_command_line_rejects_incompatible_tolerance_flags(arguments, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [ln.__file__, *arguments])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(ln.__file__, run_name='__main__')
    assert error.value.code == 2
