"""Removed reversal entry points fail without starting training or loading data."""

from pathlib import Path
import runpy
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _8_LatentNetwork as ln


@pytest.mark.parametrize('arguments', [
    ['--retrain_readout'],
    ['--source_checkpoint', 'unused.pt'],
    ['--learning_rate_recurrent', '1e-3'],
    ['--learning_rate_readout', '1e-4'],
    ['--delay_window', '12', '16'],
    ['--balance_check_interval', '50'],
    ['--stop_on_readout_tolerance'],
    ['--max_steps', '100'],
])
def test_reversal_cli_is_rejected(arguments, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', [ln.__file__, *arguments])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(ln.__file__, run_name='__main__')
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert 'unrecognized arguments' in message
    assert arguments[0] in message


def test_reversal_checkpoint_is_rejected_before_loading_training_data(tmp_path):
    checkpoint = tmp_path / 'reversal.pt'
    torch.save({'version': 5}, checkpoint)
    with pytest.raises(ValueError, match='Unsupported checkpoint format'):
        ln.load_latent_network(checkpoint, training_path=tmp_path / 'missing_data')
