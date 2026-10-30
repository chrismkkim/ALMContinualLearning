"""Retrain saved latent networks to reverse their readouts.

Examples:
    python3 run_retrained_network.py
    python3 run_retrained_network.py --dry_run
    python3 run_retrained_network.py -- --device cpu

Edit RUN_CONFIGURATIONS to change the arguments for individual runs.
Additional options after -- are shared by every run.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


RUN_CONFIGURATIONS = [
    [
        '--retrain_readout',
        '--source_checkpoint', 'data/training/recurrent/lowrank/latent_network_Dataset0_In1_Rank10_lowrank.pt',
        '--n_steps', '600',
        '--learning_rate_alpha', '1e-4',
        '--learning_rate_u', '1e-3',
        '--learning_rate_singular_values', '1e-1',
        '--learning_rate_right_vectors', '1e-1',
        '--train_initial_state_layer',
        '--train_sensory_input',
        '--learning_rate_initial_state_layer', '1e-4',
        '--learning_rate_sensory_input', '1e-3',
        '--lambda_readout', '1',
        '--lambda_coding', '1',
    ],
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry_run', action='store_true', help='Print commands without running them.')
    parser.add_argument('training_args', nargs=argparse.REMAINDER, help='Shared training options after --.')
    args = parser.parse_args()
    training_args = args.training_args
    if training_args[:1] == ['--']:
        training_args = training_args[1:]

    script_dir = Path(__file__).resolve().parent
    training_script = script_dir / '_8_LatentNetwork.py'
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'

    for index, run_args in enumerate(RUN_CONFIGURATIONS, start=1):
        # Per-run arguments take precedence over shared arguments.
        command = [sys.executable, str(training_script), *training_args, *run_args]
        print(f'[{index}/{len(RUN_CONFIGURATIONS)}] {shlex.join(command)}', flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=script_dir, env=env, check=False)
        if result.returncode:
            print(f'Run {index} failed; stopping the batch.', file=sys.stderr, flush=True)
            return result.returncode

    if not args.dry_run:
        print(f'Completed {len(RUN_CONFIGURATIONS)} runs.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
