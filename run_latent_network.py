"""Run latent-network configurations sequentially.

Examples:
    python run_latent_network.py
    python run_latent_network.py --dry_run
    python run_latent_network.py -- --n_steps 500 --dataset_id 0

Edit RUN_CONFIGURATIONS to change the arguments for individual runs.
Training options after -- are shared by every run. Plots are saved using
the Agg backend so training can proceed without waiting for plot windows.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


# RUN_CONFIGURATIONS = [
#     ['--recurrent_type', 'fullconn', '--n_steps', '50', '--dataset_id', '0'],
#     ['--recurrent_type', 'lowrank', '--recurrent_rank', '1', '--n_steps', '50', '--n_input_components', '1', '--dataset_id', '0'],
#     ['--recurrent_type', 'lowrank', '--recurrent_rank', '2', '--n_steps', '50', '--n_input_components', '1', '--dataset_id', '0'],
#     ['--recurrent_type', 'lowrank', '--recurrent_rank', '3', '--n_steps', '50', '--n_input_components', '1', '--dataset_id', '0'],
# ]

# RUN_CONFIGURATIONS = [
#     ['--recurrent_type', 'fullconn', '--n_steps', '5000', '--n_input_components', '1', '--dataset_id', '0'],
#     ['--recurrent_type', 'lowrank',  '--n_steps', '5000', '--n_input_components', '1', '--dataset_id', '0', '--recurrent_rank', '10'],
# ]

RUN_CONFIGURATIONS = [
    ['--recurrent_type', 'lowrank',  '--n_steps', '5000', '--n_input_components', '1', '--dataset_id', '0', '--recurrent_rank', '10'],
]

# RUN_CONFIGURATIONS = [
#     ['--recurrent_type', 'lowrank',  '--n_steps', '5000', '--n_input_components', '1', '--dataset_id', '0', '--recurrent_rank', '5'],
# ]

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
