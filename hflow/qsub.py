import argparse
import time
from collections.abc import Iterable
from itertools import product
from pathlib import Path

import subprocess

from nrom.configs.classes import Config, Net_Config, default
from nrom.io.utils import flatten_dataclass, set_dataclass_attr, save_pickle
from nrom.misc.misc import unique_id

from tqdm import tqdm

PROBLEM = 'vtwo'
COMMANDS = [

    # f'-cn=vlasov problem={PROBLEM} optimizer.iters=100_000 loss.sigma=1e-2',
    # f'-cn=vlasov problem={PROBLEM} optimizer.iters=250_000 loss.sigma=1e-2',
    f'-cn=vlasov problem={PROBLEM} optimizer.iters=25_000 loss.sigma=1e-3',
    f'-cn=vlasov problem={PROBLEM} optimizer.iters=25_000 loss.sigma=5e-3',
    f'-cn=vlasov problem={PROBLEM} optimizer.iters=25_000 loss.sigma=1e-2',
    f'-cn=vlasov problem={PROBLEM} optimizer.iters=25_000 loss.sigma=5e-2',
    f'-cn=vlasov problem={PROBLEM} optimizer.iters=25_000 loss.sigma=1e-1',
]


def submit(submit_args):
    # setup launch
    name = submit_args.name
    exp_dir = Path(f'./qsubs/{name}')
    exp_dir.mkdir(exist_ok=True, parents=True)

    print(f"you are launching {len(COMMANDS)} jobs 😅")

    # save config files
    config_dir = exp_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    for i, command in tqdm(enumerate(COMMANDS)):
        job_command = f'/p/home/jmb/.conda/envs/ng/bin/python hflow/run.py {command}'
        script_path = generate_pbs_script(
            job_command, exp_dir, job_name=name, hours=args.time, itr=i)
        subprocess.run(['qsub', script_path])

    print("submit done!")


def generate_pbs_script(command, save_dir, job_name='my_job', hours=1, itr=0):
    # Define the PBS script content
    pbs_script = f"""#!/bin/bash
#PBS -N {job_name}
#PBS -l select=1:ncpus=22:mpiprocs=1:ngpus=1
#PBS -l walltime={hours}:00:00
#PBS -A AFPRD43472001
#PBS -q standard
#PBS -l application=model-reduction

# Change to the directory where the script is submitted
cd $PBS_O_WORKDIR

# Your Python job commands
{command}
"""

    # Write the script to a file
    script_path = save_dir / f'submit_{job_name}_{itr}.pbs'
    with open(script_path, 'w') as script_file:
        script_file.write(pbs_script)

    return script_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name", "-n", help="exp_name", type=str, required=True)
    parser.add_argument(
        "--time", "-t", help="timeout_min in hours", type=int, default=8)

    args = parser.parse_args()
    results = submit(args)
