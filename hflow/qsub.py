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
# python hflow/qsub.py -n sweep100 -t 10


COMMANDS = [

    # f'-cn=trap optimizer.iters=25_000 loss.sigma=1e-3',
    # f'-cn=trap optimizer.iters=25_000 loss.sigma=1e-2',
    # f'-cn=trap optimizer.iters=25_000 loss.sigma=1e-1',
    # f'-cn=trap optimizer.iters=25_000 loss.sigma=0.0',

    # f'-cn=trap optimizer.iters=50_000 loss.sigma=1e-3',
    # f'-cn=trap optimizer.iters=50_000 loss.sigma=1e-2',
    # f'-cn=trap optimizer.iters=50_000 loss.sigma=1e-1',
    # f'-cn=trap optimizer.iters=50_000 loss.sigma=0.0',
    # f'-cn=osc problem=bi optimizer.iters=25_000 loss.sigma=1e-2 test.save_sol=True',
    # f'-cn=osc problem=bi optimizer.iters=25_000 loss.sigma=1e-1 test.save_sol=True',
    # f'-cn=osc problem=bi optimizer.iters=25_000 loss.sigma=0.0 test.save_sol=True',
    #     f'-cn=lz9 problem=lz9 optimizer.iters=100_000 loss.sigma=5e-2 test.save_sol=True',
    #     f'-cn=lz9 problem=lz9 optimizer.iters=100_000 loss.sigma=6e-2 test.save_sol=True',
    #     f'-cn=lz9 problem=lz9 optimizer.iters=100_000 loss.sigma=7e-2 test.save_sol=True',
    #     f'-cn=lz9 problem=lz9 optimizer.iters=100_000 loss.sigma=8e-2 test.save_sol=True',
    #     f'-cn=lz9 problem=lz9 optimizer.iters=100_000 loss.sigma=9e-2 test.save_sol=True',
    #     f'-cn=lz9 problem=lz9 optimizer.iters=100_000 loss.loss_fn=ncsm test.save_sol=True',
    #     f'-cn=lz9 problem=lz9 optimizer.iters=100_000 loss.loss_fn=cfm  test.save_sol=True',
    #     f'-cn=lz9 problem=lz9 optimizer.iters=100_000 loss.loss_fn=ncsm loss.T=50 test.save_sol=True',
    #     f'-cn=lz9 problem=lz9 optimizer.iters=100_000 loss.loss_fn=cfm  loss.T=50 test.save_sol=True',
    #     f'-cn=lz9 problem=lz9 optimizer.iters=100_000 loss.loss_fn=ncsm loss.T=25 test.save_sol=True',
    #     f'-cn=lz9 problem=lz9 optimizer.iters=100_000 loss.loss_fn=cfm  loss.T=25 test.save_sol=True',

    f'-cn=lz9 problem=lz92     optimizer.iters=100_000 loss.sigma=7e-2 test.save_sol=True  sample.scheme_t=rand  sample.scheme_n=rand  sample.bs_n=1',
    f'-cn=vlasov problem=vtwo  optimizer.iters=50_000  loss.sigma=5e-2 test.save_sol=True  sample.scheme_t=rand  sample.scheme_n=rand  sample.bs_n=1',
    f'-cn=vlasov problem=vbump optimizer.iters=50_000  loss.sigma=5e-3 test.save_sol=True  sample.scheme_t=rand  sample.scheme_n=rand  sample.bs_n=1',
    f'-cn=lz9 problem=lz92     optimizer.iters=100_000 loss.sigma=7e-2 test.save_sol=True  sample.scheme_t=rand  sample.scheme_n=rand  sample.bs_n=256',
    f'-cn=vlasov problem=vtwo  optimizer.iters=50_000  loss.sigma=5e-2 test.save_sol=True  sample.scheme_t=rand  sample.scheme_n=rand  sample.bs_n=256',
    f'-cn=vlasov problem=vbump optimizer.iters=50_000  loss.sigma=5e-3 test.save_sol=True  sample.scheme_t=rand  sample.scheme_n=rand  sample.bs_n=256',


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
