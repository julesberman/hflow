## Setup
This code was all developed with:
`
python3.11
`


First locally install the hflow package with

```bash
pip install --editable .
```
Note: if this fails you might need to move notebooks and/or other dirs out of the main directory, run the command, then move them back

Install all additional required packages run:

```bash
 pip install -r requirements.txt
```

Lastly ensure that jax is installed with the appropriate CPU or GPU support depending on where you plan to run this code. Info on installing with GPU suport can be found: [here](https://github.com/google/jax#installation)

NOTE make sure that if you install jax with GPU and cuda support that you still fix the versions of jax and jaxlib at the following:
```
jax==0.4.23
jaxlib==0.4.23
```

## Running single job

Run a problem with problem specific default settings (recommended)
```
python hflow/run.py -cn=osc problem=bi
python hflow/run.py -cn=vlasov problem=vtwo
python hflow/run.py -cn=vlasov problem=vbump
```

Run a problem with problem specific default settings and overrides
```
python hflow/run.py -cn=vlasov problem=vbump data.t_end=10 optimizer.iters=100_000 unet.width=128 loss.sigma=1e-2 data.bs_t=256
```

## Running a sweep
To launch a sweep, first configure the `SWEEP` dict and the `SLURM_CONFIG` dict in `hflow/config.py`
```
SWEEP = {
    'optimizer.iters': '5000,25_000,100_000',
    'unet.width': '32,64,128',
    'sample.bs_t': '8,16,32,64,128',
    'sample.scheme_t': 'rand,gauss',
    'unet.last_activation': 'tanh,none'
}
```
```
SLURM_CONFIG = {
    'timeout_min': 60*4,
    'cpus_per_task': 4,
    'mem_gb': 25,
    'gres': 'gpu'
}
```
The run the command using `--multirun` flag:
```
python hflow/run.py --multirun -cn=vlasov problem=vbump
```