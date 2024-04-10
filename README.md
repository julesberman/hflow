## Setup
This code was all developed with:
`
python3.11
`


First locally install the hflow package with

```bash
pip install --editable .
```

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

## Running

Run a problem will all default settings
```
python hflow/run.py problem=osc
```

Run a problem with problem specific default settings (recommended)
```
python hflow/run.py -cn=vlasov
```

Run a problem with problem specific default settings and overrides
```
python hflow/run.py -cn=vlasov data.t_end=10 optimizer.iters=100_000    
```

To launch a sweep use `--multirun`
```
python hflow/run.py --multirun -cn=vlasov 
```