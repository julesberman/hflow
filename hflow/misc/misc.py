
import os
import random
import string
from functools import wraps
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.host_callback import id_print, id_tap
from scipy.special import eval_legendre, roots_legendre
from tqdm.auto import tqdm


def jqdm(total, argnum=0, decimals=1, **kwargs):
    "Decorate a jax scan body function to include a TQDM progressbar."

    pbar = tqdm(range(100), mininterval=500, **kwargs)

    def _update(cur, transforms):
        amt = float(cur*100/total)
        amt = round(amt, decimals)
        if amt != pbar.last_print_n:
            pbar.n = amt
            pbar.last_print_n = amt
            pbar.refresh()

    def update_jqdm(cur):
        id_tap(_update, cur),

    def _jqdm(func):

        @wraps(func)
        def wrapper_body_fun(*args, **kwargs):
            cur = args[argnum]
            update_jqdm(cur)
            result = func(*args, **kwargs)
            return result  # close_tqdm(result, amt)

        return wrapper_body_fun

    return _jqdm


def unique_id(n) -> str:
    """creates unique alphanumeric id w/ low collision probability"""
    chars = string.ascii_letters + string.digits  # 64 choices
    id_str = "".join(random.choice(chars) for _ in range(n))
    return id_str


def epoch_time(decimals=0) -> int:
    return int(time()*(10**(decimals)))


def count_params(tree):
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(tree))
    return param_count


def pts_array_from_space(space):
    m_grids = jnp.meshgrid(*space,  indexing='ij')
    x_pts = jnp.asarray([m.flatten() for m in m_grids]).T
    return x_pts
