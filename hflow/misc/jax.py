import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev, jvp, vmap


def get_rand_idx(key, N, bs):
    idx = jnp.arange(0, N)
    return jax.random.choice(key, idx, shape=(bs,), replace=False)


def hess_trace_estimator(fn, argnum=0, diff='rev'):

    if diff == 'fwd':
        d_fn = jacfwd(fn, argnums=argnum)
    else:
        d_fn = jacrev(fn, argnums=argnum)

    def estimator(key, *args, **kwargs):
        args = list(args)
        primal = args[argnum]
        eps = jax.random.normal(key, shape=primal.shape)

        def s_dx_wrap(x):
            return d_fn(*args[:argnum], x, *args[argnum+1:], **kwargs)
        dx_val, jvp_val = jvp(s_dx_wrap, (primal,), (eps,))
        trace = jnp.dot(eps, jvp_val)
        return dx_val, trace

    return estimator


def meanvmap(f, mean_axes=(0,), in_axes=(0,)):
    return lambda *fargs, **fkwargs: jnp.mean(vmap(f, in_axes=in_axes)(*fargs, **fkwargs), axis=mean_axes)
