import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev, jvp, vmap


def get_rand_idx(key, N, bs):
    if bs > N:
        bs = N
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


def tracewrap(f, axis1=0, axis2=1):
    return lambda *fargs, **fkwargs: jnp.trace(f(*fargs, **fkwargs), axis1=axis1, axis2=axis2)


def batchmap(f, n_batches, argnum=0):

    def wrap(*fargs, **fkwarg):
        fargs = list(fargs)
        X = fargs[argnum]
        batches = jnp.split(X, n_batches, axis=0)

        result = []
        for B in batches:
            fargs[argnum] = B
            a = f(*fargs, **fkwarg)
            result.append(a)

        return jnp.concatenate(result)

    return wrap
