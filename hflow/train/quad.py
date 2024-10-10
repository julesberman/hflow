from scipy.special import eval_legendre, roots_legendre

import jax
import jax.numpy as jnp
import numpy as np

def get_simpsons(N):
    if (N - 1) % 2 != 0:
            raise ValueError("Number of intervals N - 1 must be even for Simpson's rule (N must be odd).")

    # Generate N equispaced points on [0,1]
    points = np.linspace(0, 1, N)
    h = 1 / (N - 1)  # Step size

    # Initialize weights
    weights = np.zeros(N)
    weights[0] = h / 3
    weights[-1] = h / 3

    # Apply Simpson's coefficients
    for i in range(1, N - 1):
        if i % 2 == 0:
            weights[i] = 2 * h / 3
        else:
            weights[i] = 4 * h / 3

    points = jnp.asarray(points)
    weights = jnp.asarray(weights)

    return points, weights


def get_simpsons_38(N):

    if (N - 1) % 3 != 0:
        raise ValueError("Number of intervals N - 1 must be a multiple of 3 for Simpson's 3/8 rule.")

    points = np.linspace(0, 1, N)
    h = 1 / (N - 1)  # Step size

    # Initialize weights
    weights = np.zeros(N)
    num_groups = (N - 1) // 3

    for group in range(num_groups):
        idx_start = group * 3
        idx_end = idx_start + 3  # Include idx_end in the group

        # Assign weights according to Simpson's 3/8 rule
        weights[idx_start] += 3 * h / 8
        weights[idx_start + 1] += 9 * h / 8
        weights[idx_start + 2] += 9 * h / 8
        weights[idx_start + 3] += 3 * h / 8

    points = jnp.asarray(points)
    weights = jnp.asarray(weights)

    return points, weights



def get_gauss(n, a=0, b=1):
    points, weights = roots_legendre(n)
    points = 0.5 * (points + 1) * (b - a) + a
    weights = weights * 0.5 * (b - a)
    return points, weights

def dirichlet_indices(key, n_samples, N_index, alpha):
    alphas = jnp.ones(n_samples + 1) * alpha
    spacings = jax.random.dirichlet(key, alphas)
    cumulative = jnp.cumsum(spacings)
    points = cumulative[:-1]
    indices = jnp.asarray(jnp.round(points*N_index,0), dtype=jnp.int32)
    return indices

def arcsin_indices(key, n_samples, N_index):
    alpha, beta = 0.5, 0.5
    points = jax.random.beta(key, alpha, beta, shape=(n_samples,))
    indices = jnp.asarray(jnp.round(points*N_index,0), dtype=jnp.int32)
    return indices

