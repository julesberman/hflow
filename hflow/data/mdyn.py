import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import jit, vmap
from matplotlib import pyplot as plt


def get_V_cell(D):
    pi = jnp.pi

    x0 = jnp.concatenate(
        (0.95*jnp.array([jnp.cos(pi/6),   jnp.sin(pi/6)]), jnp.zeros(D-2)), axis=0)
    x1 = jnp.concatenate(
        (1.05*jnp.array([jnp.cos(5*pi/6), jnp.sin(5*pi/6)]), jnp.zeros(D-2)), axis=0)
    x2 = jnp.concatenate(
        (1.00*jnp.array([jnp.cos(-pi/2),  jnp.sin(-pi/2)]), jnp.zeros(D-2)), axis=0)

    @jit
    def V(x):
        return 4 * jnp.sum((x - x0)**2) * jnp.sum((x - x1)**2) * jnp.sum((x - x2)**2)

    return V


def get_V_random(key, D, K=16, M=4, smoothness=1):
    pi = jnp.pi
    ks = []
    M = M*D
    keys = random.split(key, M)
    for i in range(M):
        k = random.normal(keys[i], (D,))
        k = k / jnp.linalg.norm(k)  # M random unit vectors
        k = jnp.outer(k, jnp.arange(start=1, stop=K+1))
        ks.append(k)
    k = jnp.hstack(ks)
    # array now holds random wave vectors, M per dimension and K modes each

    def random_f(noise, k, x):
        return (noise[0] * jnp.cos(2*pi*jnp.inner(k, x)) + noise[1] * jnp.sin(2*pi*jnp.inner(k, x))) * 1/jnp.sum(k**(2))**(smoothness)
    # random function with wave vector k

    def f_sum(key, x):
        noise = random.normal(key, (K*M, 2))
        return sum(vmap(random_f, in_axes=(0, 1, None))(noise, k, x))
    # sum of random functions with random wave vectors

    @jit
    def V(x):
        return 0.1 * f_sum(key, x) + 0.5*jnp.sum(x**2)

    return V


def plot_V(V, D, type):
    _x = jnp.linspace(-1.5, 1.5, 100)

    def partialV(x, y):
        return V(jnp.concatenate((jnp.array([x, y]), jnp.zeros(D-2)), axis=0))

    def V_x(y):
        return vmap(lambda x: partialV(x, y))(_x)
    Z = vmap(V_x)(_x)
    if type == "cell":
        im = plt.contourf(_x, _x, Z, np.linspace(0, 10, 21),
                          extend="max", cmap=plt.cm.plasma_r)
    else:
        im = plt.contourf(_x, _x, Z, 100, extend="both", cmap=plt.cm.plasma_r)
    plt.colorbar(im)
    return im


def get_mdyn_sol(key, dim, N, gamma=0.05, alpha=0, sigma=0.0, dt=5e-3, tau=None):
    # gamma is the width of the interaction kernel
    # alpha is the strength of the interaction
    # tau is the friction coefficient (tau = dt, alpha = 0 gives gradient flow)
    if tau is None:
        tau = dt
    SD_0 = 0.2

    V = get_V_cell(dim)
    particles = SD_0 * random.normal(key, (N, dim))

    def interaction_potential(x, y):
        r = jnp.sum((x - y)**2)
        return jnp.exp(-r/2/gamma**2)

    def bulk_interaction_potential(x, particles):
        return jnp.sum(vmap(lambda y: interaction_potential(x, y))(particles)) / particles.shape[0]

    t_eval = np.linspace(dt, 1, int(1/dt+1))

    @jit
    def time_step(particles, velocities, dt, key):
        # dx = -grad V(x)dt + v dt + sigma dB
        # dv = grad g(rho)(x)dt
        gradV = jax.grad(V)
        gradG = jax.grad(lambda x: bulk_interaction_potential(x, particles))
        if sigma != 0:
            dB = sigma * dt**0.5 * random.normal(key, particles.shape)
        else:
            dB = 0
        velocities -= velocities * dt / tau
        velocities -= jax.vmap(gradV)(particles) * dt / tau
        if alpha != 0:
            velocities -= alpha * jax.vmap(gradG)(particles) * dt
        particles += dt * velocities + dB
        return particles, velocities

    velocities = jnp.zeros_like(particles)
    sol = [particles]
    keys = random.split(key, len(t_eval))
    for i in range(len(t_eval[1:])):
        particles, velocities = time_step(particles, velocities, dt, keys[i])
        sol.append(particles)
    sol = jnp.stack(sol, axis=0)

    return sol
