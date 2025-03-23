import jax.numpy as jnp
import jax

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
import jax
import jax.numpy as jnp
from einops import rearrange
def rk4(x, f, t, dt, eps, key):
    # signature of f is f(x, t)
    k1 = f(x, t)
    k2 = f(x + dt/2 * k1, t + dt/2)
    k3 = f(x + dt/2 * k2, t + dt/2)
    k4 = f(x + dt * k3, t + dt)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def euler(x, f, t, dt, eps, key):
    return x + dt * f(x, t)

def euler_marujama(x, f, t, dt, eps, key):
    return x + dt * f(x, t) + eps * jnp.sqrt(dt) * jax.random.normal(key, x.shape)

def generate_sample_s(x0, v, times, L, key, eps): 
    def step(carry, t_next):
        x, t_prev, key = carry
        dt = t_next - t_prev
        # x_new = rk4(x, v, t_prev, dt, eps, key)
        x_new = euler_marujama(x, v, t_prev, dt, eps, key)
        if L != 0:
            x_new = jnp.mod(x_new, L)
        new_carry = (x_new, t_next, jax.random.split(key)[0])
        return new_carry, x_new

    init = (x0, times[0], key)
    carry, xs = jax.lax.scan(step, init, times[1:])
    xs = jnp.vstack([x0[None, ...], xs])
    return xs


### all data tensors are of shape (n_samples, n_t, n_mu, d)

### Toy data

def constant_normal(n_samples, n_t, key, d, var):
    def get_x_t(t, key):
        return jax.random.normal(key, (n_samples, 1, d)) * jnp.sqrt(var)
    return jax.vmap(get_x_t, out_axes=1)(jnp.linspace(0, 1, n_t), jax.random.split(key, n_t))

def sine_normal(n_samples, n_t, key, d, var):
    def m(t):
        return jnp.ones(d) * jnp.sin(2 * jnp.pi * t) * 1
    def get_x_t(t, key):
        return jax.random.normal(key, (n_samples, 1, d)) * jnp.sqrt(var) + m(t)
    return jax.vmap(get_x_t, out_axes=1)(jnp.linspace(0, 1, n_t), jax.random.split(key, n_t))

def styblinski_tang(x):
    return 0.2 * 0.5 * jnp.sum(x**4 - 16 * x**2 + 5 * x, axis=-1)

def oakley_ohagan(x):
    return 0.2 * 5 * jnp.sum(jnp.sin(x) + jnp.cos(x) + x**2 + x, axis=-1)

def combined_potential(x, t):
    return (jnp.sin(jnp.pi * t/2)**2 * styblinski_tang(x) 
            + jnp.cos(jnp.pi * t/2)**2 * oakley_ohagan(x)).sum()

def analytic_potential(n_samples, times, key, d, var):
    x0 = jax.random.normal(key, (n_samples, 1, d)) * jnp.sqrt(var)
    grad_s = lambda x, t: - jax.grad(combined_potential)(x, t)
    return jax.vmap(lambda x, key: generate_sample_s(x, grad_s, times, 0, key, 0.0))(x0, jax.random.split(key, n_samples))

### Inertial particle data

def inertial_particles(tau, eps, N, T, L, viscosity, max_velocity, resolution, particle_key, fluid_key):
    
    # physical parameters
    grid = grids.Grid((resolution, resolution), domain=((0, L), (0, L)))
    dt = cfd.equations.stable_time_step(max_velocity, .5, viscosity, grid)

    # setup step function using crank-nicolson runge-kutta order 4
    smooth = True # use anti-aliasing 

    # **use predefined settings for Kolmogorov flow**
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.ForcedNavierStokes2D(viscosity, grid, smooth=smooth), dt)

    dt *= 0.2
    final_time = T
    outer_steps = (final_time // dt)
    inner_steps = 1

    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)

    # initial velocity field
    v0 = cfd.initial_conditions.filtered_velocity_field(fluid_key, grid, max_velocity, 4)
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)

    _, trajectory = trajectory_fn(vorticity_hat0)

    from jax_cfd.spectral import utils as spectral_utils

    velocity_solve = spectral_utils.vorticity_to_velocity(grid)

    @jax.jit
    def reconstruct_velocities(vorticity_hat):
        vxhat, vyhat = velocity_solve(vorticity_hat)
        return (jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat))

    from jax_cfd.base.grids import GridArray

    x_offset = v0[0].array.offset
    y_offset = v0[1].array.offset

    def to_grid_array(arr, offset, grid):
        return GridArray(arr, offset, grid)

    from jax_cfd.base.interpolation import point_interpolation

    dt_inner = 1

    def push_particles(tau, N, trajectory, grid, dt):
        
        @jax.jit
        def u(x, ux, uy):
            _ux = point_interpolation(x, ux, mode='wrap')
            _uy = point_interpolation(x, uy, mode='wrap')
            return jnp.stack((_ux, _uy), axis=-1)
        
        ux, uy = reconstruct_velocities(trajectory[0])
        ux = to_grid_array(ux, x_offset, grid)
        uy = to_grid_array(uy, y_offset, grid)
        
        X = [jax.random.uniform(particle_key, (N, 2), minval=0, maxval=2*jnp.pi) ]
        V = [jax.vmap(lambda x: u(x, ux, uy))(X[-1])] 
        # initial particle velocity field is equal to flow field

        
        key, _ = jax.random.split(particle_key)
        _V = V[-1]
        _X = X[-1]
        _i = 0
        for i, t in enumerate(trajectory):
            key, _ = jax.random.split(key)
            ux, uy = reconstruct_velocities(t)
            ux = to_grid_array(ux, x_offset, grid)
            uy = to_grid_array(uy, y_offset, grid)

            U = jax.vmap(lambda x: u(x, ux, uy))(_X)
            _V = _V + dt/tau * (U - _V) + eps * jax.random.normal(key, _V.shape) * jnp.sqrt(dt)
            _X += dt * _V
            _X = jnp.mod(_X + 2*jnp.pi, 2*jnp.pi)
            
            if i % dt_inner == 0:
                X.append(_X)
                V.append(_V)
            
        return jnp.array(X) #, jnp.array(V)
    
    return push_particles(tau, N, trajectory, grid, dt)



def get_inertial_partices(key, n_samples, train=True):

    particle_noise = 0.0

    ### fluid model
    T = 0.5 # ideally we want to run for longer for prettier pictures
    viscosity = 1e-3
    max_velocity = 7
    resolution = 256
    L = 2*jnp.pi

    ### particles
    Nx = n_samples
    Nmu = 16
    log_tau_min = jnp.log(0.01)
    log_tau_max = jnp.log(1.0)


    ### test set
    mu_train = jnp.logspace(log_tau_min, log_tau_max, num=Nmu, base=jnp.exp(1))
    mu_train = jnp.sort(mu_train)

    n_test = 6 # usually 32
    key, _ = jax.random.split(key)
    mu_test = jax.random.uniform(key, shape=(n_test,), minval=(log_tau_min), maxval=(log_tau_max))
    mu_test = jnp.logspace(log_tau_min, log_tau_max, num=n_test, base=jnp.exp(1)) * ( 1 + 0.1 * jax.random.normal(key, shape=(n_test,)) )
    mu_test = jnp.sort(mu_test)


    ###
    # Training data
    ###
    key, fluid_key = jax.random.split(key)
    key, mu_key = jax.random.split(key)


    if train:
        mu_data = mu_train
        keys = jax.random.split(key, len(mu_data))
        x_data = jax.vmap(lambda tau, key: inertial_particles(tau, particle_noise, Nx, T, L, viscosity, max_velocity, resolution, key, fluid_key))(mu_data, keys).transpose(2, 1, 0, 3) 
    else: 
        mu_data = mu_test
        keys = jax.random.split(key, len(mu_data))
        x_data = jax.vmap(lambda tau, key: inertial_particles(tau, particle_noise, Nx, T, L, viscosity, max_velocity, resolution, key, fluid_key))(mu_data, keys).transpose(2, 1, 0, 3) 


    t_eval = jnp.linspace(0.0, T, x_data.shape[1])
    x_data = rearrange(x_data, 'N T M D -> M T N D')

    return x_data, mu_data, t_eval