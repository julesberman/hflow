import h5py
import jax
import jax.numpy as jnp


def get_regular_grid(domain, Ns, periodic=False):
    spacing = []
    for (a, b), N in zip(domain, Ns):
        d = jnp.linspace(a, b, N, endpoint=not periodic)
        spacing.append(d)
    m_grids = jnp.meshgrid(*spacing, indexing='ij')
    m_grids = [m.flatten() for m in m_grids]
    X = jnp.array(m_grids).T

    return X, spacing


def make_attach_scale(f, key, dim_i, scale_arr, scale_name=''):
    # label dimension
    f['u'].dims[dim_i].label = key
    # set scale in file
    f[key] = scale_arr
    # make scale
    f[key].make_scale(key)
    # attach scale to dimension
    f['u'].dims[dim_i].attach_scale(f[key])


def save_hdf5(equation, variant, u, scales, info={}, mu=None, path=None):

    Q = u.shape[0]
    D = len(u.shape)-2

    labels = ['q', 't', 'x0', 'x1', 'x2']
    filename = f'{equation}_{variant}_{D}D_{Q}Q'
    if path is None:
        path = './hdf5_data'
    f = h5py.File(f'{path}/{filename}.hdf5', 'w')
    f.attrs['equation'] = equation
    f.attrs['variant'] = variant
    f.attrs['input_dim'] = D
    f.attrs['output_dim'] = Q

    f.create_dataset("u", data=u)

    for i, (scale, label) in enumerate(zip(scales, labels)):
        make_attach_scale(f, label, i, scale)

    if mu is not None:
        f.attrs['mu'] = mu

    # add info
    info_g = f.create_group("info")
    for k, v in info.items():
        info_g.attrs[k] = v

    # clean up
    f.close()
