import glob
import os

import h5py
import numpy as np


def read_hd5f_numpy(filepath):
    with h5py.File(filepath, "r") as f:
        dataset = f['u'][()]
        spacing = []
        for d in f['u'].dims:
            spacing.append(f[d.label][()])
        mu = None
        if 'mu' in f.attrs.keys():
            mu = np.asarray(f.attrs['mu'], dtype=np.float32)
        return dataset, spacing, mu


def load_all_hdf5(path):
    mus = []
    sols = []
    files = glob.glob(os.path.join(path, "*.hdf5"))
    for filepath in files:
        sol, space, mu = read_hd5f_numpy(filepath)
        mus.append(mu.ravel())
        sols.append(sol)

    mus = np.asarray(mus)
    sols = np.asarray(sols)

    # first sort by mu
    if mus.shape[1] == 1:
        sort_idx = np.squeeze(np.argsort(mus, axis=0))
        mus, sols = mus[sort_idx], sols[sort_idx]

    return mus, sols, space


def normalize(x, axis=None, return_stats=False, method='std'):
    if method == '01':
        mm, mx = x.min(axis=axis, keepdims=True), x.max(
            axis=axis, keepdims=True)
        shift, scale = mm, (mx-mm)
    else:
        shift, scale = np.mean(x, axis=axis, keepdims=True),  np.std(
            x, axis=axis, keepdims=True)

    x = (x - shift) / scale

    if return_stats:
        return x, shift, scale
    else:
        return x
