from pathlib import Path
import glob
import inspect
import os
import h5py
import jax.numpy as jnp
import numpy as np
import pandas as pd
from einops import rearrange

def load_with_pattern(directory, filename_pattern):
    search_pattern = os.path.join(directory, filename_pattern)
    matching_files = glob.glob(search_pattern)
    return matching_files

def get_lanl_data(epsilon, sub_x, sub_t):
    dir = Path("/scratch/jmb1174/lanl")
    path_wild = f"data_epsilon_{epsilon}_*"

    file_paths = load_with_pattern(dir, path_wild)
    sols = []
    for f in file_paths:
        with h5py.File(dir / f, "r") as h5_file:
            # pressure = h5_file['pressure'][::4,::4]
            saturation = h5_file["saturation"][::sub_x, ::sub_x, ::sub_t]
            saturation = rearrange(saturation, " X Y T -> T X Y")
            sols.append(saturation)

    sols = np.asarray(sols)

    # crop due to bad data
    sols = sols[10:, :, 4:, 4:]
    # sols = np.concatenate([np.zeros_like(sols[:, :1]), sols], axis=1)

    return sols
