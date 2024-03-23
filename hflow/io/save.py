import pickle
from pathlib import Path

from hydra.core.hydra_config import HydraConfig

from hflow.config import Config
from hflow.io.utils import (convert_jax_to_numpy, convert_list_to_numpy,
                            flatten_config, save_pickle)


def consolidate_results(results: dict, cfg: Config):

    r_dict = convert_list_to_numpy(results)
    r_dict = convert_jax_to_numpy(r_dict)

    # flatten config
    args_d = flatten_config(cfg, '-')
    all_data = {**args_d, **r_dict}

    return all_data


def save_results(results: dict, cfg: Config):

    output_dir = HydraConfig.get().runtime.output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_name = 'result'
    output_path = (output_dir / output_name).with_suffix(".pkl")

    data = consolidate_results(results, cfg)

    save_pickle(output_path, data)
