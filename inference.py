from pathlib import Path

import torch
import numpy as np
import os
from utils.aggregation import aggregate_data
from utils.calculate_features import load_features
import hydra
from os import listdir
from hydra.utils import instantiate
from os.path import isfile, join


@hydra.main(config_path="conf", config_name="config")
def processing(cfg) -> None:
    """
    processing raw data for training
    """
    threshold = cfg.threshold
    if threshold > 1 or threshold < 0:
        raise AttributeError

    np.seterr(divide="ignore")

    wavs_folder = cfg.wavs_folder
    features_folder = Path(cfg.features_folder)

    wavs_names = [f for f in listdir(wavs_folder) if isfile(join(wavs_folder, f))]
    load_features(
        wavs_path=Path(wavs_folder),
        wavs_names=wavs_names,
        result_dir=Path(features_folder),
        dataset_name="inference data",
        recalculate_feature=True,
    )
    
    labels = {0:'angry',1:'sad',2:'neutral',3:'happy'}
    names_to_id = {}
    test_samples = []
    for idx,file in enumerate((Path(features_folder) / "features").glob("*")):
        test_samples.append(np.load(file))
        names_to_id[idx] = file.stem
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.best_model_folder + "/" + cfg.model["_target_"].split(".")[-1]))
    model.eval()
    with torch.no_grad():
        for idx, feature in enumerate(test_samples):
            output_label = torch.argmax(model(torch.tensor(feature[None, :]))).item()
            print(f'file {names_to_id[idx]}: prediction: {labels[output_label]}')
        

if __name__ == "__main__":
    processing()  # pylint: disable=no-value-for-parameter