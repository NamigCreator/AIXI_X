import sys
import numpy as np
import pickle
from pathlib import Path
from typing import Literal, Optional, List
import pytorch_lightning as pl
import json
import argparse

_folder_current = Path(__file__).parent
_folder = _folder_current.parent.parent
sys.path.append(str(_folder))
from classification.model.model import get_model
from classification.model.pl import PLModel, PLModel3D
from classification.data.dataset import get_dataloader

_folder_models = _folder.joinpath("res", "models")
_folder_checkpoints = _folder.joinpath("res", "checkpoints")
_folder_logs = _folder.joinpath("res", "logs")
_folder_models.mkdir(exist_ok=True)
_folder_checkpoints.mkdir(exist_ok=True)
_folder_logs.mkdir(exist_ok=True)


def main(
        model_name: str,
        filename_data : Path = _folder.joinpath("data", "processed", "split.pkl"),
        folder : Path = _folder.joinpath("data", "processed", "train_npy"),
        mode : Literal["2d_single_channel", "2d_multichannel", "3d"] = "2d_single_channel",
        n_epochs : int = 40,
        batch_size : int = 32,
        filters : List[str] = ["brain_present"],
        lr : float = 1e-4,
        model_arch : str = "efficientnet",
        weight_decay : float = 0.,
        load_model : bool = True,
        ):

    random_seed = 0
    pl.seed_everything(random_seed)
    np.random.seed(random_seed)

    data = pickle.load(open(filename_data, "rb"))
    dataloader_train = get_dataloader(data, folder, 
        dataset_mode=mode, 
        mode="train", 
        split_set="train", 
        batch_size=batch_size,
        filters=filters,    
    )
    dataloader_val = get_dataloader(data, folder,
        dataset_mode=mode, 
        mode="val", 
        split_set="val", 
        batch_size=batch_size,
        filters=filters,    
    )

    folder_model = _folder_models.joinpath(model_name)
    folder_checkpoints = _folder_checkpoints.joinpath(model_name)
    folder_logs = _folder_logs.joinpath(model_name)

    folder_model.mkdir(exist_ok=True)
    folder_checkpoints.mkdir(exist_ok=True)
    folder_logs.mkdir(exist_ok=True)

    params = {
        "mode": mode,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "dataset": dataloader_val.dataset.get_config(),
        "filters": filters,
        "model_arch": model_arch,
        "lr": lr,
    }
    filename_params = folder_model.joinpath("params.json")
    json.dump(params, open(filename_params, "w"), indent=4)

    model = get_model(mode=mode, model_name=model_arch)

    if mode.startswith("2d"):
        plmodel = PLModel
    elif mode.startswith("3d"):
        plmodel = PLModel3D
    else:
        raise ValueError(f"Unkown mode: {mode}")

    filename_checkpoint = folder_model.joinpath("last.ckpt")
    if load_model and filename_checkpoint.exists():
        plmodel = plmodel.load_from_checkpoint(filename_checkpoint)
    else:
        plmodel = plmodel(
            model=model,
            folder_model=folder_model,
            folder_checkpoints=folder_checkpoints,
            optimizer_lr=lr,
            optimizer_weight_decay=weight_decay,
        )

    logger = pl.loggers.TensorBoardLogger(folder_logs, name="", version="")
    trainer = pl.Trainer(max_epochs=n_epochs, logger=logger)
    trainer.fit(plmodel, dataloader_train, dataloader_val)

    return


def _parse_args(args : Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Runs model training")
    parser.add_argument("model_name", type=str)
    parser.add_argument("--config", type=Path, default=None,
        help="Path to config file")
    parser.add_argument("--load_model", type=int, choices=[0, 1], default=True)
    return parser.parse_args(args)


if __name__ == "__main__":
    args = _parse_args()
    config = {}
    if args.config is not None:
        config = json.load(open(args.config))
    main(
        model_name=args.model_name,
        load_model=args.load_model,
        **config,
    )
