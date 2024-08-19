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
from classification.model.model import UNet
from classification.model.pl import PLModelSegm
from classification.data.dataset import get_dataloader

_folder_models = _folder.joinpath("res", "models")
_folder_checkpoints = _folder.joinpath("res", "checkpoints")
_folder_logs = _folder.joinpath("res", "logs")
_folder_models.mkdir(exist_ok=True)
_folder_checkpoints.mkdir(exist_ok=True)
_folder_logs.mkdir(exist_ok=True)


def main(
        model_name: str,
        filename_data : Path = _folder.joinpath("data", "bhsd-dataset", "split_1.pkl"),
        folder : Path = _folder.joinpath("data", "bhsd-dataset", "images"),
        folder_labels : Path = _folder.joinpath("data", "bhsd-dataset", "ground_truths"),
        n_epochs : int = 100,
        batch_size : int = 32,
        lr : float = 1e-3,
        weight_decay : float = 0.,
        loss_type : Literal["dice", "crossentropy"] = "dice",
        load_model : bool = True,
        n_classes : Optional[int] = None,
        target_mode : Literal["multilabel", "multiclass", "binary"] = "binary",
        is_3d : bool = False,
        ):
    
    random_seed = 0
    pl.seed_everything(random_seed)
    np.random.seed(random_seed)
    
    data = pickle.load(open(filename_data, "rb"))
    dataset_args = {
        "folder_images": folder,
        "folder_labels": folder_labels,
        "batch_size": batch_size,
        "filters": [],
        "dataset_mode": "segm",
        "n_classes": n_classes,
        "is_3d": is_3d,
        "preload": True,
    }
    dataloader_train = get_dataloader(data,
        mode="train",
        split_set="train",
        **dataset_args,
    )
    dataloader_val = get_dataloader(data,
        mode="val",
        split_set="val",
        **dataset_args,
    )
    
    folder_model = _folder_models.joinpath(model_name)
    folder_checkpoints = _folder_checkpoints.joinpath(model_name)
    folder_logs = _folder_logs.joinpath(model_name)
    
    folder_model.mkdir(exist_ok=True)
    folder_checkpoints.mkdir(exist_ok=True)
    folder_logs.mkdir(exist_ok=True)
    
    params = {
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "dataset": dataloader_val.dataset.get_config(),
        "lr": lr,
        "weight_decay": weight_decay,
        "loss_type": loss_type,
        "n_classes": n_classes,
        "target_mode": target_mode,
        "is_3d": is_3d,
    }
    filename_params = folder_model.joinpath("params.json")
    json.dump(params, open(filename_params, "w"), indent=4)
    
    filename_checkpoint = folder_model.joinpath("last.ckpt")
    if load_model and filename_checkpoint.exists():
        plmodel = plmodel.load_from_checkpoint(filename_checkpoint)
    else:
        model = UNet(
            out_dim=(1 if n_classes is None else n_classes),
            mode=("3d" if is_3d else "2d"),
        )
        plmodel = PLModelSegm(
            model=model,
            folder_model=folder_model,
            folder_checkpoints=folder_checkpoints,
            optimizer_lr=lr,
            optimizer_weight_decay=weight_decay,
            loss_type=loss_type,
            n_classes=n_classes,
            target_mode=target_mode,
        )
    
    logger = pl.loggers.TensorBoardLogger(folder_logs, name="")
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