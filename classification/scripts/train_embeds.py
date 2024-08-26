import sys
import numpy as np
import pickle as pkl
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any
import pytorch_lightning as pl
import json
import argparse

_folder_current = Path(__file__).parent
_folder = _folder_current.parent.parent
sys.path.append(str(_folder))
from classification.model.model import SequenceModel
from classification.model.pl import PLModelSeq
from classification.data.dataset import get_dataloader

_folder_models = _folder.joinpath("res", "models")
_folder_checkpoints = _folder.joinpath("res", "checkpoints")
_folder_logs = _folder.joinpath("res", "logs")
_folder_models.mkdir(exist_ok=True)
_folder_checkpoints.mkdir(exist_ok=True)
_folder_logs.mkdir(exist_ok=True)


def main(
        model_name: str,
        filename_embeds_train: Path,
        filename_embeds_val: Path,
        filename_data : Path = _folder.joinpath("data", "processed", "split.pkl"),
        n_epochs : int = 40,
        batch_size : int = 32,
        lr : float = 1e-3,
        weight_decay : float = 0.,
        load_model : bool = True,
        target_mode : Literal["multiclass", "multilabel"] = "multilabel",
        model_params : Dict[str, Any] = {},
        ):
    
    random_seed = 0
    pl.seed_everything(random_seed)
    np.random.seed(random_seed)
    
    data = pkl.load(open(filename_data, "rb"))
    embeds_train = pkl.load(open(filename_embeds_train, "rb"))
    embeds_val = pkl.load(open(filename_embeds_val, "rb"))
    
    dataloader_train = get_dataloader(data,
        embeds=embeds_train["embeds"],
        preds=embeds_train["preds"],
        dataset_mode="embeds",
        mode="train",
        # split_set="val",
        split_set="train",
        batch_size=batch_size,
    )
    dataloader_val = get_dataloader(data,
        embeds=embeds_val["embeds"],
        preds=embeds_val["preds"],
        dataset_mode="embeds",
        mode="val",
        # split_set="test",
        split_set="val",
        batch_size=batch_size,
    )
    model_params["feature_dim"] = dataloader_val.dataset.embed_dim
    
    folder_model = _folder_models.joinpath(model_name)
    folder_checkpoints = _folder_checkpoints.joinpath(model_name)
    folder_logs = _folder_logs.joinpath(model_name)
    
    folder_model.mkdir(exist_ok=True)
    folder_checkpoints.mkdir(exist_ok=True)
    folder_logs.mkdir(exist_ok=True)
    
    params = {
        "model_params": model_params,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "dataset": dataloader_val.dataset.get_config(),
        "lr": lr,
        "weight_decay": weight_decay,
        "target_mode": target_mode,
    }
    filename_params = folder_model.joinpath("params.json")
    json.dump(params, open(filename_params, "w"), indent=4)
    
    model = SequenceModel(**model_params)
    
    filename_checkpoint = folder_model.joinpath("last.ckpt")
    if load_model and filename_checkpoint.exists():
        plmodel = PLModelSeq.load_from_checkpoint(filename_checkpoint)
    else:
        plmodel = PLModelSeq(
            model=model,
            folder_model=folder_model,
            folder_checkpoints=folder_checkpoints,
            optimizer_lr=lr,
            optimizer_weight_decay=weight_decay,
            target_mode=target_mode,
        )
        
    logger = pl.loggers.TensorBoardLogger(folder_logs, name="", version="")
    trainer = pl.Trainer(max_epochs=n_epochs, logger=logger, accelerator="cpu")
    trainer.fit(plmodel, dataloader_train, dataloader_val)
    
    return


def _parse_args(args : Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Runs training of sequential model on embeds from 2d")
    parser.add_argument("model_name", type=str)
    parser.add_argument("embeds_train", type=Path)
    parser.add_argument("embeds_val", type=Path)
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
        filename_embeds_train=args.embeds_train,
        filename_embeds_val=args.embeds_val,
        load_model=args.load_model,
        **config,
    )