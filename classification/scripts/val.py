import sys
from pathlib import Path
from typing import Literal, Union, Sequence, Optional, List
import pytorch_lightning as pl
import json
import argparse

_folder_current = Path(__file__).parent
_folder = _folder_current.parent.parent
sys.path.append(str(_folder))
from classification.model.model import get_model
from classification.model.pl import PLModel, PLModel3D
from classification.data.dataset import get_dataloader
from classification.scripts.train import _folder_models, _folder_checkpoints


def main(
        model_name: str,
        filename_data : Path = _folder.joinpath("data", "processed", "split.pkl"),
        folder : Path = _folder.joinpath("data", "processed", "train_npy"),
        split_set : Union[Literal["train", "val", "test"], Sequence[Literal["train", "val", "test"]]] = ["val", "test"],
        checkpoint_name : str = "best"
        ):
    
    filename_params = _folder_models.joinpath(model_name, "params.json")
    filename_checkpoint = _folder_checkpoints.joinpath(model_name, f"{checkpoint_name}.ckpt")

    params = json.load(open(filename_params))
    mode = params["mode"]

    model = get_model(
        model_type=mode, 
        filename_checkpoint=filename_checkpoint,
        model_name=params.get("model_arch", "efficientnet"),
    )
    model.eval()

    if mode.startswith("2d"):
        plmodel = PLModel
    elif mode.startswith("3d"):
        plmodel = PLModel3D
    else:
        raise ValueError(f"Unknown mode: {mode}")
    plmodel = plmodel(model)

    trainer = pl.Trainer()

    if isinstance(split_set, str):
        split_set = [split_set]
    for s in split_set:
        dataloader = get_dataloader(
            filename_data, 
            folder, 
            dataset_mode=mode, 
            split_set=s, 
            batch_size=params["batch_size"],
        )
        trainer.validate(plmodel, dataloader)
    return


def _parse_args(args : Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Calculates validation scores on prediction")
    parser.add_argument("model", type=str, help="Model name")
    parser.add_argument("--split", type=str, nargs="+", choices=["train", "val", "test"], default=["val", "test"],
        help="Names of split sets on which scores should be calculated")
    parser.add_argument("--checkpoint", type=str, default="best",
        help="Checkpoint name")
    return parser.parse_args(args)


if __name__ == "__main__":
    args = _parse_args()
    main(
        model_name=args.model,
        split_set=args.split,
        checkpoint_name=args.checkpoint,
    )