import sys
import numpy as np
from pathlib import Path
import pydicom
from typing import Optional, Tuple, List
import cv2
import argparse

_folder_current = Path(__file__).parent
_folder = _folder_current.parent.parent
sys.path.append(str(_folder))
from classification.data.misc import get_image_from_dcm
from classification.utils import run_multiprocess, init_logger

logger = init_logger(__name__)


def convert_dcm_npy(filename_in: Path, filename_out: Path, **kwargs):
    try:
        img = get_image_from_dcm(filename_in, **kwargs)
        np.save(filename_out, img)
    except:
        logger.error(f"Failed to convert {filename_in} -> {filename_out}")
    return


def convert_dcm_folder_to_npy(
        folder_in: Path,
        folder_out: Path,
        id_list : Optional[List[str]] = None,
        threads : int = 1,
        ignore_present : bool = False,
        **kwargs,
        ):
    
    if not isinstance(folder_in, Path):
        folder_in = Path(folder_in)
    if not isinstance(folder_out, Path):
        folder_out = Path(folder_out)

    if id_list is None:
        id_list = [f.stem for f in folder_in.iterdir() if f.suffix == ".dcm"]
        id_list = sorted(id_list)

    if ignore_present:
        ids_present = [f.stem for f in folder_out.iterdir()]
        ids_present_s = set(ids_present)
        id_list = [i for i in id_list if i not in ids_present_s]

    def get_args():
        for id in id_list:
            d = {
                "filename_in": folder_in.joinpath(f"{id}.dcm"),
                "filename_out": folder_out.joinpath(f"{id}.npy"),
            }
            d.update(kwargs)
            yield d

    folder_out.mkdir(exist_ok=True)
    run_multiprocess(convert_dcm_npy, get_args(),
        total=len(id_list), threads=threads)
    return 


def _parse_args(args : Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Converts dcm files into npy arrays")
    parser.add_argument("--data", type=Path, 
        default=_folder.joinpath("data", "rsna-intracranial-hemorrhage-detection", "stage_2_train"),
        help="Path to folder containing input .dcm files",
    )
    parser.add_argument("--out", type=Path, 
        default=_folder.joinpath("data", "processed", "train_npy"),
        help="Folder for output .npy files",
    )
    parser.add_argument("--threads", type=int, default=10,
        help="Number of processes for parallel running")
    parser.add_argument("--ignore_present", type=int, default=True, choices=[0, 1],
        help="If 1, files with ids already present in output folder are ignored")
    parser.add_argument("--image_size", type=int, default=256,
        help="Size of output dowsampled image")
    return parser.parse_args(args=args)


def main(args : Optional[List[str]] = None):
    args = _parse_args(args)

    convert_dcm_folder_to_npy(
        folder_in=args.data,
        folder_out=args.out,
        threads=args.threads,
        ignore_present=args.ignore_present,
        image_size=args.image_size,
    )
    return


if __name__ == "__main__":
    main()