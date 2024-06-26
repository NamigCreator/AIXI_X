import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
import argparse
import pickle

_folder_current = Path(__file__).parent
_folder = _folder_current.parent.parent
sys.path.append(str(_folder))
from classification.data.misc import get_metadata_from_dcm, n_classes, class_name_to_index
from classification.utils import Progress, init_logger

logger = init_logger(__name__)


def get_data_from_df(filename: Path) -> Tuple[np.ndarray, np.ndarray]:
    logger.debug(f"Reading dataframe: {filename}")
    data = pd.read_csv(str(filename))
    data["id"] = data["ID"].str.split("_", n=3, expand=True)[1]
    data["class"] = data["ID"].str.split("_", n=3, expand=True)[2]
    logger.debug(f"Read {data.shape}")

    id_to_indexes = {}
    id_to_index = {}
    for i, id in enumerate(data.id.values):
        id_to_indexes.setdefault(id, []).append(i)
        if id not in id_to_index:
            id_to_index[id] = len(id_to_index)

    labels = np.zeros((len(id_to_index), n_classes), np.float32)
    for id, c, l in zip(data.id.values, data["class"].values, data.Label.values):
        index = id_to_index[id]
        c = class_name_to_index[c]
        labels[index, c] = l

    # labels for NONE := 1 - ANY
    labels[:, -1] = ~np.any(labels[:, :-1], axis=1)

    logger.info(f"Retrieved [{len(id_to_index)}] ids")

    return np.array(list(id_to_index.keys())), labels


def _parse_args(args : Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Merges data from competition train and test data and metadata from dcm files")
    parser.add_argument("--folder", type=Path, 
        default=_folder.joinpath("data", "rsna-intracranial-hemorrhage-detection"),
        help="Path to folder containing train and test data from kaggle competition",
    )
    parser.add_argument("--out", type=Path,
        default=_folder.joinpath("data", "processed"),
        help="Output folder",
    )
    parser.add_argument("--with_metadata", default=True, type=int, choices=[0, 1],
        help="If 1, metadata from dcm files is processed and saved into csv files")
    return parser.parse_args(args=args)


def main(args : Optional[List[str]] = None):
    args = _parse_args(args)

    folder_dcm_train = args.folder.joinpath("stage_2_train")
    folder_dcm_test = args.folder.joinpath("stage_2_test")
    filename_train = args.folder.joinpath("stage_2_train.csv")
    filename_sample = args.folder.joinpath("stage_2_sample_submission.csv")

    ids_train, labels_train = get_data_from_df(filename_train)
    ids_test, labels_test = get_data_from_df(filename_sample)

    args.out.mkdir(exist_ok=True)
    filename_ids_train = args.out.joinpath("train_ids_labels.pkl")
    filename_ids_test = args.out.joinpath("test_ids_labels.pkl")
    pickle.dump((ids_train, labels_train), open(filename_ids_train, "wb"))
    pickle.dump((ids_test, labels_test), open(filename_ids_test, "wb"))

    filename_out_dcm_train = args.out.joinpath("train_dcm_data.csv")
    filename_out_dcm_test = args.out.joinpath("test_dcm_data.csv")

    dcm_data_train = []
    for i in Progress(ids_train, desc="Reading metadata from train dcm"):
        filename = folder_dcm_train.joinpath(f"ID_{i}.dcm")
        d = get_metadata_from_dcm(filename)
        dcm_data_train.append(d)
    dcm_data_train = pd.DataFrame(dcm_data_train)
    dcm_data_train.to_csv(filename_out_dcm_train)

    dcm_data_test = []
    for i in Progress(ids_test, desc="Reading metadata from test dcm"):
        filename = folder_dcm_test.joinpath(f"ID_{i}.dcm")
        d = get_metadata_from_dcm(filename)
        dcm_data_test.append(d)
    dcm_data_test = pd.DataFrame(dcm_data_test)
    dcm_data_test.to_csv(filename_out_dcm_test)

    return


if __name__ == "__main__":
    main()