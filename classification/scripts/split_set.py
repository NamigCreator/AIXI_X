import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Tuple, Union, List
from sklearn.model_selection import train_test_split
import argparse

_folder_current = Path(__file__).parent
_folder = _folder_current.parent.parent
sys.path.append(str(_folder))
from classification.utils import init_logger

logger = init_logger(__name__)


def split_set(
        ids: np.ndarray, 
        labels: np.ndarray, 
        data: pd.DataFrame,
        val_size : float = 0.1,
        test_size : float = 0.1,
        random_seed : Optional[int] = 0,
        subset_size : Optional[float] = None,
        stratify_by_classes : Optional[Union[int, tuple]] = None,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    logger.debug(f"Input data: {ids.shape} {labels.shape} {data.shape}")

    id_to_index = {id: i for i, id in enumerate(ids)}

    patient_ids_to_inds = {}
    patient_ids_to_inds_data = {}
    for i, (patient_id, sop_id) in enumerate(zip(data.patient_id.values, data.sop_id.values)):
        patient_ids_to_inds.setdefault(patient_id, []).append(id_to_index[sop_id])
        patient_ids_to_inds_data.setdefault(patient_id, []).append(i)
    patient_ids = list(patient_ids_to_inds.keys())    

    logger.debug(f"[{len(patient_ids_to_inds)}] unique patient ids")

    labels_patients = np.zeros((len(patient_ids_to_inds), labels.shape[1]), labels.dtype)
    for index, (patient_id, inds) in enumerate(patient_ids_to_inds.items()):
        labels_patients[index, :-1] = labels[inds, :-1].max(axis=0)
    labels_patients[:, -1] = ~np.any(labels_patients[:, :-1], axis=1)

    logger.debug(f"Count labels sop    : {labels.sum(axis=0).astype(int)}")
    logger.debug(f"Count labels patient: {labels_patients.sum(axis=0).astype(int)}")

    rel_inds = np.arange(len(patient_ids))
    if subset_size is not None:
        labels_strat = labels_patients[rel_inds]
        if stratify_by_classes is not None:
            labels_strat = labels_strat[:, stratify_by_classes]
        _, rel_inds = train_test_split(rel_inds,
            test_size=subset_size, random_state=random_seed, stratify=labels_strat)
        logger.info(f"Subset: {len(rel_inds):5d} {labels_patients[rel_inds].sum(axis=0).astype(int)}")
    
    labels_strat = labels_patients[rel_inds]
    if stratify_by_classes is not None:
        labels_strat = labels_strat[:, stratify_by_classes]
    trainval_inds, test_inds = train_test_split(rel_inds,
        test_size=test_size, random_state=random_seed, stratify=labels_strat)
    labels_strat = labels_patients[trainval_inds]
    if stratify_by_classes is not None:
        labels_strat = labels_strat[:, stratify_by_classes]
    train_inds, val_inds = train_test_split(trainval_inds, 
        test_size=(val_size)/(1.-test_size), random_state=random_seed, 
        stratify=labels_strat,
    )
    logger.info(f"Train: {len(train_inds):5d} {labels_patients[train_inds].sum(axis=0).astype(int)}")
    logger.info(f"Val  : {len(val_inds):5d} {labels_patients[val_inds].sum(axis=0).astype(int)}")
    logger.info(f"Test : {len(test_inds):5d} {labels_patients[test_inds].sum(axis=0).astype(int)}")

    train_inds_sop = np.concatenate([patient_ids_to_inds[patient_ids[i]] for i in train_inds])
    val_inds_sop = np.concatenate([patient_ids_to_inds[patient_ids[i]] for i in val_inds])
    test_inds_sop = np.concatenate([patient_ids_to_inds[patient_ids[i]] for i in test_inds])

    logger.info(f"Train: {len(train_inds_sop):6d} {labels[train_inds_sop].sum(axis=0).astype(int)}")
    logger.info(f"Val  : {len(val_inds_sop):6d} {labels[val_inds_sop].sum(axis=0).astype(int)}")
    logger.info(f"Test : {len(test_inds_sop):6d} {labels[test_inds_sop].sum(axis=0).astype(int)}")

    return train_inds_sop, val_inds_sop, test_inds_sop


def _parse_args(args : Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Splits dataset into train / validation / test sets")
    _folder_processed = _folder.joinpath("data", "processed")
    parser.add_argument("--ids", type=Path, default=_folder_processed.joinpath("train_ids_labels.pkl"),
        help="Path to pkl file containing ids and labels for train")
    parser.add_argument("--dcm_data", type=Path, default=_folder_processed.joinpath("train_dcm_data.csv"),
        help="Path to csv file containing metadata from dcm files for train")
    parser.add_argument("--out", type=Path, default=_folder_processed.joinpath("split.pkl"),
        help="Output pkl file with data for splitted train / validation / test sets")
    parser.add_argument("--subset_size", type=float, default=None,
        help="Part of initial dataset that should be used; if not specified, the whole set is used")
    parser.add_argument("--stratify_by_classes", type=int, nargs="+", default=None,
        help="Indexes of classes used in stratified split; if not specified, all classes are used")
    return parser.parse_args(args=args)


def main(args : Optional[List[str]] = None):
    args = _parse_args(args)

    ids, labels = pickle.load(open(args.ids, "rb"))
    data = pd.read_csv(str(args.dcm_data))

    train_inds, val_inds, test_inds = split_set(ids, labels, data,
        subset_size=args.subset_size, stratify_by_classes=args.stratify_by_classes)

    data = {
        "train_inds": train_inds,
        "val_inds": val_inds,
        "test_inds": test_inds,
        "train_data": data.iloc[train_inds],
        "val_data": data.iloc[val_inds],
        "test_data": data.iloc[test_inds],
        "train_labels": labels[train_inds],
        "val_labels": labels[val_inds],
        "test_labels": labels[test_inds],
    }
    pickle.dump(data, open(args.out, "wb"))
    return data


if __name__ == "__main__":
    main()