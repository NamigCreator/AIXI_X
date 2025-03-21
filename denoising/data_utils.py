import multiprocessing
from pathlib import Path
import sys

import pydicom
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
# from utils.dcm_utils import get_image_from_dcm, window_image
from general_utils.dcm_utils import get_image_from_dcm, window_image

# from utils.filters import get_filter_by_name
from general_utils.filters import get_filter_by_name

import os
from typing import List, Union
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from pytorch_lightning import LightningDataModule

import numpy as np
import cv2 as cv
import torch

import pickle


anomalies = [
    "00c5876eb",
    "0ba50cfdf",
    "0d5797d6a",
    "1c41b6175",
    "3afb77129",
    "3eb121826",
    "0a0605eb0",
    "0bd6d4fb2",
    "0ea0a3e8e",
    "2f616707a",
    "3da04c3bc",
]


class CTBrainDataset(Dataset):
    def __init__(
        self,
        img_dir,
        info_table,
        subset="train",
        transform=None,
        target_transform=None,
        threshold=0.02,
        window_center=40,
        window_width=80,
    ):
        self.img_dir = Path(img_dir)
        self.info_table = load_json(info_table)
        self.subset = subset
        self.window_center = window_center
        self.window_width = window_width
        self.threshold = threshold
        self.transform = transform
        if transform == "train":
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.RandomRotation(degrees=(-30, 30)),
                    v2.RandomHorizontalFlip(p=0.5),
                ]
            )
        self.target_transform = target_transform

        if self.subset == "train":
            self.indices, self.labels = self.remain_healthy()
        elif self.subset == "val":
            self.indices, self.labels = self.remain_anomalies()
        elif self.subset == "test":
            self.indices, self.labels = self.remain_anomalies()
        else:
            self.indices = self.info_table[f"{self.subset}_inds"]
            self.labels = self.info_table[f"{self.subset}_labels"]

        if threshold is not None:
            self.indices, self.labels = self.delete_empty_scans()

    def remain_healthy(self):
        indices = []
        labels = []
        for i, l in zip(
            self.info_table[f"{self.subset}_inds"],
            self.info_table[f"{self.subset}_labels"],
        ):
            # l is a one-hot vector (0, 0, 0, 0, 0, 1)
            if l[5]:
                indices.append(i)
                labels.append(l)
        return indices, labels

    def remain_anomalies(self):
        indices = []
        labels = []
        for i, l in zip(
            self.info_table[f"{self.subset}_inds"],
            self.info_table[f"{self.subset}_labels"],
        ):
            name = self.info_table[f"{self.subset}_data"].loc[i, "sop_id"]
            if name in set(anomalies):
                indices.append(i)
                labels.append(l)
        return indices, labels

    def delete_empty_scans(self):
        remain_indices = []
        remain_labels = []
        for i, (img, _, _, _) in enumerate(self):
            occupancy = img.mean()
            if occupancy > self.threshold:
                remain_indices.append(i)

        # convert consecutive indices to indices in dataframe
        remain_labels = [self.labels[i] for i in remain_indices]
        remain_indices = [self.indices[i] for i in remain_indices]

        return remain_indices, remain_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        i = self.indices[idx]
        label = self.labels[idx]  # one-hot vector
        label = np.argmax(label)  # one-hot vector to idx of a class
        name, _, _, intercept, slope = self.info_table[f"{self.subset}_data"].loc[
            i, ["sop_id", "window_center", "window_width", "intercept", "slope"]
        ]
        img_path = self.img_dir.joinpath(f"ID_{name}.npy")

        image = np.load(img_path)  # (H, W)
        image = window_image(
            image, self.window_center, self.window_width, intercept, slope
        )
        image = normalize_minmax(image)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).unsqueeze(0)

        edges = apply_canny(image, maxVal=120)
        image = image.to(torch.float32)
        edges = edges.to(torch.float32)

        return (image, edges, label, name)

    

class AIMIDataset:
    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        patient_ids: Union[str, os.PathLike],
        subset="train",
        transform=None,
        target_transform=None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.patient_ids = np.load(patient_ids, allow_pickle=True)
        self.pathes = self._get_pathes_from_patient_ids(self.patient_ids)
        self._delete_empty_scans()

        self.transform = None
        if transform == "train":
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.RandomRotation(degrees=(-30, 30)),
                    v2.RandomHorizontalFlip(p=0.5),
                ]
            )
        self.target_transform = target_transform

    def _get_pathes_from_patient_ids(self, patient_ids: List[str]) -> List[str]:
        pathes = []
        for p_id in patient_ids:
            num = p_id.split("_")[1]
            if len(num) < 4:
                batch_num = 1
            else:
                batch_num = int(num[0])
            dcm_folder = self.data_dir.joinpath(
                f"batch_{batch_num}/{p_id}/reconstructed_image"
            )
            patient_dcms = list(dcm_folder.rglob("image_*.dcm"))
            pathes.extend(patient_dcms)
        return pathes

    def _delete_empty_scans(self) -> None:

        print(f"Number of scans: {len(self.pathes)}")
        with multiprocessing.Pool(os.cpu_count()) as p:
            emptiness_mask = p.map(check_dcm_for_emptiness, self.pathes)
        self.pathes = [f for m, f in zip(emptiness_mask, self.pathes) if m]
        print(f"Number of non-empty scans remain: {len(self.pathes)}")

    def __len__(self):
        return len(self.pathes)

    def __getitem__(self, idx):

        dcm_path = self.pathes[idx]

        name = str(dcm_path).split("/")
        name = f"{name[-3]}_{name[-1]}"

        image = get_image_from_dcm(dcm_path)
        image = window_image(image, window_center=40, window_width=80)

        image = normalize_minmax(image)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).unsqueeze(0)

        edges = apply_canny(image, maxVal=120)
        image = image.to(torch.float32)
        edges = edges.to(torch.float32)

        return (image, edges, 0, name)

def preprocess_dicom(dcm_path):
    
    image = get_image_from_dcm(dcm_path)
    image = window_image(image, window_center=40, window_width=80)
    
    image = normalize_minmax(image)
    image_numpy = image

    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    edges = apply_canny(image, maxVal=120).unsqueeze(0)
    
    image_tensor = image.to(torch.float32)
    edges_tensor = edges.to(torch.float32)
    concat_tensor = torch.concat([image_tensor, edges_tensor], dim=1)

    return image_numpy, image_tensor, edges_tensor, concat_tensor 


class AIMIBrainDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        healthy_patient_ids: Union[str, os.PathLike],
        img_dir: Union[str, os.PathLike],
        info_table: Union[str, os.PathLike],
        batch_size: int = 24,
    ):
        super().__init__()

        # aimi dataset
        self.data_dir = data_dir
        self.healthy_patient_ids = healthy_patient_ids

        # rsna dataset for validation
        self.img_dir = img_dir
        self.info_table = info_table

        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == "fit":
            self.train_set = AIMIDataset(
                data_dir=self.data_dir, patient_ids=self.healthy_patient_ids
            )
            self.val_set = CTBrainDataset(
                img_dir=self.img_dir, info_table=self.info_table, subset="val"
            )
        elif stage == "test":
            self.test_set = self.val_set = CTBrainDataset(
                img_dir=self.img_dir, info_table=self.info_table, subset="val"
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1)


class CTBrainDataModule(LightningDataModule):
    def __init__(self, img_dir: str, info_table: str, batch_size: int = 32):
        super().__init__()
        self.img_dir = img_dir
        self.info_table = info_table
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == "fit":
            self.train_set = CTBrainDataset(
                img_dir=self.img_dir,
                info_table=self.info_table,
                subset="train",
                transform="train",
            )
            self.val_set = CTBrainDataset(
                img_dir=self.img_dir, info_table=self.info_table, subset="val"
            )
        elif stage == "test":
            self.test_set = CTBrainDataset(
                img_dir=self.img_dir, info_table=self.info_table, subset="val"
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1)


def read_image(fname):
    img = cv.imread(str(fname))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def apply_canny(
    img: Union[torch.Tensor, np.ndarray], minVal=100, maxVal=200
) -> Union[np.ndarray, torch.Tensor]:
    return_ndarray = True
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        return_ndarray = False
    img = img * 255
    img = img.astype(np.uint8)
    img = img.squeeze()
    canny = cv.Canny(img, minVal, maxVal)
    canny = canny / 255
    if return_ndarray:
        return canny.unsquueeze(0)
    return torch.from_numpy(canny).unsqueeze(0)  # (1, H, W)


def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    if (mi == 0 and ma == 0) or (ma == mi):
        return img
    return (img - mi) / (ma - mi)


def load_json(fname):
    with open(fname, "rb") as fin:
        return pickle.load(fin)


def check_dcm_for_emptiness(fname) -> bool:
    f = get_filter_by_name(name="brain_present")
    img = get_image_from_dcm(fname)
    # f takes unwindowed image
    is_ok = f(img)
    if is_ok is None:
        return 0
    else:
        return 1


if __name__ == "__main__":

    # fname = "/home/mark/Data/tmp/br/ctsinogram/head_ct_dataset_anon/batch_9/series_9104/reconstructed_image/.azDownload-a8a49a28-daed-dc4f-7802-11a0308cf3ce-image_23.dcm"
    # img = get_image_from_dcm(fname)
    # dcm = pydicom.dcmread(str(fname), force=True)
    # print(dcm)
    # dataset = AIMIDataset(
    # data_dir="/home/mark/Data/tmp/br/ctsinogram/head_ct_dataset_anon",
    # patient_ids="/home/mark/Data/tmp/br/ctsinogram/head_ct_dataset_anon/healthy_patient_ids.npy",
    # )

    proj_dir = Path(__file__).resolve().parent.parent
    info_table = proj_dir.joinpath("data/split_subset_005.pkl")
    img_dir = proj_dir.joinpath("data/train_npy_subset_005")

    datamodule = AIMIBrainDataModule(
        data_dir="/home/mark/Data/tmp/br/ctsinogram/head_ct_dataset_anon",
        healthy_patient_ids="/home/mark/Data/tmp/br/ctsinogram/head_ct_dataset_anon/healthy_patient_ids.npy",
        info_table=info_table,
        img_dir=img_dir,
    )

    # datamodule.setup("fit")
    # for b in datamodule.val_dataloader():
    # print(b)
    # break
