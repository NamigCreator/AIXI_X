from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule

import numpy as np
import cv2 as cv
import torch

import pickle
from pathlib import Path


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

        if self.subset == "train":
            self.indices, self.labels = self.remain_healthy()
        else:
            self.indices = self.info_table[f"{self.subset}_inds"]
            self.labels = self.info_table[f"{self.subset}_labels"]

        if threshold is not None:
            self.indices, self.labels = self.delete_empty_scans()

        self.transform = transform
        self.target_transform = target_transform

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

    def delete_empty_scans(self):
        remain_indices = []
        remain_labels = []
        for i, (img, _, _, _) in enumerate(self):
            # now 3 rgb channels which are the same
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
        image = image * 255  # seems that canny filter needs uint8
        image = image.astype(np.uint8)
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        edges = {}
        for threshold in [120, 170, 200]:
            canny = apply_canny(image, maxVal=threshold)
            canny = torch.Tensor(canny).unsqueeze(0)  # (1, H, W)
            edges[threshold] = canny

        image = torch.Tensor(image)
        image = torch.moveaxis(image, -1, 0)  # (H, W, 3) -> (3, H, W)
        image = image / 255

        return (image, edges, label, name)


class CTBrainDataModule(LightningDataModule):
    def __init__(self, img_dir: str, info_table: str, batch_size: int = 32):
        super().__init__()
        self.img_dir = img_dir
        self.info_table = info_table
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == "fit":
            self.train_set = CTBrainDataset(
                img_dir=self.img_dir, info_table=self.info_table, subset="train"
            )
            self.val_set = CTBrainDataset(
                img_dir=self.img_dir, info_table=self.info_table, subset="val"
            )
        elif stage == "test":
            self.test_set = CTBrainDataset(
                img_dir=self.img_dir, info_table=self.info_table, subset="test"
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


def apply_canny(img, minVal=100, maxVal=200):
    canny = cv.Canny(img, 100, 200)
    return canny


def window_image(img, window_center, window_width, intercept, slope):
    img = img * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img


def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    if (mi == 0 and ma == 0) or (ma == mi):
        return img
    return (img - mi) / (ma - mi)


def load_json(fname):
    with open(fname, "rb") as fin:
        return pickle.load(fin)


# if __name__ == "__main__":

#     img_dir = "/home/markzaretckii/Desktop/br/data/train_npy_subset_005"
#     info_table = "/home/markzaretckii/Desktop/br/data/split_subset_005.pkl"

#     d = CTBrainDataset(
#         img_dir=img_dir,
#         info_table=info_table,
#         subset="val",
#     )

    # m = CTBrainDataModule(img_dir=img_dir, info_table=info_table)
    # m.setup(stage="fit")
    # dl = m.val_dataloader()
    # for b in dl:
    #     x_true, edges, target, name = b
    #     if name[0] in anomalies:
    #         print("Im here ", name[0])

    #     break

    # img = np.load(
    #     "/home/maestro/Desktop/brain/data/train_npy_subset_005/ID_ffffb670a.npy"
    # )
    # img = img[:90, :90]
    # img = img.reshape(3, 90, -1)
    # print(img.shape)
    # img = normalize_minmax(img)
    # img = img.astype(np.uint8)
    # apply_canny(img)
