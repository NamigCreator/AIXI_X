import numpy as np
from pathlib import Path
from typing import Literal, Optional, List, Union, Tuple, Any, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import pydicom
import cv2
import torchvision.transforms as transforms

from .misc import get_image_from_dcm, change_image_size, window_image, load_study, read_sitk_image
from .filter import Filter, get_filter_by_name
from ..utils import init_logger, Progress, get_logging_level

logger = init_logger(__name__)


class DatasetBase(Dataset):
    def __init__(self,
            folder : Optional[Path] = None,
            ids : Optional[List[str]] = None,
            images : Optional[List[np.ndarray]] = None,
            labels : Optional[np.ndarray] = None,
            patient_ids : Optional[np.ndarray] = None,
            z_positions : Optional[np.ndarray] = None,
            image_size : Tuple[int, int] = (256, 256),
            transforms : Optional[torch.nn.Module] = None,
            intercepts : Optional[Union[np.ndarray, float]] = -1024.,
            slopes : Optional[Union[np.ndarray, float]] = 1.,
            preprocess : bool = True,
            ):
        self.folder = folder
        if self.folder is not None and \
                not isinstance(self.folder, Path):
            self.folder = Path(self.folder)
        self.ids = ids
        self.images = images
        if self.images is None and (self.folder is None or self.ids is None):
            raise ValueError(f"'images' or 'folder' and 'ids' should be specified on initialization")
        self.labels = labels
        self.patient_ids = patient_ids
        self.z_positions = z_positions
        self.image_size = image_size

        # if self.patient_ids is not None:
        #     self._mapping_patient_id_to_index = {}
        #     for i, p in enumerate(self.patient_ids):
        #         if p not in self._mapping_patient_id_to_index:
        #             self._mapping_patient_id_to_index[p] = len(self._mapping_patient_id_to_index)
        #     self._patient_inds = [self._mapping_patient_id_to_index[p] for p in self.patient_ids]
        #     self._patient_inds = np.array(self._patient_inds, np.int32)

        if self.patient_ids is not None:
            self._set_mappings()

        self.relevant_inds = None
        self.transforms = transforms
        self.intercepts = intercepts
        self.slopes = slopes

        self.preprocess = preprocess

    def _set_mappings(self):
        self._mapping_patient_id_to_index = {}
        for i, p in enumerate(self.patient_ids):
            if p not in self._mapping_patient_id_to_index:
                self._mapping_patient_id_to_index[p] = len(self._mapping_patient_id_to_index)
        self._patient_inds = [self._mapping_patient_id_to_index[p] for p in self.patient_ids]
        self._patient_inds = np.array(self._patient_inds, np.int32)

        self._mapping_patient_to_inds = {}
        for i, patient_id in enumerate(self.patient_ids):
            self._mapping_patient_to_inds.setdefault(patient_id, []).append(i)
        self._mapping_patient_to_inds = {k: np.array(v, np.int32) 
            for k, v in self._mapping_patient_to_inds.items()}
        self._patient_ids_unique = np.array(list(self._mapping_patient_to_inds.keys()))

        if self.z_positions is not None:
            for p in list(self._mapping_patient_to_inds.keys()):
                inds = self._mapping_patient_to_inds[p]
                z = self.z_positions[inds]
                inds_sort = np.argsort(z)
                inds = inds[inds_sort]
                self._mapping_patient_to_inds[p] = inds
        return

    def get_config(self) -> Dict[str, Any]:
        config = {
            "image_size": self.image_size,
        }
        return config

    def __len__(self):
        if self.relevant_inds is None:
            if self.images is None:
                return len(self.ids)
            else:
                return len(self.images)
        else:
            return len(self.relevant_inds)

    def convert_image(self, img: np.ndarray, **kwargs) -> np.ndarray:
        return window_image(img, **kwargs)

    @staticmethod
    def read_dcm(
            filename: Path, 
            image_size : Optional[Tuple[int, int]] = (256, 256),
            ) -> np.ndarray:
        # duplicated function
        dcm = pydicom.dcmread(str(filename))
        img = dcm.pixel_array
        if image_size is not None:
            img = cv2.resize(img, dsize=image_size, interpolation=cv2.INTER_CUBIC)
        return img
    
    def _get_convert_kwargs(self, index : Optional[int] = None) -> Dict[str, Any]:
        convert_kwargs = {}
        if self.intercepts is not None:
            if isinstance(self.intercepts, float) or index is None:
                convert_kwargs["intercept"] = self.intercepts
            else:
                convert_kwargs["intercept"] = self.intercepts[index]
        if self.slopes is not None:
            if isinstance(self.slopes, float) or index is None:
                convert_kwargs["slope"] = self.slopes
            else:
                convert_kwargs["slope"] = self.slopes[index]
        return convert_kwargs

    def load_file(self, name: str) -> np.ndarray:
        filename_npy = self.folder.joinpath(f"ID_{name}.npy")
        filename_dcm = self.folder.joinpath(f"ID_{name}.dcm")
        if filename_npy.exists():
            img = np.load(filename_npy)
        elif filename_dcm.exists():
            try:
                img = get_image_from_dcm(filename_dcm, image_size=self.image_size)
            except:
                logger.error(f"Failed to read dcm file: {filename_dcm}")
                return np.zeros(self.image_size, np.int16)
        else:
            # raise ValueError(f"File is not found: {filename_npy} {filename_dcm}")
            logger.warning(f"File not found: {filename_npy} {filename_dcm}")
            img = np.zeros(self.image_size, np.int16)
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.relevant_inds is not None:
            idx = self.relevant_inds[idx]
        if self.ids is not None:
            id = self.ids[idx]
        else:
            id = idx
        if self.images is None:
            img = self.load_file(id)
        else:
            img = self.images[idx]
        if img.shape[0] != self.image_size[0] or img.shape[1] != self.image_size[1]:
            img = change_image_size(img, image_size=self.image_size)
        
        convert_kwargs = self._get_convert_kwargs(idx)
        img = self.convert_image(img, **convert_kwargs)
        
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        if self.labels is None:
            labels = None
        else:
            labels = torch.from_numpy(self.labels[idx])
        if self.patient_ids is not None:
            patient_index = self._patient_inds[idx]
            patient_id = self.patient_ids[idx]
        else:
            patient_index = idx
            patient_id = idx
        data = {
            "img": img,
            "index": idx,
            "id": id,
            "patient_index": patient_index,
            "patient_id": patient_id,
        }
        if labels is not None:
            data["labels"] = labels
        return data

    def preload_images(self):
        self.images = []
        if self.preprocess:
            if isinstance(self.intercepts, float):
                self.intercepts = 0
            inds_all = []
            for patient_id, inds in self._mapping_patient_ot_inds.items():
                ids = [self.ids[i] for i in inds]
                loaded = load_study(self.folder, names=ids, as_array=True)
                if isinstance(self.intercepts, np.ndarray):
                    self.intercepts[inds] = 0
                self.images += [im for im in loaded]
                inds_all.append(inds)
            inds_all = np.concatenate(inds_all)
            inds = {i: j for j, i in enumerate(inds_all)}
            self.images = [self.images[inds[i]] for i in range(len(inds))]
        else:
            for id in self.ids:
                img = self.load_file(id)
                self.images.append(img)
        return
    
    def _filter_single_image(self, 
            index: int,
            filter: Filter,
            ) -> bool:
        if self.images is None:
            img = self.load_file(self.ids[index])
        else:
            img = self.images[index]
        if img is None or np.all(img == 0):
            return False
        processed = filter.filter(img, **self._get_convert_kwargs(index))
        return processed is not None
    
    def apply_filters(self, filters: List[Filter]):
        n = len(self.ids) if self.ids is not None else len(self.images)
        if self.relevant_inds is None:
            is_relevant = np.ones(n, bool)
        else:
            is_relevant = np.zeros(n, bool)
            is_relevant[self.relevant_inds] = 1
        for filter in Progress(filters, desc="filters", 
                show=(get_logging_level() in ["DEBUG", "INFO"])):
            inds = np.where(is_relevant)[0]
            for index in Progress(inds, desc="images", 
                    show=(get_logging_level() in ["DEBUG", "INFO"]), leave=False):
                is_relevant[index] = is_relevant[index] and self._filter_single_image(index, filter)
        logger.info(f"Filtered out [{np.count_nonzero(~is_relevant)}] images")
        self.relevant_inds = np.where(is_relevant)[0]
        return
    

class DatasetSingleChannel(DatasetBase):
    def __init__(self, 
            *args,
            window_center : float = 40.,
            window_width : float = 80.,
            **kwargs,
            ):
        super(DatasetSingleChannel, self).__init__(*args, **kwargs)
        self.window_center = window_center
        self.window_width = window_width

    def get_config(self) -> Dict[str, Any]:
        config = super(DatasetSingleChannel, self).get_config()
        config.update({
            "window_center": self.window_center,
            "window_width": self.window_width,
        })
        return config

    def convert_image(self, img: np.ndarray, **kwargs) -> np.ndarray:
        img = window_image(img, 
            window_center=self.window_center, 
            window_width=self.window_width
            **kwargs,
        )
        if self.transforms is not None:
            img = torch.from_numpy(img)
            img = self.transforms(img)
        return np.expand_dims(img, 0)


class DatasetMultiChannel(DatasetBase):
    def __init__(self, 
            *args,
            brain_window_center : float = 40,
            brain_window_width : float = 80,
            subdural_window_center : float = 80,
            subdural_window_width : float = 200,
            soft_window_center : float = 40,
            soft_window_width : float = 380,
            **kwargs,
            ):
        super(DatasetMultiChannel, self).__init__(*args, **kwargs)
        self.brain_window_center = brain_window_center
        self.brain_window_width = brain_window_width
        self.subdural_window_center = subdural_window_center
        self.subdural_window_width = subdural_window_width
        self.soft_window_center = soft_window_center
        self.soft_window_width = soft_window_width

    def get_config(self) -> Dict[str, Any]:
        config = super(DatasetMultiChannel, self).get_config()
        config.update({
            "brain_window_center": self.brain_window_center,
            "brain_window_width": self.brain_window_width,
            "subdural_window_center": self.subdural_window_center,
            "subdural_window_width": self.subdural_window_width,
            "soft_window_center": self.soft_window_center,
            "soft_window_width": self.soft_window_width,
        })
        return config

    @property
    def brain_intercept(self) -> float:
        return self.brain_window_center - self.brain_window_width // 2
    @property
    def brain_slope(self) -> float:
        return self.brain_window_width
    @property
    def subdural_intercept(self) -> float:
        return self.subdural_window_center - self.subdural_window_width // 2
    @property
    def subdural_slope(self) -> float:
        return self.subdural_window_width
    @property
    def soft_intercept(self) -> float:
        return self.soft_window_center - self.soft_window_width // 2
    @property
    def soft_slope(self) -> float:
        return self.soft_window_width

    def convert_image(self, img: np.ndarray, **kwargs) -> np.ndarray:
        brain_img = window_image(img, 
            window_center=self.brain_window_center, 
            window_width=self.brain_window_width,
            **kwargs,
        )
        subdural_img = window_image(img, 
            window_center=self.subdural_window_center,
            window_width=self.subdural_window_width,
            **kwargs,
        )
        soft_img = window_image(img,
            window_center=self.soft_window_center,
            window_width=self.soft_window_width,
            **kwargs,
        )
        bsb_img = np.array([brain_img, subdural_img, soft_img])
        # if bsb_img.ndim > 3:
        #     bsb_img = np.transpose(bsb_img, (1, 0, 2, 3))
        if self.transforms is not None:
            bsb_img = torch.from_numpy(bsb_img)
            bsb_img = self.transforms(bsb_img)
        return bsb_img


class Dataset3D(DatasetMultiChannel):
    def __init__(self, 
            patient_ids: np.ndarray,
            z_positions: np.ndarray,
            z_size : int = 64,
            **kwargs,
            ):
        super(Dataset3D, self).__init__(
            patient_ids=patient_ids, 
            z_positions=z_positions,
            **kwargs,
        )
        self.z_size = z_size
        self._set_mappings()

    # def _set_mappings(self):
    #     self._mapping_patient_to_inds = {}
    #     for i, patient_id in enumerate(self.patient_ids):
    #         self._mapping_patient_to_inds.setdefault(patient_id, []).append(i)
    #     self._mapping_patient_to_inds = {k: np.array(v, np.int32) 
    #         for k, v in self._mapping_patient_to_inds.items()}
    #     self._patient_ids_unique = np.array(list(self._mapping_patient_to_inds.keys()))

    #     for p in list(self._mapping_patient_to_inds.keys()):
    #         inds = self._mapping_patient_to_inds[p]
    #         z = self.z_positions[inds]
    #         inds_sort = np.argsort(z)
    #         inds = inds[inds_sort]
    #         self._mapping_patient_to_inds[p] = inds

    def get_config(self) -> Dict[str, Any]:
        config = super(Dataset3D, self).get_config()
        config.update({
            "z_size": self.z_size,
        })
        return config
    
    def __len__(self) -> int:
        return len(self._mapping_patient_to_inds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patient_id = self._patient_ids_unique[idx]
        inds = self._mapping_patient_to_inds[patient_id]
        if self.labels is None:
            ls = None
        else:
            ls = self.labels[inds]

        if len(inds) > self.z_size:
            i_start = (len(inds) - self.z_size) // 2
            if ls is not None:
                ls = ls[i_start : i_start + self.z_size]
            inds = inds[i_start : i_start + self.z_size]

        stacked = np.zeros((3, self.z_size, self.image_size[0], self.image_size[1]), np.float32)
        i_start = (self.z_size - len(inds)) // 2
        masks = np.zeros(self.z_size, bool)
        if ls is None:
            labels_out = None
        else:
            labels_out = np.zeros((self.z_size, ls.shape[1]), bool)
        indexes_masked = []
        ids = []
        for i, index in enumerate(inds):
            if self.ids is None:
                sop_id = index
            else:
                sop_id = self.ids[index]
            if self.images is None:
                img = self.load_file(sop_id)
            else:
                img = self.images[index]
            if img.shape[0] != self.image_size[0] or img.shape[1] != self.image_size[1]:
                img = change_image_size(img, image_size=self.image_size)
                
            convert_kwargs = self._get_convert_kwargs(idx)
            img = self.convert_image(img, **convert_kwargs)
            
            ids.append(sop_id)
            stacked[:, i_start+i] = img
            masks[i_start+i] = 1
            if labels_out is not None:
                labels_out[i_start+i] = ls[i]
            indexes_masked.append(i_start+i)
        indexes_masked = np.array(indexes_masked, np.int32)

        data = {
            "img": stacked,
            "id": ids,
            "index": inds,
            "index_masked": indexes_masked,
            "patient_index": idx,
            "patient_id": patient_id,
        }
        if labels_out is not None:
            data["labels"] = labels_out
        return data
    
    def apply_filters(self, filters: List[Filter], *args, **kwargs):
        super().apply_filters(filters, *args, **kwargs)
        self._mapping_patient_to_inds = {k: v[
            np.isin(v, self.relevant_inds, assume_unique=True)] 
            for k, v in self._mapping_patient_to_inds.items()}
        return

    
def collate_fn_3d(
        data: List[Dict[str, Any]],
        ) -> Dict[str, Any]:
    if "img" in data[0]:
        images = [d["img"] for d in data]
    else:
        images = None
    if "embeds" in data[0]:
        embeds = [d["embeds"] for d in data]
    else:
        embeds = None
    if "preds" in data[0]:
        preds = [d["preds"] for d in data]
    else:
        preds = None
    labels = [d["labels"] for d in data] if "labels" in data[0] else None
    inds = [d["index"] for d in data]
    if "index_masked" in data[0]:
        inds_masked = [d["index_masked"] for d in data]
        inds_masked = list(inds_masked)
    else:
        inds_masked = None
    patient_indexes = [d["patient_index"] for d in data]
    patient_ids = [d["patient_id"] for d in data]
    ids = [d["id"] for d in data]
    
    if "slice_indexes" in data[0]:
        slice_indexes = [d["slice_indexes"] for d in data]
        slice_indexes = np.array(slice_indexes)
    else:
        slice_indexes = None

    if images is not None:
        if isinstance(images[0], np.ndarray):
            images = np.array(images)
            images = torch.from_numpy(images)
        else:
            images = torch.stack(images)
    if embeds is not None:
        if isinstance(embeds[0], np.ndarray):
            embeds = np.array(embeds)
            embeds = torch.from_numpy(embeds)
        else:
            embeds = torch.stack(embeds)
    if preds is not None:
        if isinstance(preds[0], np.ndarray):
            preds = np.array(preds)
            preds = torch.from_numpy(preds)
        else:
            preds = torch.stack(preds)

    if labels is not None:
        if labels[0].ndim <= 2:
            labels_m = [l[i] for l, i in zip(labels, inds_masked)]
            labels_m = np.concatenate(labels_m)
            labels_m = torch.from_numpy(labels_m)
            labels = np.array(labels)
            labels = torch.from_numpy(labels)
            labels_pat = labels.max(1).values
        else:
            labels_m = None
            labels_pat = None
            labels = np.array(labels)
            labels = torch.from_numpy(labels)

    inds = list(inds)

    data_out = {
        "id": ids,
        "index": inds,
        "patient_index": patient_indexes,
        "patient_id": patient_ids,
    }
    if inds_masked is not None:
        data_out["index_masked"] = inds_masked
    if images is not None:
        data_out["img"] = images
    if embeds is not None:
        data_out["embeds"] = embeds
    if preds is not None:
        data_out["preds"] = preds
    if labels is not None:
        if labels_m is not None and labels_pat is not None:
            data_out.update({
                "labels_m": labels_m,
                "labels_pat": labels_pat,
            })
        else:
            data_out["labels"] = labels
    if slice_indexes is not None:
        data_out["slice_indexes"] = slice_indexes
    return data_out


class DatasetEmbeds(Dataset):
    def __init__(self,
            preds: np.ndarray,
            embeds: np.ndarray,
            patient_ids: np.ndarray,
            z_positions: np.ndarray,
            labels : Optional[np.ndarray] = None,
            ids : Optional[np.ndarray] = None,
            z_size : int = 64,   
            ):
        super().__init__()
        self.preds = preds
        self.embeds = embeds
        self.patient_ids = patient_ids
        self.z_positions = z_positions
        self.labels = labels
        self.ids = ids
        self.z_size = z_size
        self._set_mappings()
        
    def _set_mappings(self):
        self._mapping_patient_to_inds = {}
        for i, patient_id in enumerate(self.patient_ids):
            self._mapping_patient_to_inds.setdefault(patient_id, []).append(i)
        self._mapping_patient_to_inds = {k: np.array(v, np.int32) 
            for k, v in self._mapping_patient_to_inds.items()}
        self._patient_ids_unique = np.array(list(self._mapping_patient_to_inds.keys()))

        for p in list(self._mapping_patient_to_inds.keys()):
            inds = self._mapping_patient_to_inds[p]
            z = self.z_positions[inds]
            inds_sort = np.argsort(z)
            inds = inds[inds_sort]
            self._mapping_patient_to_inds[p] = inds
    
    @property
    def embed_dim(self) -> int:
        return self.embeds.shape[1]
    
    def __len__(self) -> int:
        return len(self._mapping_patient_to_inds)
    
    def get_config(self) -> Dict[str, Any]:
        config = {
            "z_size": self.z_size,
        }
        return config
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patient_id = self._patient_ids_unique[idx]
        inds = self._mapping_patient_to_inds[patient_id]
        if self.labels is None:
            ls = None
        else:
            ls = self.labels[inds]
            
        if len(inds) > self.z_size:
            i_start = (len(inds) - self.z_size) // 2
            if ls is not None:
                ls = ls[i_start : i_start + self.z_size]
            inds = inds[i_start : i_start + self.z_size]
            
        stacked = np.zeros((self.z_size, self.embed_dim), np.float32)
        i_start = (self.z_size - len(inds)) // 2
        masks = np.zeros(self.z_size, bool)
        if ls is None:
            labels_out = None
        else:
            labels_out = np.zeros((self.z_size, ls.shape[1]), bool)
        indexes_masked = []
        ids = []
        stacked_preds = np.zeros((self.z_size, self.preds.shape[1]), np.float32)
        for i, index in enumerate(inds):
            if self.ids is None:
                sop_id = index
            else:
                sop_id = self.ids[index]
            pred = self.preds[index]
            embed = self.embeds[index]
            ids.append(sop_id)
            stacked[i_start+i] = embed
            stacked_preds[i_start+i] = pred
            masks[i_start+i] = 1
            if labels_out is not None:
                labels_out[i_start+i] = ls[i]
            indexes_masked.append(i_start+i)
        indexes_masked = np.array(indexes_masked, np.int32)

        data = {
            "embeds": stacked,
            "id": ids,
            "index": inds,
            "index_masked": indexes_masked,
            "patient_index": idx,
            "patient_id": patient_id,
            "preds": stacked_preds,
        }
        if labels_out is not None:
            data["labels"] = labels_out
        return data
    
    
class DatasetBHSD(DatasetMultiChannel):
    def __init__(self, 
            folder : Optional[Path] = None,
            ids : Optional[List[str]] = None,
            images : Optional[List[np.ndarray]] = None,
            folder_labels : Optional[Path] = None,
            labels : Optional[List[np.ndarray]] = None,
            image_size : Optional[Tuple[int, int]] = (256, 256),
            z_size : int = 64,
            brain_window_center : float = 40,
            brain_window_width : float = 80,
            subdural_window_center : float = 80,
            subdural_window_width : float = 200,
            soft_window_center : float = 40,
            soft_window_width : float = 380,
            n_classes : Optional[int] = 6,
            is_3d : bool = True,
            preload : bool = False,
            transforms : Optional[torch.nn.Module] = None,
            ):
        self.folder = folder
        if self.folder is not None and not isinstance(self.folder, Path):
            self.folder = Path(self.folder)
        self.ids = ids
        self.images = images
        if self.images is None and (self.folder is None or self.ids is None):
            raise ValueError(f"'images' or 'folder' and 'ids' should be specified on initialization")
        self.labels = labels
        self.folder_labels = folder_labels
        if self.folder_labels is not None and isinstance(self.folder_labels, Path):
            self.folder_labels = Path(self.folder_labels)
        self.image_size = image_size
        self.z_size = z_size
        
        self.brain_window_center = brain_window_center
        self.brain_window_width = brain_window_width
        self.subdural_window_center = subdural_window_center
        self.subdural_window_width = subdural_window_width
        self.soft_window_center = soft_window_center
        self.soft_window_width = soft_window_width

        self.n_classes = n_classes
        self.is_3d = is_3d
        self.patient_indexes = None
        self.slice_indexes = None
        
        self.transforms = transforms

        if preload:
            self.preload()
        
    def get_config(self) -> Dict[str, Any]:
        config = {
            "image_size": self.image_size, 
            "z_size": self.z_size,
            "brain_window_center": self.brain_window_center,
            "brain_window_width": self.brain_window_width,
            "subdural_window_center": self.subdural_window_center,
            "subdural_window_width": self.subdural_window_width,
            "soft_window_center": self.soft_window_center,
            "soft_window_width": self.soft_window_width,
            "n_classes": self.n_classes,
            "is_3d": self.is_3d,
        }
        return config
            
    def __len__(self) -> int:
        if self.is_3d:
            return len(self.images) if self.images is not None else len(self.ids)
        else:
            return len(self.images)

    def load_file(self, idx: int, label : bool = False) -> np.ndarray:
        if label:
            folder = self.folder_labels
        else:
            folder = self.folder
        filename = folder.joinpath(f"{self.ids[idx]}")
        if filename.exists():
            return read_sitk_image(filename, as_array=True)
        elif filename.with_suffix(".nii").exists():
            return read_sitk_image(filename.with_suffix(".nii"), as_array=True)
        elif filename.with_suffix(".nii.gz").exists():
            return read_sitk_image(filename.with_suffix(".nii.gz"), as_array=True)
        elif filename.with_suffix(".npy").exists():
            return np.load(filename.with_suffix(".npy"))
        else:
            raise ValueError(f"No file {self.ids[idx]} in folder {folder}")

    def preload(self):
        n = len(self.ids)
        self.images = []
        self.labels = []
        for i in Progress(range(n), 
                show=(get_logging_level() in ["DEBUG", "INFO"]),
                leave=False,
                desc="Preloading dataset images",
                ):
            image = self.load_file(i, label=False)
            label = self.load_file(i, label=True)
            self.images.append(image)
            self.labels.append(label)
        if not self.is_3d:
            self.patient_indexes = np.concatenate([[i]*len(images) for images in self.images])
            self.slice_indexes = np.concatenate([np.arange(len(images)) for images in self.images])
            self.images = [image for images in self.images for image in images]
            self.labels = [label for labels in self.labels for label in labels]
        return
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.is_3d:
            if self.images is None:
                image = self.load_file(idx, label=False)
            else:
                image = self.images[idx]
            if image.shape[1] != self.image_size[0] or image.shape[2] != self.image_size[1]:
                image = np.asarray([change_image_size(img, image_size=self.image_size) for img in image])
            image = self.convert_image(image, intercept=0, slope=1)
            if self.labels is not None:
                label = self.labels[idx]
            elif self.folder_labels is not None:
                label = self.load_file(idx, label=True)
            else:
                label = None
            if label is not None and (
                    label.shape[1] != self.image_size[0] or label.shape[2] != self.image_size[1]):
                label = np.asarray([change_image_size(img, image_size=self.image_size, 
                    interpolation="nearest").astype(np.uint8) for img in label])
            if self.n_classes is None and label is not None:
                label = label > 0
        
            if image.shape[0] > self.z_size:
                i_start = (image.shape[1] - self.z_size) // 2
                image = image[:, i_start : i_start + self.z_size]
                if label is not None:
                    label = label[i_start : i_start + self.z_size]
                slice_indexes = np.arange(i_start, i_start + self.z_size)
            elif image.shape[0] < self.z_size:
                i_start = (self.z_size - image.shape[1]) // 2
                slice_indexes = np.full(self.z_size, -1, np.int32)
                slice_indexes[i_start:i_start+image.shape[1]] = np.arange(image.shape[1])
                im_1 = np.zeros((image.shape[0], i_start, image.shape[2], image.shape[3]), dtype=image.dtype)
                im_2 = np.zeros((image.shape[0], self.z_size - i_start - image.shape[1], 
                    image.shape[2], image.shape[3]), dtype=image.dtype)
                image = np.concatenate([im_1, image, im_2], 1)
                if label is not None:
                    l_1 = np.zeros((i_start,) + tuple(label.shape[1:]), dtype=label.dtype)
                    l_2 = np.zeros((self.z_size - i_start - label.shape[0],) + \
                        tuple(label.shape[1:]), dtype=label.dtype)
                    label = np.concatenate([l_1, label, l_2])
            else:
                slice_indexes = np.arange(self.z_size)

            id = self.ids[idx] if self.ids is not None else idx
            patient_index = idx
            patient_id = self.ids if self.ids is not None else idx
        else:
            image = self.images[idx]
            if image.shape[0] != self.image_size[0] or image.shape[1] != self.image_size[1]:
                image = change_image_size(image, image_size=self.image_size)
            image = self.convert_image(image, intercept=0, slope=1)
            if self.labels is None:
                label = None
            else:
                label = self.labels[idx]
            if label is not None and (
                    label.shape[1] != self.image_size[0] or label.shape[2] != self.image_size[1]):
                label = change_image_size(label, image_size=self.image_size, interpolation="nearest").astype(np.uint8)
            if self.n_classes is None and label is not None:
                label = label > 0

            id = idx
            patient_index = self.patient_indexes[idx]
            patient_id = patient_index
            slice_indexes = self.slice_indexes[idx]

        data = {
            "img": image,
            "index": idx,
            "id": id,
            "patient_index": patient_index,
            "patient_id": patient_id,
            "slice_indexes": slice_indexes,
        }
        if label is not None:
            data["labels"] = label
        return data
            

def init_dataset(
        mode: Literal["2d_single_channel", "2d_multichannel", "3d", "embeds", "segm"] = "2d_single_channel",
        **kwargs,
        ) -> Union[DatasetBase, DatasetEmbeds]:
    if mode == "2d_single_channel":
        dataset_class = DatasetSingleChannel
    elif mode == "2d_multichannel":
        dataset_class = DatasetMultiChannel
        kwargs["transforms"] = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
        )
    elif mode == "3d":
        dataset_class = Dataset3D
    elif mode == "embeds":
        dataset_class = DatasetEmbeds
    elif mode == "segm":
        dataset_class = DatasetBHSD
        kwargs["transforms"] = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )
    else:
        raise ValueError(f"Unknown mode of dataset initialization: {mode}")
    dataset = dataset_class(**kwargs)
    return dataset


def get_dataset_for_training(
        data: Union[Path, dict],
        folder_images : Optional[Path] = None,
        mode : Literal["2d_single_channel", "2d_multichannel", "3d", "embeds", "segm"] = "2d_single_channel",
        split_set : Literal["train", "val", "test"] = "val",
        **kwargs,
        ) -> Dataset:
    
    if not isinstance(data, dict):
        data = pkl.load(open(data, "rb"))
    
    d = data[f"{split_set}_data"]
    if mode in ["2d_single_channel", "2d_multichannel", "3d"]:
        args = {
            "ids": d.sop_id.values,
            "folder": folder_images,
            "labels": data[f"{split_set}_labels"],
            # "patient_ids": data[f"{split_set}_data"].patient_id.values,
            "patient_ids": d.study_id.values,
            "intercepts": d.intercept.values,
            "slopes": d.slope.values,
            "z_positions": d.position_z.values,
        }
        # if mode == "3d":
        #     args["z_positions"] = d.position_z.values
    elif mode == "embeds":
        args = {
            "ids": d.sop_id.values,
            "labels": data[f"{split_set}_labels"],
            "patient_ids": d.study_id.values,
            "z_positions": d.position_z.values,
        }
    elif mode == "segm":
        args = {
            "folder": folder_images,
            "ids": d["ids"],
        }
    else:
        raise ValueError(f"Unknown mode for dataset initialization: {mode}")
    return init_dataset(mode=mode, **args, **kwargs)


def get_dataloader(
        data: Union[Path, dict],
        folder_images : Optional[Path] = None,
        dataset_mode : Literal["2d_single_channel", "2d_multichannel", "3d", "embeds", "segm"] = "2d_single_channel",
        mode : Literal["train", "val", "test", "inference"] = "val",
        batch_size : int = 4, 
        num_workers : int = 8,
        filters : List[str] = ["brain_present"],
        **kwargs, 
        ) -> DataLoader:
    dataset = get_dataset_for_training(data=data, 
        folder_images=folder_images, mode=dataset_mode, **kwargs)
    if len(filters) > 0 and isinstance(dataset, DatasetBase):
        filters = [get_filter_by_name(f) for f in filters]
        dataset.apply_filters(filters)
    add_args = {}
    if dataset_mode in ["3d", "embeds"] \
            or (dataset_mode == "segm" and kwargs.get("is_3d", False)):
        add_args["collate_fn"] = collate_fn_3d
    if mode == "train":
        add_args.update({
            "shuffle": True,
            "drop_last": True,
        })
    dataloader = DataLoader(dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True,
        **add_args,
    )
    return dataloader