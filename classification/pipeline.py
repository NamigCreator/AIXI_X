from pathlib import Path
from typing import Optional, Literal, List, Union, Dict, Tuple
import json
import torch
import numpy as np
import pandas as pd
import pydicom

from torch.utils.data import DataLoader

from .data.dataset import init_dataset, collate_fn_3d
from .model.model import get_model
from .model.pl import PLModel, PLModel3D, PLModelSeq, PLModelSegm
from .utils import Progress, init_logger, get_logging_level, run_multiprocess
from .data.misc import get_metadata_from_dcm
from .data.filter import get_filter_by_name


_folder_current = Path(__file__).parent
_folder = _folder_current.parent
_folder_models = _folder.joinpath("res", "models")
_folder_checkpoints = _folder.joinpath("res", "checkpoints")
_folder_models.mkdir(exist_ok=True)

logger = init_logger(__name__)


class ClassificationPipeline:
    def __init__(self,
            model_name: str,
            seq_model_name : Optional[str] = None,
            segm_model_name : Optional[str] = None,
            checkpoint_name : str = "best",
            batch_size : Optional[str] = None,
            device : str = "cuda",
            aggregate_mode : Literal["mean", "max"] = "max",
            threads : int = 8,
            ):
        self.model_name = model_name
        self.batch_size = batch_size

        filename_params = _folder_models.joinpath(model_name, "params.json")
        filename_checkpoint = _folder_checkpoints.joinpath(model_name, f"{checkpoint_name}.ckpt")
        self.model_params = json.load(open(filename_params))
        self.mode = self.model_params["mode"]
        logger.debug("Initializing model")
        self.model = get_model(
            model_type=self.mode, 
            filename_checkpoint=filename_checkpoint,
            model_name=self.model_params.get("model_arch", "efficientnet"),
        )
        self.device = device
        self.model.eval()
        self.model.to(self.device)
        if self.batch_size is None:
            self.batch_size = self.model_params["batch_size"]

        if self.mode.startswith("2d"):
            plmodel = PLModel
        elif self.mode.startswith("3d"):
            plmodel = PLModel3D
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        logger.debug("Initializing PL model")
        self.plmodel = plmodel(self.model, 
            aggregate_mode=aggregate_mode, 
            target_mode=self.model_params["target_mode"],
        )
        self.plmodel.to(self.device)
        
        self.seq_model_name = seq_model_name
        if seq_model_name is None:
            self.seq_model = None
            self.plmodel_seq = None
        else:
            filename_params_seq = _folder_models.joinpath(seq_model_name, "params.json")
            filename_checkpoint_seq = _folder_checkpoints.joinpath(seq_model_name, "best.ckpt")
            self.seq_model_params = json.load(open(filename_params_seq))
            logger.debug("Initializing Sequence model")
            self.seq_model = get_model(
                model_type="seq",
                filename_checkpoint=filename_checkpoint_seq,
                **self.seq_model_params["model_params"],
            )
            self.seq_model.eval()
            self.seq_model.to(self.device)
            logger.debug("Initializing PL model")
            self.plmodel_seq = PLModelSeq(self.seq_model, 
                aggregate_mode=aggregate_mode,
                target_mode=self.seq_model_params["target_mode"],
            )
            self.plmodel_seq.to(self.device)

        self.segm_model_name = segm_model_name
        if segm_model_name is None:
            self.segm_model = None
            self.plmodel_segm = None
        else:
            filename_params_segm = _folder_models.joinpath(segm_model_name, "params.json")
            filename_checkpoint_segm = _folder_checkpoints.joinpath(segm_model_name, "best.ckpt")
            self.segm_model_params = json.load(open(filename_params_segm))
            logger.debug("Initializing Segmentation model")
            segm_n_classes = self.segm_model_params["n_classes"]
            segm_target_mode = self.segm_model_params["target_mode"]
            segm_is_3d = self.segm_model_params["is_3d"]
            self.segm_model = get_model(
                model_type="segm", 
                out_dim=(1 if segm_n_classes is None else segm_n_classes),
                mode=("3d" if segm_is_3d else "2d"),
                filename_checkpoint=filename_checkpoint_segm,
            )
            self.segm_model.eval()
            self.segm_model.to(self.device)
            self.plmodel_segm = PLModelSegm(
                model=self.segm_model,
                n_classes=segm_n_classes,
                target_mode=segm_target_mode,
            )
            self.plmodel_segm.to(self.device)

        self.threads = threads
        self.filters = self.model_params.get("filters", [])
        self.filters = [get_filter_by_name(f) for f in self.filters]

        logger.debug(f"Initialized model: {model_name}")


    @staticmethod
    def extract_data_from_folder(
            folder: Path, 
            ids : Optional[Union[List[str], Path]] = None,
            threads : int = 8,
            ) -> pd.DataFrame:
        if not isinstance(folder, Path):
            folder = Path(folder)
        if ids is None:
            ids = [f.stem for f in folder.iterdir() if f.suffix == ".dcm"]
            ids = [f[3:] if f.startswith("ID_") else f for f in ids]
            ids = sorted(ids)
        elif isinstance(ids, Path):
            ids = open(ids).read().strip().split("\n")
        def _get_args():
            for i in ids:
                filename = folder.joinpath(f"{i}.dcm")
                if not filename.exists():
                    filename = folder.joinpath(f"ID_{i}.dcm")
                yield filename
        dcm_data = run_multiprocess(
            get_metadata_from_dcm, 
            _get_args(),
            total=len(ids), 
            threads=threads, 
            progress=Progress(
                total=len(ids), 
                desc="Loading dcm meta data", 
                show=(get_logging_level() in ["DEBUG", "INFO"]),
            ),
        )
        dcm_data = pd.DataFrame(dcm_data)
        dcm_data["id"] = ids
        return dcm_data


    def predict(self,
            images : Optional[Union[List[np.ndarray], List[pydicom.FileDataset]]] = None,
            folder : Optional[Path] = None,
            ids : Optional[List[str]] = None,
            patient_ids : Optional[List[str]] = None,
            z_positions : Optional[np.ndarray] = None,
            intercepts : Optional[Union[np.ndarray, float]] = None,
            slopes : Optional[Union[np.ndarray, float]] = None,
            preload : bool = False,
            get_full_data : bool = False,
            get_embeddings : bool = False,
            ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
        
        if ((z_positions is None and self.mode.startswith("3d")) \
                or patient_ids is None) and folder is not None:
            dcm_data = self.extract_data_from_folder(folder, ids=ids, threads=self.threads)
            if patient_ids is None:
                # patient_ids = dcm_data.patient_id.values
                patient_ids = dcm_data.study_id.values
            if z_positions is None:
                z_positions = dcm_data.position_z.values
            if intercepts is None:
                intercepts = dcm_data.intercept.values
            if slopes is None:
                slopes = dcm_data.slope.values
            if ids is None or isinstance(ids, Path):
                ids = dcm_data.id.values
        if images is not None and all(isinstance(img, pydicom.FileDataset) for img in images):
            if z_positions is None:
                z_positions = [float(img.ImagePositionPatient[0]) for img in images]
                z_positions = np.array(z_positions, np.float32)
            if patient_ids is not None:
                patient_ids = [float(img.StudyInstanceUID[3:]) for img in images]
            if intercepts is None:
                intercepts = [(float(img.RescaleIntercept)) for img in images]
                intercepts = np.array(intercepts, np.float32)
            if slopes is None:
                slopes = [(float(img.RescaleSlope)) for img in images]
                slopes = np.array(slopes, np.float32)
            images = [img.pixel_array for img in images]
        n_images = len(images) if images is not None else len(ids)
        if patient_ids is None:
            patient_ids = [0]*n_images
        dataset_args = {
            "folder": folder,
            "ids": ids,
            "images": images,
            "patient_ids": patient_ids,
            "intercepts": intercepts,
            "slopes": slopes,
            "z_positions": z_positions,
        }
        # if self.mode.startswith("3d"):
        #     dataset_args["z_positions"] = z_positions
        dataset = init_dataset(mode=self.mode, **dataset_args)
        if preload:
            dataset.preload_images()
        if len(self.filters) > 0:
            dataset.apply_filters(self.filters)

        dataloader_args = {
            "batch_size": self.batch_size,
            "num_workers": self.threads,
        }
        if self.mode.startswith("3d"):
            dataloader_args["collate_fn"] = collate_fn_3d
        dataloader = DataLoader(dataset, **dataloader_args)

        with torch.no_grad():
            preds_all = []
            preds_pat_all = []
            sop_inds_all = []
            sop_ids_all = []
            patient_inds_all = []
            patient_ids_all = []
            if get_embeddings or self.seq_model is not None:
                embeds = []
            else:
                embeds = None
            if self.segm_model is not None:
                segm_masks = []
            else:
                segm_masks = None
            for batch in Progress(dataloader, desc="Prediction",
                    show=(get_logging_level() in ["DEBUG", "INFO"])):
                
                sop_index = batch["index"]
                sop_id = batch["id"]
                p_index = batch["patient_index"]
                p_id = batch["patient_id"]

                if self.mode.startswith("2d"):
                    images = batch["img"]
                    if embeds is None:
                        pred = self.plmodel.forward(images.to(self.device))
                    else:
                        pred, embed = self.plmodel.forward(
                            images.to(self.device), get_embeddings=True)
                    preds_all.append(pred.cpu())
                elif self.mode.startswith("3d"):
                    images = batch["img"]
                    indexes_masked = batch["index_masked"]
                    pred, pred_patient = self.plmodel.forward(images.to(self.device), indexes_masked)
                    preds_all.append(pred.cpu())
                    preds_pat_all.append(pred_patient.cpu())
                    p_index = np.concatenate([[p]*len(i) for i, p in zip(sop_index, p_index)])
                    p_id = np.concatenate([[p]*len(i) for i, p in zip(sop_index, p_id)])
                    sop_index = np.concatenate(sop_index)
                    sop_id = np.concatenate(sop_id)

                if self.segm_model is not None:
                    images = batch["img"]
                    pred = self.plmodel_segm.forward(images.to(self.device))
                    pred = self.plmodel_segm.activation(pred)
                    segm_masks.append(pred.cpu())

                sop_inds_all.append(sop_index)
                sop_ids_all.append(sop_id)
                patient_inds_all.append(p_index)
                patient_ids_all.append(p_id)
                if embeds is not None:
                    embeds.append(embed.cpu())

            sop_inds_all = np.concatenate(sop_inds_all)
            sop_ids_all = np.concatenate(sop_ids_all)
            patient_inds_all = np.concatenate(patient_inds_all)
            patient_ids_all = np.concatenate(patient_ids_all)

            preds_all = torch.cat(preds_all)
            if len(preds_pat_all) > 0:
                preds_pat_all = torch.cat(preds_pat_all)
            else:
                preds_pat_all = self.plmodel.aggregate_preds_for_patients(
                    preds_all, patient_inds_all, 
                    mode=self.plmodel.aggregate_mode,
                    target_mode=self.plmodel.target_mode,
                )
            preds_all = self.plmodel.activation(preds_all)
            preds_pat_all = self.plmodel.activation(preds_pat_all)
            if embeds is not None:
                embeds = torch.cat(embeds)

            if segm_masks is not None:
                segm_masks = torch.cat(segm_masks)

            preds_all = preds_all.numpy()
            preds_pat_all = preds_pat_all.numpy()
            if embeds is not None:
                embeds = embeds.numpy()
            if segm_masks is not None:
                segm_masks = segm_masks.numpy()

        patient_ids_unique, ind = np.unique(patient_ids_all, return_index=True)
        patient_ids_unique = patient_ids_unique[np.argsort(ind)]

        preds_final = np.zeros((n_images, preds_all.shape[1]), np.float32)
        preds_final[:, -1] = 1
        preds_final[sop_inds_all] = preds_all
        if embeds is not None:
            embeds_final = np.zeros((n_images, embeds.shape[1]), np.float32)
            embeds_final[sop_inds_all] = embeds
        if segm_masks is not None:
            segm_masks_final = np.zeros((n_images,)+tuple(segm_masks.shape[1:]), np.float32)
            segm_masks_final[sop_inds_all] = segm_masks
            segm_masks = None
        else:
            segm_masks_final = None
        
        if ids is None:
            sop_ids_final = np.arange(len(images))
        else:
            sop_ids_final = ids
            
        if self.seq_model is not None:
            dataset_seq = init_dataset(
                mode="embeds",
                ids=ids,
                patient_ids=patient_ids,
                z_positions=z_positions,
                embeds=embeds_final,
                preds=preds_final,
            )
            dataloader_seq = DataLoader(dataset_seq,
                batch_size=self.batch_size, 
                num_workers=self.threads,
                collate_fn=collate_fn_3d,
            )
            preds_all_seq = []
            preds_pat_all_seq = []
            sop_inds_all = []
            sop_ids_all = []
            patient_inds_all = []
            patient_ids_all = []
            with torch.no_grad():
                for batch in Progress(dataloader_seq, desc="Sequential model preds",
                        show=(get_logging_level() in ["DEBUG", "INFO"])):
                    sop_index = batch["index"]
                    sop_id = batch["id"]
                    p_index = batch["patient_index"]
                    p_id = batch["patient_id"]
                    pred, _, pred_patient = self.plmodel_seq.forward(
                        embeds=batch["embeds"].to(self.device),
                        preds=batch["preds"].to(self.device),
                        inds=batch["index_masked"],
                    )
                    p_index = np.concatenate([[p]*len(i) for i, p in zip(sop_index, p_index)])
                    p_id = np.concatenate([[p]*len(i) for i, p in zip(sop_index, p_id)])
                    sop_index = np.concatenate(sop_index)
                    sop_id = np.concatenate(sop_id)
                    preds_all_seq.append(pred.cpu())
                    preds_pat_all_seq.append(pred_patient.cpu())
                    sop_inds_all.append(sop_index)
                    sop_ids_all.append(sop_id)
                    patient_inds_all.append(p_index)
                    patient_ids_all.append(p_id)
                    
            sop_inds_all = np.concatenate(sop_inds_all)
            sop_ids_all = np.concatenate(sop_ids_all)
            patient_inds_all = np.concatenate(patient_inds_all)
            patient_ids_all = np.concatenate(patient_ids_all)
            
            preds_all_seq = torch.cat(preds_all_seq)
            preds_pat_all_seq = torch.cat(preds_pat_all_seq)
            preds_all_seq = self.plmodel_seq.activation(preds_all_seq)
            preds_pat_all_seq = self.plmodel_seq.activation(preds_pat_all_seq)
            preds_all_seq = preds_all_seq.numpy()
            preds_pat_all_seq = preds_pat_all_seq.numpy()
            
            patient_ids_unique, ind = np.unique(patient_ids_all, return_index=True)
            patient_ids_unique = patient_ids_unique[np.argsort(ind)]
            
            preds_initial = preds_final
            preds_pat_all_initial = preds_pat_all
            
            preds_pat_all = preds_pat_all_seq
            
            preds_final = np.zeros((n_images, preds_all_seq.shape[1]), np.float32)
            preds_final[:, -1] = 1
            preds_final[sop_inds_all] = preds_all_seq
            
        if get_full_data:
            result = {
                "ids": sop_ids_final,
                "preds": preds_final,
                "patient_ids": patient_ids_unique,
                "patient_preds": preds_pat_all,
            }
            if get_embeddings:
                result["embeds"] = embeds_final
            if self.seq_model is not None:
                result.update({
                    "preds_initial": preds_initial,
                    "patient_preds_initial": preds_pat_all_initial,
                })
            if self.segm_model is not None:
                result.update({
                    "segm_masks": segm_masks_final,
                })
            return result
        else:
            if segm_masks_final is None:
                return preds_final
            else:
                return preds_final, segm_masks_final