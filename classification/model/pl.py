import numpy as np
from pathlib import Path
from typing import Literal, Optional, List, Tuple, Union, Dict, Any
import gc

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torchvision

from ..train.metrics import MultMetrics, DiceLoss, MetricsSegmentation
from ..data.misc import make_image_from_slice_channel, make_image_from_classes


class ExtractStateDictCallback(pl.Callback):
    def __init__(self, 
            folder_checkpoints: Path, 
            folder_model: Path, 
            filename: Path, 
            **kwargs,
            ):
        super(ExtractStateDictCallback, self).__init__(**kwargs)
        self.folder_checkpoints = folder_checkpoints
        self.folder_model = folder_model
        self.filename = filename

    def on_validation_epoch_end(self, *args, **kwargs):
        filename_checkpoint = self.folder_checkpoints.joinpath(f"{self.filename}.ckpt")
        if filename_checkpoint.is_file():
            state_dict = {k[6:]: v for k, v in torch.load(filename_checkpoint)["state_dict"].items()}
            torch.save(state_dict, self.folder_model.joinpath(f"{self.filename}.pth"))
        return


class PLModel(pl.LightningModule):
    def __init__(self,
            model: torch.nn.Module,
            optimizer_lr : float = 1e-3,
            optimizer_weight_decay : float = 0.,
            scheduler_factor : float = 0.1,
            scheduler_patience : int = 10,
            target_mode : Literal["multilabel", "multiclass", "binary"] = "multiclass",
            monitor_metric : str = "val_ce",
            monitor_mode : Literal["max", "min"] = "min",
            folder_model : Optional[Path] = None,
            folder_checkpoints : Optional[Path] = None,
            class_weights : Optional[np.ndarray] = None,
            early_stopping_patience : Optional[int] = None,
            aggregate_mode : Literal["mean", "max"] = "max",
            ):
        super(PLModel, self).__init__()
        self.model = model
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.target_mode = target_mode
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.folder_model = folder_model
        if self.folder_model is not None \
                and not isinstance(self.folder_model, Path):
            self.folder_model = Path(self.folder_model)
        self.folder_checkpoints = folder_checkpoints
        if self.folder_checkpoints is not None \
                and not isinstance(self.folder_checkpoints, Path):
            self.folder_checkpoints = Path(self.folder_checkpoints)
        self.class_weights = class_weights
        if self.class_weights is not None \
                and not isinstance(self.class_weights, torch.Tensor):
            self.class_weights = torch.from_numpy(self.class_weights).to(self.device)
        self.early_stopping_patience = early_stopping_patience

        if self.target_mode == "multilabel":
            self.loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")
        elif self.target_mode == "multiclass":
            self.loss_function = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        elif self.target_mode == "binary":
            self.loss_function = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown target mode: {self.target_mode}")
        self.metrics = MultMetrics(
            class_weights=self.class_weights, 
            device=self.device,
            task=self.target_mode,
        )
        self.aggregate_mode = aggregate_mode

    def forward(self, 
            x: torch.Tensor, 
            get_embeddings : bool = False,
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if get_embeddings:
            embeds = self.model[0](x)
            result = self.model[1](embeds)
            return result, embeds
        else:
            return self.model(x)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_mode in ["multilabel", "binary"]:
            x = torch.sigmoid(x)
        elif self.target_mode == "multiclass":
            x = torch.softmax(x, 1)
        else:
            raise ValueError(f"Unkown target mode: {self.target_mode}")
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=self.optimizer_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.optimizer_weight_decay)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.monitor_mode,
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            ),
            "monitor": self.monitor_metric,
        }
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        results = []
        if self.early_stopping_patience is not None:
            early_stop = EarlyStopping(
                monitor=self.monitor_metric,
                mode=self.monitor_mode,
                patience=self.early_stopping_patience,
            )
            results.append(early_stop)
        if self.folder_checkpoints is not None:
            self.folder_checkpoints.mkdir(exist_ok=True)
            checkpoint = ModelCheckpoint(
                dirpath=self.folder_checkpoints,
                monitor=self.monitor_metric,
                filename="best",
                mode=self.monitor_mode,
                verbose=0,
                enable_version_counter=False,
            )
            checkpoint_last = ModelCheckpoint(
                dirpath=self.folder_checkpoints,
                filename="last",
                verbose=0,
                save_last=True,
                enable_version_counter=False,
            )
            model = ExtractStateDictCallback(self.folder_checkpoints, self.folder_model, "best")
            results += [checkpoint, checkpoint_last, model]
        if self.folder_model is not None:
            self.folder_model.mkdir(exist_ok=True)
        return results

    def shared_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        images = batch["img"]
        labels = batch.get("labels")
        patient_indexes = batch["patient_index"]
        pred = self.forward(images)

        if labels is not None:
            loss = self.loss_function(pred, labels.to(pred.dtype))
            if self.target_mode == "multilabel":
                if self.class_weights is not None:
                    loss = self.class_weights.to(loss.device) * loss
                    loss = torch.sum(loss)
                else:
                    loss = torch.mean(loss)
        else:
            loss = None
            
        pred = self.activation(pred)
        d = {
            "pred": pred.detach().cpu(),
            "patient_inds": patient_indexes.cpu().numpy(),
        }
        if labels is not None:
            # self.metrics.update(pred, labels)
            d.update({
                "labels": labels.cpu(),
                "loss": loss,
            })
        return d

    def training_step(self, train_batch, batch_idx):
        d = self.shared_step(train_batch)
        self.log("train_loss", d["loss"].item(), on_step=True, on_epoch=False)
        self.train_step_outputs.append(d)
        return d

    def validation_step(self, val_batch, batch_idx):
        d = self.shared_step(val_batch)
        self.log("val_loss", d["loss"].item(), on_step=True, on_epoch=False)
        self.val_step_outputs.append(d)
        return d

    def test_step(self, test_batch, batch_idx):
        d = self.shared_step(test_batch)
        self.log("test_loss", d["loss"].item(), on_step=True, on_epoch=False)
        return d
    

    @staticmethod
    def aggregate_preds_for_patients(
            preds: torch.Tensor,
            patient_inds: np.ndarray,
            mode : Literal["mean", "max"] = "max",
            target_mode : Literal["multiclass", "multilabel"] = "multiclass",
            ) -> torch.Tensor:
        mapping = {}
        for i, p in enumerate(patient_inds):
            mapping.setdefault(p, []).append(i)

        preds_agg_f = [preds[i] for i in mapping.values()]
        if mode == "mean":
            preds_agg = [p.mean(0) for p in preds_agg_f]
        elif mode == "max":
            if target_mode == "multilabel":
                preds_agg = []
                for p in preds_agg_f:
                    pp = torch.cat([p[:, :-1].max(0).values, torch.unsqueeze(p[:, -1].min(), 0)])
                    preds_agg.append(pp)
            else:
                preds_agg = [p.max(0).values for p in preds_agg_f]
        else:
            raise ValueError(f"Unknown aggregate mode: {mode}")
        preds_agg = torch.stack(preds_agg, dim=0)
        return preds_agg


    def aggregate_scores_for_patients(self,
            preds: torch.Tensor, 
            labels: torch.Tensor, 
            patient_inds: np.ndarray,
            ) -> dict:
        preds_agg = self.aggregate_preds_for_patients(
            preds, patient_inds, 
            mode=self.aggregate_mode, 
            target_mode=self.target_mode,
        )
        labels_agg = self.aggregate_preds_for_patients(
            labels, patient_inds, 
            mode="max",
            target_mode=self.target_mode,    
        )
        self.metrics.update(preds_agg, labels_agg)
        scores = self.metrics.compute()
        self.metrics.reset()
        return scores


    def get_val_scores(self, outputs: dict) -> dict: 
        preds, labels = [], []
        patient_inds = []
        for out in outputs:
            if "pred" in out:
                preds.append(out["pred"])
            if "labels" in out:
                labels.append(out["labels"])
            if "patient_inds" in out:
                patient_inds.append(out["patient_inds"])
        if len(preds) > 0 and len(labels) > 0:
            preds = torch.cat(preds)
            labels = torch.cat(labels)
            self.metrics.reset()
            self.metrics.update(preds, labels)
            scores = self.metrics.compute()
            self.metrics.reset()
            patient_inds = np.concatenate(patient_inds)
            scores_agg = self.aggregate_scores_for_patients(preds, labels, patient_inds)
            for k, v in scores_agg.items():
                scores[f"agg_{k}"] = v
            return scores
        else:
            return {}

    def log_val_scores(self, 
            outputs: dict, 
            mode : Literal["train", "val", "test"] = "val",
            ):
        scores = self.get_val_scores(outputs, mode=mode)
        for k, v in scores.items():
            self.log(f"{mode}_scores/{k}", v, on_step=False, on_epoch=True,
                prog_bar=(k in ["ce", "dice"]), logger=True)
        if "ce" in scores:
            self.log_dict({f"{mode}_ce": scores["ce"]})
        else:
            self.log_dict({f"{mode}_dice": scores["dice"]})

    def on_train_epoch_start(self):
        self.train_step_outputs = []
        self.log("lr", self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"], 
            on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
        gc.collect()

    def on_validation_epoch_start(self):
        self.val_step_outputs = []
        torch.cuda.empty_cache()
        gc.collect()

    def on_train_epoch_end(self):
        outputs = self.train_step_outputs
        self.log_val_scores(outputs, mode="train")
        torch.cuda.empty_cache()
        gc.collect()

    def on_validation_epoch_end(self):
        outputs = self.val_step_outputs
        self.log_val_scores(outputs, mode="val")
        torch.cuda.empty_cache()
        gc.collect()


class PLModel3D(PLModel):
    
    def forward(self, images: torch.Tensor, inds: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = self.model(images)
        results = []
        results_pat = []
        for i, (preds_single, inds_single) in enumerate(zip(preds, inds)):
            p = preds_single[:, inds_single]
            results.append(p)
            if len(inds_single) == 0:
                p = torch.zeros(preds_single.shape[0], 
                    dtype=preds_single.dtype, device=preds_single.device)
            elif self.aggregate_mode == "mean":
                p = p.mean(1)
            elif self.aggregate_mode == "max":
                p = p.max(1).values
            else:
                raise ValueError(f"Unknown aggregate mode: {self.aggregate_mode}")
            results_pat.append(p)
        results = torch.cat(results, dim=1)
        results = torch.transpose(results, 0, 1)
        results_pat = torch.stack(results_pat)
        return results, results_pat


    def shared_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        images = batch["img"]
        indexes = batch["index"]
        indexes_masked = batch["index_masked"]
        labels_m = batch.get("labels_m")
        label_patient = batch.get("labels_pat")

        pred, pred_patient = self.forward(images, indexes_masked)

        if labels_m is not None:
            loss = self.loss_function(pred, labels_m.to(pred.dtype))
            if self.target_mode == "multilabel":
                if self.class_weights is not None:
                    loss = self.class_weights.to(loss.device) * loss
                    loss = torch.sum(loss)
                else:
                    loss = torch.mean(loss)
        else:
            loss = None

        pred = self.activation(pred)
        pred_patient = self.activation(pred_patient)

        # self.metrics.update(pred, labels_m)

        d = {
            "pred": pred.detach().cpu(),
            "pred_patient": pred_patient.detach().cpu(),
            "inds": indexes,
        }
        if labels_m is not None:
            d.update({
                "labels": labels_m.cpu(),
                "labels_patient": label_patient.cpu(),
                "loss": loss,
            })
        return d


    def get_val_scores(self, outputs: dict, **kwargs) -> dict: 
        preds, labels = [], []
        preds_patients, labels_patients = [], []
        for out in outputs:
            if "pred" in out:
                preds.append(out["pred"])
            if "labels" in out:
                labels.append(out["labels"])
            if "pred_patient" in out:
                preds_patients.append(out["pred_patient"])
            if "labels_patient" in out:
                labels_patients.append(out["labels_patient"])
        if len(preds) > 0 and len(labels) > 0:
            preds = torch.cat(preds)
            labels = torch.cat(labels)
            self.metrics.reset()
            self.metrics.update(preds, labels)
            scores = self.metrics.compute()
            self.metrics.reset()

            preds_patients = torch.cat(preds_patients)
            labels_patients = torch.cat(labels_patients)
            self.metrics.update(preds_patients, labels_patients)
            scores_agg = self.metrics.compute()
            self.metrics.reset()

            for k, v in scores_agg.items():
                scores[f"agg_{k}"] = v
            return scores
        else:
            return {}
        
        
class PLModelSeq(PLModel3D):
    def forward(self, 
            embeds: torch.Tensor,
            preds: torch.Tensor,
            inds: List[List[int]],    
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preds, preds_0 = self.model(embeds, preds)
        results = []
        results_pat = []
        results_0 = []
        for i, (preds_single, preds_0_single, inds_single) in enumerate(zip(preds, preds_0, inds)):
            p = preds_single[:, inds_single]
            p0 = preds_0_single[:, inds_single]
            results.append(p)
            results_0.append(p0)
            if len(inds_single) == 0:
                p = torch.zeros(preds_single.shape[0], 
                    dtype=preds_single.dtype, device=preds_single.device)
            elif self.aggregate_mode == "mean":
                p = p.mean(1)
            elif self.aggregate_mode == "max":
                p = p.max(1).values
            else:
                raise ValueError(f"Unknown aggregate mode: {self.aggregate_mode}")
            results_pat.append(p)
        results = torch.cat(results, dim=1)
        results = torch.transpose(results, 0, 1)
        results_pat = torch.stack(results_pat)
        results_0 = torch.cat(results_0, dim=1)
        results_0 = torch.transpose(results_0, 0, 1)
        return results, results_0, results_pat
    
    def shared_step(self, batch: tuple) -> dict:
        embeds = batch["embeds"]
        input_preds = batch["preds"]
        indexes = batch["index"]
        indexes_masked = batch["index_masked"]
        labels_m = batch.get("labels_m")
        label_patient = batch.get("labels_pat")

        pred, pred_0, pred_patient = self.forward(embeds, input_preds, indexes_masked)

        if labels_m is not None:
            loss = self.loss_function(pred, labels_m.to(pred.dtype))
            loss_2 = self.loss_function(pred_0, labels_m.to(pred.dtype))
            loss += loss_2
            if self.target_mode == "multilabel":
                if self.class_weights is not None:
                    loss = self.class_weights.to(loss.device) * loss
                    loss = torch.sum(loss)
                else:
                    loss = torch.mean(loss)
        else:
            loss = None

        pred = self.activation(pred)
        pred_patient = self.activation(pred_patient)

        # self.metrics.update(pred, labels_m)

        d = {
            "pred": pred.detach().cpu(),
            "pred_patient": pred_patient.detach().cpu(),
            "inds": indexes,
        }
        if labels_m is not None:
            d.update({
                "labels": labels_m.cpu(),
                "labels_patient": label_patient.cpu(),
                "loss": loss,
            })
        return d
    
    
class PLModelSegm(PLModel):
    def __init__(self, 
            *args, 
            target_mode : Literal["multilabel", "multiclass", "binary"] = "multiclass",
            monitor_metric : str = "val_dice",
            monitor_mode : Literal["max", "min"] = "max",
            class_weights : Optional[np.ndarray] = None,
            loss_type : Literal["dice", "crossentropy"] = "dice",
            n_classes : int = 6,
            log_n_batches : int = 2,
            **kwargs,
            ):
        if class_weights is None and target_mode != "binary":
            class_weights = np.array([0.99676, 0.00018, 0.00144, 0.00041, 0.00047, 0.00074], np.float32)
            class_weights = 1. / class_weights
            class_weights /= class_weights.sum()
        super(PLModelSegm, self).__init__(
            *args, 
            target_mode=target_mode,
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            class_weights=class_weights,
            **kwargs,
        )
        # self.metrics = MultMetrics(
        #     class_weights=self.class_weights,
        #     device=self.device,
        #     task=self.target_mode,
        #     mode="segmentation",
        # )
        self.n_classes = n_classes
        self.metrics = MetricsSegmentation(
            class_weights=self.class_weights, 
            n_classes=self.n_classes,
        )
        self.loss_type = loss_type
        if self.loss_type == "dice":
            self.loss_function = DiceLoss(
                n_classes=self.n_classes, 
                class_weights=self.class_weights,
                activation=("softmax" if self.target_mode == "multiclass" else "sigmoid"),
            )
        elif self.loss_type == "crossentropy":
            self.loss_function = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_type}")

        self.log_n_batches = log_n_batches


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def shared_step(self, batch: Dict[str, Any], get_images : bool = False) -> Dict[str, Any]:
        images = batch["img"]
        indexes = batch["slice_indexes"]
        labels = batch.get("labels")
        
        pred = self.forward(images)
        
        if labels is not None:
            loss = self.loss_function(pred, labels.to(torch.int64))
            if loss.ndim > 4:
                loss = loss.mean(dim=(0, 2, 3, 4))
                self.class_weights = self.class_weights.to(loss.device)
                loss = (self.class_weights * loss).sum() / self.class_weights.sum()
            pred = self.activation(pred)
            scores = self.metrics(pred, labels)
        else:
            loss = None
            pred = self.activation(pred)
            
        d = {
            "inds": indexes,
        }
        if labels is not None:
            d.update({
                "loss": loss,
                "scores": scores,
            })
        if get_images:
            d.update({
                "images": images.detach().cpu(),
                "pred": pred.detach().cpu(),
            })
            if labels is not None:
                d["labels"] = labels.detach().cpu()
        return d
    
    def training_step(self, train_batch, batch_idx):
        d = self.shared_step(train_batch, get_images=(batch_idx < self.log_n_batches))
        self.log("train_loss", d["loss"].item(), on_step=True, on_epoch=False)
        self.train_step_outputs.append(d)
        return d

    def validation_step(self, val_batch, batch_idx):
        d = self.shared_step(val_batch, get_images=(batch_idx < self.log_n_batches))
        self.log("val_loss", d["loss"].item(), on_step=True, on_epoch=False)
        self.val_step_outputs.append(d)
        return d

    def test_step(self, test_batch, batch_idx):
        d = self.shared_step(test_batch, get_images=(batch_idx < self.log_n_batches))
        self.log("test_loss", d["loss"].item(), on_step=True, on_epoch=False)
        return d

    def log_images(self, 
            study: torch.Tensor, 
            target: torch.Tensor, 
            pred: torch.Tensor,
            nrow : int = 6,
            ignore_empty : bool = True,
            mode : Literal["train", "val"] = "val",
            ):
        if isinstance(study, list):
            study = torch.cat(study)
        if isinstance(target, list):
            target = torch.cat(target)
        if isinstance(pred, list):
            pred = torch.cat(pred)

        if study.ndim == 5:
            study = torch.permute(study, (0, 2, 1, 3, 4))
            study = study.reshape(-1, study.shape[2], study.shape[3], study.shape[4])
        if target.ndim == 4:
            target = target.view(-1, target.shape[2], target.shape[3])
        if pred.ndim == 5:
            pred = torch.permute(pred, (0, 2, 1, 3, 4))
            pred = pred.reshape(-1, pred.shape[2], pred.shape[3], pred.shape[4])

        image_list = []
        for i in range(study.shape[0]):
            if ignore_empty and ((study[i, 0] > 0) & (study[i, 1] < 1)).float().mean() < 0.02:
                continue
            image_list.extend([
                torch.from_numpy(make_image_from_slice_channel(study[i], 0)),
                torch.from_numpy(make_image_from_classes(target[i], n_classes=self.n_classes)),
                torch.from_numpy(make_image_from_classes(pred[i], n_classes=self.n_classes)),
            ])
        grid = torchvision.utils.make_grid(image_list, nrow=nrow, pad_value=1)
        self.logger.experiment.add_image(f"{mode}/images", grid, self.current_epoch)
        return
    
    def get_val_scores(self, 
            outputs: Dict[str, Any], 
            mode : Literal["train", "val", "test"] = "val",
            ) -> Dict[str, Any]:
        scores = {}
        for out in outputs:
            if "scores" in out:
                for k, v in out["scores"].items():
                    scores.setdefault(k, []).append(v)
        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores

    def on_train_epoch_end(self):
        outputs = self.train_step_outputs
        self.log_val_scores(outputs, mode="train")
        self.log_images(
            study=[out["images"] for out in outputs if "images" in out],
            target=[out["labels"] for out in outputs if "labels" in out],
            pred=[out["pred"] for out in outputs if "pred" in out],
            mode="train",
        )
        torch.cuda.empty_cache()
        gc.collect()

    def on_validation_epoch_end(self):
        outputs = self.val_step_outputs
        self.log_val_scores(outputs, mode="val")
        self.log_images(
            study=[out["images"] for out in outputs if "images" in out],
            target=[out["labels"] for out in outputs if "labels" in out],
            pred=[out["pred"] for out in outputs if "pred" in out],
            mode="val",
        )
        torch.cuda.empty_cache()
        gc.collect()