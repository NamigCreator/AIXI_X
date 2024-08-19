import numpy as np
import torch
from torchmetrics.classification import (
    Precision, Recall, Accuracy, F1Score,
    MatthewsCorrCoef, AveragePrecision,
    AUROC,
    Dice, JaccardIndex
)
from torchmetrics import Metric
from typing import Optional, Literal, Dict, Any, Callable, Union


class CrossEntropyMetric(Metric):
    def __init__(self, 
            task : Literal["multilabel", "multiclass", "binary"] = "multiclass",
            class_weights : Optional[torch.Tensor] = None,
            ):
        super().__init__()
        self.task = task
        self.class_weights = class_weights
        if task == "multilabel":
            self.loss_function = torch.nn.BCELoss(reduction="none")
        elif task == "multiclass":
            self.loss_function = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        elif task == "binary":
            self.loss_function = torch.nn.BCELoss(reduction="none")
        else:
            raise ValueError(f"Unknown mode: {task}")
        self.add_state("sum_loss", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        loss = self.loss_function(preds, targets.to(preds.dtype)).mean(0)
        if loss.ndim > 0 and self.class_weights is not None:
            loss = torch.sum(self.class_weights * loss)
        elif loss.ndim > 0:
            loss = torch.mean(loss)
        self.sum_loss += loss
        self.total_batches += 1
    
    def compute(self) -> torch.Tensor:
        return self.sum_loss / self.total_batches


class DiceLoss(torch.nn.Module):
    def __init__(self, 
            n_classes : Optional[int] = 6, 
            eps : float = 1e-8,
            class_weights : Optional[torch.Tensor] = None,
            activation : Union[Literal["sigmoid", "softmax", None], Callable] = None,
            ):
        super().__init__()
        self.n_classes = n_classes
        self.eps = eps
        self.class_weights = class_weights
        self.activation = activation
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            pred = torch.sigmoid(pred)
        elif self.activation == "softmax":
            pred = torch.softmax(pred, dim=1)
        elif isinstance(self.activation, Callable):
            pred = self.activation(pred)
        else:
            raise ValueError(f"Unkown activation: {self.activation}")

        if pred.ndim == 5:
            dims_1 = (0, 4, 1, 2, 3)
            dims_2 = (0, 2, 3, 4)
        else:
            dims_1 = (0, 3, 1, 2)
            dims_2 = (0, 2, 3)

        if self.n_classes is not None and self.n_classes > 1:
            target_one_hot = torch.nn.functional.one_hot(
                target.to(torch.int64), num_classes=self.n_classes)
            target_one_hot = torch.permute(target_one_hot, dims_1)
        elif (self.n_classes == 1 or self.n_classes is None) and pred.ndim > target.ndim:
            target_one_hot = target.view(target.shape[0], 1, *target.shape[1:])
        else:
            target_one_hot = target
        
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(pred.device)

        intersection = torch.sum(pred * target_one_hot, dim=dims_2)
        cardinality = torch.sum(pred + target_one_hot, dim=dims_2)
        
        dice_score = 2.0 * intersection / (cardinality + self.eps)
        dice_loss = -dice_score + 1.0
        if self.class_weights is None:
            dice_loss = torch.mean(dice_loss)
        else:
            dice_loss = torch.sum(dice_loss * self.class_weights) / self.class_weights.sum()
        return dice_loss


class MultMetrics(torch.nn.Module):
    def __init__(self, 
            class_weights : Optional[np.ndarray] = None,
            n_classes : Optional[int] = 6,
            task : Literal["multilabel", "multiclass", "binary"] = "multiclass",
            threshold : float = 0.01,
            device : str = "cuda",
            mode : Literal["classification", "segmentation"] = "classification",
            ):
        super().__init__()
        self._n_classes = n_classes
        self.class_weights = class_weights
        if self.class_weights is None:
            self.class_weights = torch.ones(self._n_classes, dtype=torch.float32, device=device)
            self.class_weights /= self.class_weights.sum()
        self.threshold = threshold
        self.task = task
        add_args = {"task": task}
        if task == "multilabel":
            add_args["num_labels"] = self.n_classes
        elif task == "multiclass":
            add_args["num_classes"] = self.n_classes
        elif task == "binary":
            pass
        else:
            raise ValueError(f"Unknown mode: {task}")
        self.mode = mode
        if self.mode == "classification":
            self.metrics = {
                "ce": CrossEntropyMetric(task=task),
                "acc": Accuracy(average=None, threshold=threshold, **add_args).to(device),
                "auc": AUROC(average=None, **add_args).to(device),
                "map": AveragePrecision(average=None, **add_args).to(device),
                "f1": F1Score(average=None, threshold=threshold, **add_args).to(device),
                "mcc": MatthewsCorrCoef(threshold=threshold, **add_args).to(device),
                "pre": Precision(threshold=threshold, average=None, **add_args).to(device),
                "rec": Recall(threshold=threshold, average=None, **add_args).to(device),
            }
        else:
            self.metrics = {
                "iou": JaccardIndex(threshold=threshold, **add_args).to(device),
                "acc": Accuracy(average=None, threshold=threshold, **add_args).to(device),
                "dice": Dice(num_classes=self.n_classes, average="macro").to(device),
            }

    @property
    def n_classes(self) -> int:
        if self.class_weights is None:
            return self._n_classes
        else:
            return len(self.class_weights)

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        for k in self.metrics.keys():
            self.metrics[k] = self.metrics[k].to(preds.device)
        self.class_weights = self.class_weights.to(preds.device)
        labels = labels.to(torch.int)
        if self.task == "multiclass" and self.mode == "classification":
            labels_m = torch.argmax(labels, dim=1)
        elif self.mode == "segmentation":
            if self.n_classes is not None and self.n_classes > 1:
                labels_m = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=self.n_classes)
                if preds.ndim == 5:
                    dims = (0, 4, 1, 2, 3)
                else:
                    dims = (0, 3, 1, 2)
                labels_m = torch.permute(labels_m, dims)
            elif preds.shape[1] == 1 and labels.ndim < preds.ndim:
                labels_m = torch.view(labels, (labels.shape[0], 1, labels.shape[1], labels.shape[2]))
            # labels_m = labels.to(torch.int64)
        else:
            labels_m = labels
        for metric_name, metric_func in self.metrics.items():
            if metric_name == "ce":
                metric_func.update(preds, labels)
            elif metric_name == "dice":
                metric_func.update(torch.argmax(preds, dim=1), labels)
            elif self.mode == "segmentation":
                metric_func.update(preds, labels_m)
            else:
                metric_func.update(preds, labels_m)
        return
    
    def compute(self) -> Dict[str, Any]:
        scores = {}
        for metric_name, metric_func in self.metrics.items():
            score = metric_func.compute()
            if score.ndim > 0:
                for i, s in enumerate(score):
                    scores[f"{metric_name}_{i}"] = s.item()
                score = torch.sum(self.class_weights * score)
            scores[metric_name] = score.item()
        return scores
    
    def reset(self):
        for metric_func in self.metrics.values():
            metric_func.reset()
        return
    
    
class MetricsSegmentation(torch.nn.Module):
    def __init__(self,
            class_weights : Optional[np.ndarray] = None,
            n_classes : Optional[int] = 6,
            threshold : float = 0.01,   
            device : str = "cuda",
            ):
        super().__init__()
        self.device = device
        self._n_classes = n_classes
        self.class_weights = class_weights
        if self.class_weights is None and self._n_classes is not None and self._n_classes > 0:
            self.class_weights = torch.ones(self._n_classes, dtype=torch.float32, device=self.device)
            self.class_weights /= self.class_weights.sum()
        self.threshold = threshold
        self.eps = 1e-8
        
    @property
    def n_classes(self) -> int:
        if self.class_weights is None:
            return self._n_classes
        else:
            return len(self.class_weights)
        
    def _to_onehot(self, target: torch.Tensor) -> torch.Tensor:
        target_one_hot = torch.nn.functional.one_hot(target.to(torch.int64), num_classes=self.n_classes)
        if target.ndim == 4:
            dims = (0, 4, 1, 2, 3)
        else:
            dims = (0, 3, 1, 2)
        target_one_hot = torch.permute(target_one_hot, dims)
        return target_one_hot
        
    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(preds.device)
        labels = labels.to(torch.int)
        
        preds_bin = preds >= self.threshold

        if self.n_classes is not None:
            labels = self._to_onehot(labels)
        elif preds.ndim > labels.ndim:
            labels = labels.view(labels.shape[0], 1, *labels.shape[1:])
        
        if preds.ndim == 5:
            dims = (0, 2, 3, 4)
        else:
            dims = (0, 2, 3)

        intersection = (preds_bin * labels).sum(dim=dims)
        union = (preds_bin | labels).sum(dim=dims)
        target = labels.sum(dim=dims)
        iou = (intersection / (union + self.eps))
        rec = (intersection / (target + self.eps))
        dice = 2 * intersection / (preds_bin.sum(dim=dims) + labels.sum(dim=dims) + self.eps)

        if self.class_weights is not None:
            mean_iou = (iou * self.class_weights).sum() / self.class_weights.sum()
            mean_rec = (rec * self.class_weights).sum() / self.class_weights.sum()
            mean_dice = (dice * self.class_weights).sum() / self.class_weights.sum()
        else:
            mean_iou = iou.mean()
            mean_rec = rec.mean()
            mean_dice = dice.mean()
        
        scores = {
            "iou": mean_iou.item(),
            "rec": mean_rec.item(),
            "dice": mean_dice.item(),
        }
        if self.n_classes is not None:
            for i in range(self.n_classes):
                scores[f"iou_{i}"] = iou[i].item()
                scores[f"rec_{i}"] = rec[i].item()
                scores[f"dice_{i}"] = dice[i].item()
        return scores
