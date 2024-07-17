import numpy as np
import torch
from torchmetrics.classification import (
    Precision, Recall, Accuracy, F1Score,
    MatthewsCorrCoef, AveragePrecision,
    AUROC,
)
from torchmetrics import Metric
from typing import Optional, Literal, Dict, Any


class CrossEntropyMetric(Metric):
    def __init__(self, 
            task : Literal["multilabel", "multiclass"] = "multiclass",
            class_weights : Optional[torch.Tensor] = None,
            ):
        super().__init__()
        self.task = task
        self.class_weights = class_weights
        if task == "multilabel":
            self.loss_function = torch.nn.BCELoss(reduction="none")
        elif task == "multiclass":
            self.loss_function = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")
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



class MultMetrics(torch.nn.Module):
    def __init__(self, 
            class_weights : Optional[np.ndarray] = None,
            n_classes : Optional[int] = 6,
            task : Literal["multilabel", "multiclass"] = "multiclass",
            threshold : float = 0.01,
            device : str = "cuda",
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
        else:
            raise ValueError(f"Unknown mode: {task}")
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
        if self.task == "multiclass":
            labels_m = torch.argmax(labels, dim=1)
        else:
            labels_m = labels
        for metric_name, metric_func in self.metrics.items():
            if metric_name == "ce":
                metric_func.update(preds, labels)
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