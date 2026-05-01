import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, roc_auc_score


CLASS_NAMES = [0, 1, 2, 3, 4, 5]
CLASS_PERCENT = [0, 20, 40, 60, 80, 100]


def classification_metrics(labels, pred_classes, probabilities):
    labels = np.asarray(labels)
    pred_classes = np.asarray(pred_classes)
    probabilities = np.asarray(probabilities)

    metrics = {}
    per_class_f1 = f1_score(
        labels,
        pred_classes,
        labels=CLASS_NAMES,
        average=None,
        zero_division=0,
    )
    for class_idx, percent in enumerate(CLASS_PERCENT):
        metrics[f"F1@{percent}"] = float(per_class_f1[class_idx])

    metrics["macro_F1"] = float(
        f1_score(labels, pred_classes, labels=CLASS_NAMES, average="macro", zero_division=0)
    )

    try:
        metrics["multi_class_AUROC"] = float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["multi_class_AUROC"] = float("nan")

    return metrics


def regression_metrics(targets, predictions, prefix):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    return {
        f"MAE({prefix})": float(mean_absolute_error(targets, predictions)),
        f"MSE({prefix})": float(mean_squared_error(targets, predictions)),
    }
