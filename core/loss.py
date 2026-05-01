import torch.nn.functional as F


def calculate_multitask_loss(
    outputs,
    labels,
    lir,
    jaccard,
    sentence_jaccard,
    lambda_lir=1.0,
    lambda_jaccard=1.0,
    lambda_sentence_jaccard=1.0,
    regression_loss_type="mse",
):
    cls_loss = F.cross_entropy(outputs["logits"], labels)

    if regression_loss_type == "l1":
        regression_loss = F.l1_loss
    else:
        regression_loss = F.mse_loss

    lir_loss = regression_loss(outputs["pred_lir"], lir)
    jaccard_loss = regression_loss(outputs["pred_jaccard"], jaccard)
    sentence_jaccard_loss = regression_loss(
        outputs["pred_sentence_jaccard"], sentence_jaccard
    )

    total_loss = (
        cls_loss
        + lambda_lir * lir_loss
        + lambda_jaccard * jaccard_loss
        + lambda_sentence_jaccard * sentence_jaccard_loss
    )
    return total_loss, {
        "loss_cls": cls_loss.detach(),
        "loss_lir": lir_loss.detach(),
        "loss_jaccard": jaccard_loss.detach(),
        "loss_sentence_jaccard": sentence_jaccard_loss.detach(),
    }
