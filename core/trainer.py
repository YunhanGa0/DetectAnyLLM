import os
import time

import numpy as np
import torch
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CustomDataset
from .loss import calculate_multitask_loss
from .metrics import classification_metrics, regression_metrics
from .model import DiscrepancyEstimator


class Trainer:
    def train(
        self,
        accelerator: Accelerator,
        model: DiscrepancyEstimator,
        train_dataset: CustomDataset,
        eval_dataset: CustomDataset,
        learning_rate=1e-4,
        num_epochs=5,
        eval_freq=1,
        save_freq=5,
        save_directory="./ckpt/",
        save_name=None,
        train_batch_size=1,
        eval_batch_size=1,
        track_with_wandb=True,
        eval=False,
        lambda_lir=1.0,
        lambda_jaccard=1.0,
        lambda_sentence_jaccard=1.0,
        regression_loss_type="mse",
    ):
        start_time = time.time()
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=eval_dataset.collate_fn,
        )
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs * len(train_loader)),
            eta_min=0,
            last_epoch=-1,
        )
        model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, eval_loader, lr_scheduler
        )

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            model.train()
            total_losses = []
            total_cls_losses = []
            total_lir_losses = []
            total_jaccard_losses = []
            total_sentence_losses = []

            for step, batch in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Training epoch {epoch + 1}",
                disable=not accelerator.is_main_process,
            ):
                outputs = model(batch["input_ids"], batch["attention_mask"])
                loss, loss_parts = calculate_multitask_loss(
                    outputs=outputs,
                    labels=batch["labels"],
                    lir=batch["lir"],
                    jaccard=batch["jaccard"],
                    sentence_jaccard=batch["sentence_jaccard"],
                    lambda_lir=lambda_lir,
                    lambda_jaccard=lambda_jaccard,
                    lambda_sentence_jaccard=lambda_sentence_jaccard,
                    regression_loss_type=regression_loss_type,
                )
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()

                total_losses.append(loss.detach())
                total_cls_losses.append(loss_parts["loss_cls"])
                total_lir_losses.append(loss_parts["loss_lir"])
                total_jaccard_losses.append(loss_parts["loss_jaccard"])
                total_sentence_losses.append(loss_parts["loss_sentence_jaccard"])

                if track_with_wandb and accelerator.is_main_process:
                    accelerator.log(
                        {
                            "train/loss": float(loss.detach().cpu()),
                            "train/lr": lr_scheduler.get_last_lr()[0],
                        },
                        step=step + epoch * len(train_loader),
                    )

            accelerator.wait_for_everyone()
            gathered_train = {
                "loss": accelerator.gather_for_metrics(torch.stack(total_losses)).mean().item(),
                "loss_cls": accelerator.gather_for_metrics(torch.stack(total_cls_losses)).mean().item(),
                "loss_lir": accelerator.gather_for_metrics(torch.stack(total_lir_losses)).mean().item(),
                "loss_jaccard": accelerator.gather_for_metrics(torch.stack(total_jaccard_losses)).mean().item(),
                "loss_sentence_jaccard": accelerator.gather_for_metrics(
                    torch.stack(total_sentence_losses)
                ).mean().item(),
            }

            log_dict = {f"train/{k}": v for k, v in gathered_train.items()}
            if accelerator.is_main_process:
                accelerator.print(
                    f"Epoch {epoch + 1} | Time {time.time() - epoch_start_time:.2f}s | "
                    f"Loss {gathered_train['loss']:.4f}"
                )

            if eval and ((epoch + 1) % eval_freq == 0 or (epoch + 1) == num_epochs):
                eval_metrics, _ = self.evaluate(accelerator, model, eval_loader)
                log_dict.update({f"eval/{k}": v for k, v in eval_metrics.items()})
                if accelerator.is_main_process:
                    accelerator.print(
                        " | ".join(
                            [
                                f"Eval macro-F1 {eval_metrics['macro_F1']:.4f}",
                                f"AUROC {eval_metrics['multi_class_AUROC']:.4f}",
                                f"MAE(LIR) {eval_metrics['MAE(LIR)']:.4f}",
                            ]
                        )
                    )

            if track_with_wandb and accelerator.is_main_process:
                accelerator.log(log_dict, step=(epoch + 1) * len(train_loader))

            if (epoch + 1) == num_epochs or (epoch + 1) % save_freq == 0:
                accelerator.wait_for_everyone()
                if save_name is None:
                    raise ValueError("save_name should not be None")
                this_epoch_save_name = f"{save_name}_e{epoch + 1}"
                output_dir = os.path.join(save_directory, this_epoch_save_name)
                os.makedirs(output_dir, exist_ok=True)
                if accelerator.is_main_process:
                    accelerator.print(f"Saving model to {output_dir}")
                    accelerator.unwrap_model(model).save_pretrained(output_dir)

        if track_with_wandb:
            accelerator.end_training()
        if accelerator.is_main_process:
            accelerator.print(f"Finished Training! Total Time: {time.time() - start_time:.2f} sec")

    def evaluate(self, accelerator, model, eval_loader, save_predictions=False):
        model.eval()
        all_labels = []
        all_probabilities = []
        all_pred_classes = []
        all_lir_targets = []
        all_lir_preds = []
        all_jaccard_targets = []
        all_jaccard_preds = []
        all_sentence_targets = []
        all_sentence_preds = []
        prediction_rows = []

        with torch.no_grad():
            for batch in tqdm(
                eval_loader,
                total=len(eval_loader),
                desc="Evaluating",
                disable=not accelerator.is_main_process,
            ):
                outputs = model(batch["input_ids"], batch["attention_mask"])

                gathered_labels = accelerator.gather_for_metrics(batch["labels"]).cpu().numpy()
                gathered_probabilities = (
                    accelerator.gather_for_metrics(outputs["probabilities"]).cpu().numpy()
                )
                gathered_pred_classes = (
                    accelerator.gather_for_metrics(outputs["pred_class"]).cpu().numpy()
                )
                gathered_lir_targets = accelerator.gather_for_metrics(batch["lir"]).cpu().numpy()
                gathered_lir_preds = accelerator.gather_for_metrics(outputs["pred_lir"]).cpu().numpy()
                gathered_jaccard_targets = (
                    accelerator.gather_for_metrics(batch["jaccard"]).cpu().numpy()
                )
                gathered_jaccard_preds = (
                    accelerator.gather_for_metrics(outputs["pred_jaccard"]).cpu().numpy()
                )
                gathered_sentence_targets = (
                    accelerator.gather_for_metrics(batch["sentence_jaccard"]).cpu().numpy()
                )
                gathered_sentence_preds = accelerator.gather_for_metrics(
                    outputs["pred_sentence_jaccard"]
                ).cpu().numpy()

                all_labels.extend(gathered_labels.tolist())
                all_probabilities.extend(gathered_probabilities.tolist())
                all_pred_classes.extend(gathered_pred_classes.tolist())
                all_lir_targets.extend(gathered_lir_targets.tolist())
                all_lir_preds.extend(gathered_lir_preds.tolist())
                all_jaccard_targets.extend(gathered_jaccard_targets.tolist())
                all_jaccard_preds.extend(gathered_jaccard_preds.tolist())
                all_sentence_targets.extend(gathered_sentence_targets.tolist())
                all_sentence_preds.extend(gathered_sentence_preds.tolist())

                if save_predictions and accelerator.is_main_process:
                    local_probs = outputs["probabilities"].detach().cpu().numpy()
                    local_pred_classes = outputs["pred_class"].detach().cpu().numpy()
                    local_lir_preds = outputs["pred_lir"].detach().cpu().numpy()
                    local_jaccard_preds = outputs["pred_jaccard"].detach().cpu().numpy()
                    local_sentence_preds = (
                        outputs["pred_sentence_jaccard"].detach().cpu().numpy()
                    )
                    for idx, sample_id in enumerate(batch["ids"]):
                        prediction_rows.append(
                            {
                                "id": sample_id,
                                "split": batch["splits"][idx],
                                "text": batch["texts"][idx],
                                "gold_label_6way": int(batch["labels"][idx].detach().cpu()),
                                "gold_ai_ratio": float(batch["target_ai_ratio"][idx].detach().cpu()),
                                "gold_lir": float(batch["lir"][idx].detach().cpu()),
                                "gold_jaccard": float(batch["jaccard"][idx].detach().cpu()),
                                "gold_sentence_jaccard": float(
                                    batch["sentence_jaccard"][idx].detach().cpu()
                                ),
                                "pred_class_6way": int(local_pred_classes[idx]),
                                "prob_6way": [float(x) for x in local_probs[idx]],
                                "pred_lir": float(local_lir_preds[idx]),
                                "pred_jaccard": float(local_jaccard_preds[idx]),
                                "pred_sentence_jaccard": float(local_sentence_preds[idx]),
                                "target_notes": batch["target_notes"][idx],
                            }
                        )

        metrics = classification_metrics(all_labels, all_pred_classes, all_probabilities)
        metrics.update(regression_metrics(all_lir_targets, all_lir_preds, "LIR"))
        metrics.update(regression_metrics(all_jaccard_targets, all_jaccard_preds, "Jaccard"))
        metrics.update(
            regression_metrics(
                all_sentence_targets, all_sentence_preds, "Sentence Jaccard"
            )
        )
        metrics["num_samples"] = len(all_labels)

        return metrics, prediction_rows
