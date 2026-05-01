import argparse
import datetime
import os

from accelerate import Accelerator
from peft import LoraConfig, TaskType

from core.dataset import CustomDataset
from core.model import DiscrepancyEstimator
from core.trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument("--scoring_model_name", type=str, default="roberta-base")
parser.add_argument("--cache_dir", type=str, default="./model/")
parser.add_argument("--train_data_path", type=str, required=True)
parser.add_argument("--eval_data_path", type=str, required=True)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--eval_freq", type=int, default=1)
parser.add_argument("--save_freq", type=int, default=1)
parser.add_argument("--save_directory", type=str, default="./ckpt/")
parser.add_argument("--ckpt_name", type=str, default=None)
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument("--wandb_dir", type=str, default="./log/")
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lora_rank", type=int, default=8)
parser.add_argument("--lora_alpha", type=float, default=32.0)
parser.add_argument("--lora_dropout", type=float, default=0.1)
parser.add_argument("--lambda_lir", type=float, default=1.0)
parser.add_argument("--lambda_jaccard", type=float, default=1.0)
parser.add_argument("--lambda_sentence_jaccard", type=float, default=1.0)
parser.add_argument("--regression_loss_type", type=str, default="mse", choices=["mse", "l1"])
parser.add_argument("--eval", type=bool, default=True)


def main(args):
    model = DiscrepancyEstimator(
        scoring_model_name=args.scoring_model_name,
        cache_dir=args.cache_dir,
        dropout=args.dropout,
    )
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.add_lora_config(lora_config)

    if args.wandb:
        os.makedirs(args.wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = args.wandb_dir

    run_name = args.ckpt_name or (
        f"multitask_{args.scoring_model_name.split('/')[-1]}_"
        f"{os.path.basename(args.train_data_path).split('.')[0]}_"
        f"lr{args.learning_rate}_bs{args.train_batch_size}"
    )

    if args.wandb:
        if args.wandb_entity is None:
            assert os.environ.get("WANDB_MODE") == "offline", (
                "Please set WANDB_MODE to offline or provide a wandb_entity"
            )
        now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(
            project_name="DetectAnyLLM-Multitask",
            config=vars(args),
            init_kwargs={"wandb": {"entity": args.wandb_entity, "name": f"{run_name}_{now_time}"}},
        )
    else:
        accelerator = Accelerator()

    if accelerator.is_main_process:
        accelerator.print(args)
        if hasattr(model.backbone, "print_trainable_parameters"):
            model.backbone.print_trainable_parameters()

    train_dataset = CustomDataset(
        data_path=args.train_data_path,
        tokenizer=model.scoring_tokenizer,
        max_length=args.max_length,
    )
    eval_dataset = CustomDataset(
        data_path=args.eval_data_path,
        tokenizer=model.scoring_tokenizer,
        max_length=args.max_length,
    )

    trainer = Trainer()
    trainer.train(
        accelerator=accelerator,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        save_directory=args.save_directory,
        save_name=run_name,
        track_with_wandb=args.wandb,
        eval=args.eval,
        lambda_lir=args.lambda_lir,
        lambda_jaccard=args.lambda_jaccard,
        lambda_sentence_jaccard=args.lambda_sentence_jaccard,
        regression_loss_type=args.regression_loss_type,
    )


if __name__ == "__main__":
    main(parser.parse_args())
