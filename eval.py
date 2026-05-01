import argparse
import json
import os

from accelerate import Accelerator
from torch.utils.data import DataLoader

from core.dataset import CustomDataset
from core.model import DiscrepancyEstimator
from core.trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument("--scoring_model_name", type=str, default="roberta-base")
parser.add_argument("--cache_dir", type=str, default="./model/")
parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
parser.add_argument("--eval_data_path", type=str, default="./mydata/benchmark_grouped/test.jsonl")
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--save_dir", type=str, default="./results/")
parser.add_argument("--save_file", type=str, default=None)
parser.add_argument("--max_length", type=int, default=1024)


def main(args):
    accelerator = Accelerator()
    model = DiscrepancyEstimator(
        scoring_model_name=args.scoring_model_name,
        cache_dir=args.cache_dir,
        pretrained_ckpt=args.pretrained_model_name_or_path,
    )

    eval_dataset = CustomDataset(
        data_path=args.eval_data_path,
        tokenizer=model.scoring_tokenizer,
        max_length=args.max_length,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn,
    )

    model, eval_loader = accelerator.prepare(model, eval_loader)
    trainer = Trainer()
    metrics, predictions = trainer.evaluate(
        accelerator=accelerator,
        model=model,
        eval_loader=eval_loader,
        save_predictions=True,
    )

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        save_name = args.save_file or (
            f"eval_{os.path.basename(args.eval_data_path).split('.')[0]}.json"
        )
        if not save_name.endswith(".json"):
            save_name = f"{save_name}.json"
        save_path = os.path.join(args.save_dir, save_name)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": args.pretrained_model_name_or_path,
                    "eval_dataset": args.eval_data_path,
                    "metrics": metrics,
                    "predictions": predictions,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        accelerator.print(json.dumps(metrics, ensure_ascii=False, indent=2))
        accelerator.print(f"Saved results to {save_path}")


if __name__ == "__main__":
    main(parser.parse_args())
