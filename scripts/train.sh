SCORING_MODEL="roberta-base"
TRAIN_DATA="./mydata/benchmark_grouped/train.jsonl"
EVAL_DATA="./mydata/benchmark_grouped/test.jsonl"
CKPT_NAME="multitask_benchmark_grouped"
WANDB_DIR="./log"

# Default to offline wandb so the script works on servers without network access.
# For online logging, run:
#   wandb login
#   WANDB_MODE=online WANDB_ENTITY=your_entity sh scripts/train.sh
export WANDB_MODE="${WANDB_MODE:-offline}"

WANDB_ARGS="--use_wandb --wandb_dir ${WANDB_DIR}"
if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ARGS="${WANDB_ARGS} --wandb_entity ${WANDB_ENTITY}"
fi

accelerate launch train.py \
    --scoring_model_name ${SCORING_MODEL} \
    --train_data_path ${TRAIN_DATA} \
    --eval_data_path ${EVAL_DATA} \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --num_epochs 5 \
    --save_freq 1 \
    --ckpt_name ${CKPT_NAME} \
    --lambda_lir 1.0 \
    --lambda_jaccard 1.0 \
    --lambda_sentence_jaccard 1.0 \
    --regression_loss_type mse \
    --eval True \
    ${WANDB_ARGS}
