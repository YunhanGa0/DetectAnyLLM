SCORING_MODEL="${SCORING_MODEL:-roberta-base}"
TRAIN_DATA="${TRAIN_DATA:-./mydata/benchmark_grouped/train.jsonl}"
TEST_DATA="${TEST_DATA:-./mydata/benchmark_grouped/test.jsonl}"
CKPT_NAME="${CKPT_NAME:-multitask_benchmark_grouped}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WANDB_DIR="${WANDB_DIR:-./log}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
RESULTS_FILE="${RESULTS_FILE:-benchmark_grouped_test_eval.json}"

# Default to offline wandb so the script works on servers without network access.
# For online logging, run:
#   wandb login
#   WANDB_MODE=online WANDB_ENTITY=your_entity sh scripts/train_and_eval.sh
export WANDB_MODE="${WANDB_MODE:-offline}"

WANDB_ARGS="--use_wandb --wandb_dir ${WANDB_DIR}"
if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ARGS="${WANDB_ARGS} --wandb_entity ${WANDB_ENTITY}"
fi

accelerate launch train.py \
    --scoring_model_name ${SCORING_MODEL} \
    --train_data_path ${TRAIN_DATA} \
    --eval_data_path ${TEST_DATA} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --save_freq 1 \
    --eval_freq 1 \
    --ckpt_name ${CKPT_NAME} \
    --lambda_lir 1.0 \
    --lambda_jaccard 1.0 \
    --lambda_sentence_jaccard 1.0 \
    --regression_loss_type mse \
    --eval True \
    ${WANDB_ARGS}

CKPT_PATH="./ckpt/${CKPT_NAME}_e${NUM_EPOCHS}"

accelerate launch eval.py \
    --scoring_model_name ${SCORING_MODEL} \
    --pretrained_model_name_or_path ${CKPT_PATH} \
    --eval_data_path ${TEST_DATA} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --save_dir ${RESULTS_DIR} \
    --save_file ${RESULTS_FILE}
