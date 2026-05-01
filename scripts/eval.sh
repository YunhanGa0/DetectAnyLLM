SCORING_MODEL="roberta-base"
CKPT_NAME="multitask_benchmark_grouped"
CKPT_PATH="./ckpt/${CKPT_NAME}_e5"
EVAL_DATA="./mydata/benchmark_grouped/test.jsonl"

accelerate launch eval.py \
    --scoring_model_name ${SCORING_MODEL} \
    --pretrained_model_name_or_path ${CKPT_PATH} \
    --eval_data_path ${EVAL_DATA} \
    --eval_batch_size 8 \
    --save_dir ./results \
    --save_file benchmark_grouped_test_eval.json
