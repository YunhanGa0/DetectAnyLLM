SCORING_MODEL="roberta-base"
CKPT_NAME="multitask_deepseek_v32_demo"
CKPT_PATH="./ckpt/${CKPT_NAME}_e5"
EVAL_DATA="./mixed_dataset_DeepSeek-V3.2.jsonl"

accelerate launch eval.py \
    --scoring_model_name ${SCORING_MODEL} \
    --pretrained_model_name_or_path ${CKPT_PATH} \
    --eval_data_path ${EVAL_DATA} \
    --eval_batch_size 8 \
    --save_dir ./results \
    --save_file mixed_dataset_eval.json
