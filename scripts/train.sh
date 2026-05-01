SCORING_MODEL="roberta-base"
TRAIN_DATA="./mixed_dataset_DeepSeek-V3.2.jsonl"
EVAL_DATA="./mixed_dataset_DeepSeek-V3.2.jsonl"
CKPT_NAME="multitask_deepseek_v32_demo"

accelerate launch train.py \
    --scoring_model_name ${SCORING_MODEL} \
    --train_data_path ${TRAIN_DATA} \
    --eval_data_path ${EVAL_DATA} \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --num_epochs 5 \
    --save_freq 1 \
    --ckpt_name ${CKPT_NAME} \
    --eval True
