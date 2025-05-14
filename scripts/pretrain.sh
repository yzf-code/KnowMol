#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:
PROMPT_VERSION=v1
MODEL_VERSION=vicuna-v1-3-7b

GRAPH_TOWER="moleculestm"
if [ "$GRAPH_TOWER" == "graphmvp" ]; then
    INIT_CHECKPOINT_GNN="./checkpoints/graphmvp.pth"
elif [ "$GRAPH_TOWER" == "moleculestm" ]; then
    INIT_CHECKPOINT_GNN="./checkpoints/MoleculeSTM/molecule_model.pth"
else
    echo "Not supported graph tower"
fi

CHECKPOINT_FOLDER_PREFIX="/data/yangzaifei/backup_checkpoints/new_arch_and_token_my_dataset_100k_r_64_a_256_dp01"

deepspeed --include localhost:2,3,4,5,6,7 --master_port 29501 llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path "" \
    --data_type "pretrain_2_task" \
    --graph_tower $GRAPH_TOWER \
    --init_checkpoint $INIT_CHECKPOINT_GNN \
    --add_molecule_tokens False \
    --lora_dropout 0.1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/pretrain_2task-llava-$GRAPH_TOWER-$MODEL_VERSION-pretrain_lora \
    --num_train_epochs 5 \
    --per_device_train_batch_size 14 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 8e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to none



