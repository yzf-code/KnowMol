#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

GRAPH_TOWER="moleculestm"
if [ "$GRAPH_TOWER" == "graphmvp" ]; then
    INIT_CHECKPOINT_GNN="./checkpoints/graphmvp.pth"
elif [ "$GRAPH_TOWER" == "moleculestm" ]; then
    INIT_CHECKPOINT_GNN="./checkpoints/MoleculeSTM/molecule_model.pth"
else
    echo "Not supported graph tower"
fi

pretrain_dataset=new_arch_and_token_my_dataset_100k_r_64_a_256_dp01

FINETUNED_WEIGHT=/data/yangzaifei/backup_checkpoints/$pretrain_dataset/pretrain_2task-llava-moleculestm-vicuna-v1-3-7b-pretrain_lora

CHECKPOINT_FOLDER_PREFIX="/data/yangzaifei/backup_checkpoints/"
TASK="reagent_pred"

deepspeed --include localhost:2,3,4 llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path /home/yangzaifei/MolLLM/data/Mol-Instructions/processed_Molecule-oriented_Instructions/reagent_prediction_train.json \
    --data_type reagent_pred \
    --graph_tower $GRAPH_TOWER \
    --init_checkpoint $INIT_CHECKPOINT_GNN \
    --add_molecule_tokens True \
    --previous_finetune_weights $FINETUNED_WEIGHT \
    --lora_dropout 0.1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/$pretrain_dataset/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora_full_graph\
    --num_train_epochs 15 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 8e-5 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to none
