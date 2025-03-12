# Reference Running: bash train/sft.sh
# {'train_runtime': 5268.8407, 'train_samples_per_second': 0.949, 'train_steps_per_second': 0.119, 'train_loss': 0.1172730620391667, 'epoch': 5.0}
uid="$(date +%Y%m%d_%H%M%S)"
# base_model="Qwen/Qwen2.5-32B-Instruct"
lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4 # -> the same training pipe as slurm_training
micro_batch_size=1 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file train/deepspeed_zero3.yaml \
    train/sft.py \
    --block_size=16384 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --max_steps=${max_steps} \
    --train_file_path="simplescaling/s1K_tokenized" \
    --model_name="Qwen/Qwen2.5-32B-Instruct" \
    --warmup_ratio=0.05 \
    --bf16=True \
    --eval_strategy="steps" \
    --eval_steps=50 \
    --logging_steps=1 \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=1e-4 \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="ckpts/s1.1-32B/limo_tokenized_${uid}" \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --save_strategy=no \
    --dataset_text_field="text" \
    --wandb_project="HKSTP" \
    --wandb_entity="junw_huhudawang-zhejiang-university"

