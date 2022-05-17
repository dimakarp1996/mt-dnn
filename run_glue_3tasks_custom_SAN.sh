#!/bin/bash
prefix="mt-dnn-lite-glue-3task-custom_san"
BATCH_SIZE=32
tstr=$(date +"%FT%H%M")

train_datasets="rte,qnli,mrpc"
test_datasets="rte,qnli,mrpc"
MODEL_ROOT="checkpoints"
BERT_PATH="mt_dnn_models/bert_model_base_uncased.pt"
DATA_DIR="data/canonical_data/bert_uncased_lower"

answer_opt=1
optim="adam"
grad_clipping=0
global_grad_clipping=1
lr="2e-5"
num_train_epochs=5

task_def='experiments/glue/glue_san_task_def.yml'
model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"

python3 train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on --epochs 5 --max_seq_len 128 --task_def ${task_def} --log_per_updates 500 --custom
