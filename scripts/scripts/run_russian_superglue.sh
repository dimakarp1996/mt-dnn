#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "train.sh <batch_size> <gpu>"
  exit 1
fi
prefix="rusuperglue"
BATCH_SIZE=$1
gpu=$2
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

train_datasets="lidirus,rcb,parus,muserc,terra,russe,rwsd,danetqa,rucos"
test_datasets="lidirus,rcb,parus,muserc,terra,russe,rwsd,danetqa,rucos"


MODEL_ROOT="checkpoints_rus"
BERT_PATH="multilingual_base_cased"
DATA_DIR="data/canonical_data/russian_superglue_for_multilingual"
RUBERT_PATH="rubert_base_cased"
RUBERT_DATA_DIR="data/canonical_data/russian_superglue_for_rubert"
TASK_DEF="../experiments/russian_superglue/glue_task_def.yml"

answer_opt=1
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="5e-5"

model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"
python train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on --task_def ${TASK_DEF}
