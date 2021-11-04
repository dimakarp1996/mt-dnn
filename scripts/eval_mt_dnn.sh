#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "train.sh <batch_size> <gpu>"
  exit 1
fi
prefix="mt-dnn-rte"
BATCH_SIZE=$1
gpu=$2
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

train_datasets="mnli,rte,qqp,qnli,mrpc,sst,cola,stsb"
test_datasets="mnli_matched,mnli_mismatched,rte,qqp,qnli,mrpc,sst,cola,stsb"
MODEL_ROOT="checkpoints"
BERT_PATH="mt_dnn_models/bert_model_large_uncased.pt"
DATA_DIR="data/canonical_data/bert_uncased_lower"

answer_opt=1
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="5e-5"

model_dir="checkpoints/mt-dnn-rte_adamax_answer_opt1_gc0_ggc1_2021-09-16T1027"
log_file="${model_dir}/log.log"
python train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on --epochs 0
