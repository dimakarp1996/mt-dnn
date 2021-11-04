#!/bin/bash
prefix="mt-dnnbasecased"
BATCH_SIZE=16
gpu=9
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")
#tstr="2021-09-16T1027"
train_datasets="mnli,rte,qqp,qnli,mrpc,sst,cola,stsb,wnli"
test_datasets="rte,mnli_matched,mnli_mismatched,qqp,qnli,mrpc,sst,cola,stsb,wnli"
MODEL_ROOT="checkpoints"
BERT_PATH="mt_dnn_models/bert_model_base_uncased.pt"
DATA_DIR="data/canonical_data/bert-base-uncased"

answer_opt=1
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="5e-5"

model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"
python train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on --batch_size_eval 1 --epoch 25
