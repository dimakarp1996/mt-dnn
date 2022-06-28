#!/bin/bash
BATCH_SIZE=32
tstr=$(date +"%FT%H%M")
SAMPLING=$1
BIN=$2
prefix="mt-dnn-8task--glue"

train_datasets="stsb,rte,qnli,mrpc,sst,cola,qqp,mnli"
test_datasets="stsb,rte,qnli,mrpc,sst,cola,qqp,mnli_matched,mnli_mismatched"
MODEL_ROOT="checkpoints"
BERT_PATH="mt_dnn_models/bert_model_base_uncased.pt"
DATA_DIR="data/canonical_data/bert_uncased_lower"
#DEEPPAVLOV_CHECKPOINT="checkpoints/mt-dnn-8task--glue_adamax_answer_opt1_gc0_ggc1_2022-06-09T1727/model_4.pt"

answer_opt=1
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="2e-5"
num_train_epochs=5


model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}_${SAMPLING}_${BIN}"
log_file="${model_dir}/log.log"
echo $BIN
if [ $BIN -eq 0 ]; then
    echo "Sampling ${SAMPLING} NOT BY BINS"
    python3 train.py --data_dir ${DATA_DIR} --sampling ${SAMPLING} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on --epochs 5 --max_seq_len 128
else
    echo "Sampling ${SAMPLING} BY BINS"
    python3 train.py  --bin_on --data_dir ${DATA_DIR} --sampling ${SAMPLING} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on --epochs 5 --max_seq_len 128
fi