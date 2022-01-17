
prefix="mt-dnn-evseevner"
BATCH_SIZE=8
gpu=9
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")
#tstr="2021-09-16T1027"
train_datasets="ner,relations"
test_datasets="ner,relations"
MODEL_ROOT="checkpoints"
BERT_PATH="checkpoints/evseev1/model_9.pt"
DATA_DIR="rubert_base_cased/data_mt_preproc"
config="../rubert_base_cased2/config.json"
answer_opt=1
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr=1e-5
decay=1e-6

model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"
python3 trainevseev.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on --ckpt_config ${config} --batch_size_eval 1 --grad_accumulation_step 4 --epochs 10 --task_def "rubert_base_cased/evseev_task_def.yml" --weight_decay ${decay}
