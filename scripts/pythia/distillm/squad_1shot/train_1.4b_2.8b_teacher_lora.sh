#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-16}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/MiniLLM"}
CKPT="EleutherAI/pythia-1.4b"
PEFT_CKPT_NAME="pythia-1.4b/39063"
PEFT_CKPT="${BASE_PATH}/results/pythia/train/minillm_init/squad_1shot/sft_1.4b_lora/${PEFT_CKPT_NAME}/"
TEACHER_CKPT="EleutherAI/pythia-2.8b"
TEACHER_PEFT_CKPT_NAME="pythia-2.8b/130210"
TEACHER_PEFT_CKPT="${BASE_PATH}/results/pythia/train/sft/sft_squadv2_1shot/sft_2.8b_lora/${TEACHER_PEFT_CKPT_NAME}/"
# data
DATA_DIR="${BASE_PATH}/processed_data_pythia70m/squad_1shot/full/pythia/"
LM_DATA_DIR="${BASE_PATH}/processed_data_pythia70m/openwebtext/pythia/512/1M/"
# hp
BATCH_SIZE=4
LR=0.0001
#LR=0.0005
GRAD_ACC=1
EVAL_BATCH_SIZE=4
# length
MAX_LENGTH=1800
# runtime
SAVE_PATH="${BASE_PATH}/results/pythia/train/distillm/squad_1shot/1.4b_2.8b"
# seed
SEED=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
#OPTS+=" --ckpt-name ${CKPT_NAME}"
#OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type pythia"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --num-workers 4"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 10"
OPTS+=" --kd-ratio 1.0"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 1500"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# lora
OPTS+=" --peft lora"
OPTS+=" --do-train"
OPTS+=" --peft-name ${PEFT_CKPT_NAME}"
OPTS+=" --peft-path ${PEFT_CKPT}"
OPTS+=" --teacher-peft-name ${TEACHER_PEFT_CKPT_NAME}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_CKPT}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type srkl-aesop"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# GKD
OPTS+=" --student-gen"
OPTS+=" --init-threshold 0.2"
OPTS+=" --loss-eps 0.2"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}