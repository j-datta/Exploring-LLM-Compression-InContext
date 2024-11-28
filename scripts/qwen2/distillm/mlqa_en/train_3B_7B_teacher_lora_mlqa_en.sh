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
CKPT="Qwen/Qwen2.5-3B"
PEFT_CKPT_NAME="Qwen2.5-3B/867"
PEFT_CKPT="${BASE_PATH}/results/qwen2.5/train/minillm_init/mlqa_en/sft_3B_lora/${PEFT_CKPT_NAME}/"
#TEACHER_CKPT="Qwen/Qwen2.5-7B"
TEACHER_CKPT_NAME="Qwen2.5-7B"
TEACHER_CKPT="${BASE_PATH}/checkpoints/${TEACHER_CKPT_NAME}/"
TEACHER_PEFT_CKPT_NAME="Qwen2.5-7B/1440"
TEACHER_PEFT_CKPT="${BASE_PATH}/results/qwen2.5/train/sft/mlqa_en/sft_7B_lora/${TEACHER_PEFT_CKPT_NAME}/"
# data
DATA_DIR="${BASE_PATH}/processed_data_test/mlqa_en/full/qwen/"
LM_DATA_DIR="${BASE_PATH}/processed_data_test/openwebtext/qwen/512/1M/"
# hp
BATCH_SIZE=2
LR=0.00007
GRAD_ACC=4
EVAL_BATCH_SIZE=2
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen2.5/train/distillm/mlqa_en/3B_7B"
# seed
SEED=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
#OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type qwen"
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
OPTS+=" --max-prompt-length 256"
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