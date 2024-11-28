#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2113}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-1}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/MiniLLM"}
CKPT="Qwen/Qwen2.5-0.5B"
#CKPT_NAME=${4-"Qwen2.5-7B"}
#CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
PEFT_CKPT_NAME=${5-"0.5B_7B/45950"}
PEFT_CKPT="${BASE_PATH}/results/qwen2.5/train/distillm/mlqa_en_1shot/${PEFT_CKPT_NAME}/"
# data
DATA_NAMES="mlqa"
DATA_DIR="${BASE_PATH}/data/mlqa_1shot_qwen2/english/"
#DATA_DIR="${BASE_PATH}/processed_data_pythia70M/squad_v2/prompt/pythia/"
# hp
EVAL_BATCH_SIZE=8
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen2.5/eval/eval_distill_mlqa_en_1shot/eval_main_0.5B_7B"
TYPE="eval_main"

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
#OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type qwen"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length 3096"
OPTS+=" --max-prompt-length 2600"
# runtime
OPTS+=" --do-eval"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
# lora
OPTS+=" --peft lora"
OPTS+=" --peft-name ${PEFT_CKPT_NAME}"
OPTS+=" --peft-path ${PEFT_CKPT}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --type ${TYPE}"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
#OPTS+=" --repetition-penalty 1.2"


export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/evaluate.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
