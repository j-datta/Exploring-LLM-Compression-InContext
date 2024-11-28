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
CKPT="EleutherAI/pythia-70m"
CKPT_NAME=${4-"pythia-70m"}
#CKPT="${BASE_PATH}/results/pythia/train/distillm/${CKPT_NAME}/"
#CKPT="/home/jongwoo/.cache/huggingface/hub/models--openlm-research--open_llama_7b_v2/snapshots/e5961def23172a2384543940e773ab676033c963/"
PEFT_CKPT_NAME=${5-"pythia-70m_LR4/65150"}
PEFT_CKPT="${BASE_PATH}/results/pythia/train/sft/sft_70m_diffLR_squadv2/sft_70m_LR4/${PEFT_CKPT_NAME}/"
# data
DATA_NAMES="squad_v2"
DATA_DIR="${BASE_PATH}/data/squad_json/"
#DATA_DIR="${BASE_PATH}/processed_data_pythia70M/squad_v2/prompt/pythia/"
# hp
EVAL_BATCH_SIZE=16
# runtime
SAVE_PATH="${BASE_PATH}/results/pythia/eval/eval_sft_squadv2/eval_main_sft_70m"
TYPE="eval_main"

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type pythia"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
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


export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/evaluate.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
