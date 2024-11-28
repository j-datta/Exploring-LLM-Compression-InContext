#--model-path ${BASE_PATH}/checkpoints/pythia-70m \
BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

echo "BASE_PATH: ${BASE_PATH}"  /home/IAIS/jdatta/distillm-new/tools/process_squad_pythia.py
echo "Running: python3 ${BASE_PATH}/tools/process_squad_pythia.py ..."

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_squad_pythia.py \
    --data-dir ${BASE_PATH}/data/squad_1shot/ \
    --processed-data-dir ${BASE_PATH}/processed_data_pythia70m/squad_1shot/prompt \
    --model-path EleutherAI/pythia-70m \
    --data-process-workers 32 \
    --max-prompt-length 2048 \
    --only-prompt \
    --model-type pythia

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_squad_pythia.py \
    --data-dir ${BASE_PATH}/data/squad_1shot/ \
    --processed-data-dir ${BASE_PATH}/processed_data_pythia70m/squad_1shot/full \
    --model-path EleutherAI/pythia-70m \
    --data-process-workers 32 \
    --max-prompt-length 2048 \
    --model-type pythia
