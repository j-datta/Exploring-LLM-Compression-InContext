BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

echo "BASE_PATH: ${BASE_PATH}"  /home/IAIS/jdatta/distillm-new/tools/process_squad_qwen2.py
echo "Running: python3 ${BASE_PATH}/tools/process_squad_qwen2.py ..."

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_squad_qwen2.py \
    --data-dir ${BASE_PATH}/data/squad_1shot_qwen2/ \
    --processed-data-dir ${BASE_PATH}/processed_data_test/squad_1shot/prompt \
    --model-path Qwen/Qwen2.5-0.5B \
    --data-process-workers 32 \
    --max-prompt-length 4096 \
    --only-prompt \
    --model-type qwen

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_squad_qwen2.py \
    --data-dir ${BASE_PATH}/data/squad_1shot_qwen2/ \
    --processed-data-dir ${BASE_PATH}/processed_data_test/squad_1shot/full \
    --model-path Qwen/Qwen2.5-0.5B \
    --data-process-workers 32 \
    --max-prompt-length 4096 \
    --model-type qwen
