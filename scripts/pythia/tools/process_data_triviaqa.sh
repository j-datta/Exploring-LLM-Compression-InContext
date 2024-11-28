BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

echo "BASE_PATH: ${BASE_PATH}"  /home/IAIS/jdatta/distillm-new/tools/process_triviaqa_pythia.py
echo "Running: python3 ${BASE_PATH}/tools/process_triviaqa_pythia.py ..."

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_triviaqa_pythia.py \
    --data-dir ${BASE_PATH}/data/trivia_qa/rc_nocontext/ \
    --processed-data-dir ${BASE_PATH}/processed_data_pythia70m/trivia_qa/prompt \
    --model-path ${BASE_PATH}/checkpoints/pythia-70m \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --only-prompt \
    --model-type pythia

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_triviaqa_pythia.py \
    --data-dir ${BASE_PATH}/data/trivia_qa/rc_nocontext/ \
    --processed-data-dir ${BASE_PATH}/processed_data_pythia70m/trivia_qa/full \
    --model-path ${BASE_PATH}/checkpoints/pythia-70m \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --model-type pythia