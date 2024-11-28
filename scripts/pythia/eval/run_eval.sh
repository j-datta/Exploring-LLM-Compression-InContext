#!/bin/bash

MASTER_PORT=2040
DEVICE=${1}
ckpt=EleutherAI/pythia-70m

# squad_v2 eval
for seed in 10 20 30 40 50
do
    bash ./scripts/pythia/eval/eval_main_mlqa_en_lora.sh ./ ${MASTER_PORT} 1 ${ckpt} 70m_2.8b/11460 --seed $seed  --eval-batch-size 4
    #bash ./scripts/pythia/eval/eval_main_squadv2_lora.sh ./ ${MASTER_PORT} 1 ${ckpt} pythia-70m_LR4/65150 --seed $seed  --eval-batch-size 4
    #bash ./scripts/pythia/eval/eval_main_squadv2_lora.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size 4
    #CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_self_inst_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B ${ckpt} --seed $seed  --eval-batch-size 4
    #CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_vicuna_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B ${ckpt} --seed $seed  --eval-batch-size 4
    #CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_sinst_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B ${ckpt} --seed $seed  --eval-batch-size 4
    #CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_uinst_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B ${ckpt} --seed $seed  --eval-batch-size 4
done