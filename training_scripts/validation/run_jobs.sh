#!/bin/bash

N=30  # Total number of jobs    
START=1 
MAX_JOBS_PER_GPU=4 
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
MAX_CONCURRENT=$((MAX_JOBS_PER_GPU * NUM_GPUS))
TASK_NAME="K_Bit_Flipflop"
RANKS=(1 6)

mkdir -p Three_Body_RNN/outputs/validation  # ensure output directory exists

for i in $(seq $START $((START + N - 1))); do
  while [ $(jobs -r | wc -l) -ge $MAX_CONCURRENT ]; do
    sleep 5
  done

  GPU_IDX=$(( (i - 1) % NUM_GPUS ))
  GPU_ID=${GPUS[$GPU_IDX]}

  CUDA_VISIBLE_DEVICES=$GPU_ID nohup /home/gilada/miniconda3/envs/TBRNN_env/bin/python \
    /home/gilada/Three_Body_RNN/training_scripts/validation/train_validation.py \
    --run_name "${TASK_NAME}/$(printf "%03d" $i)" \
    > "Three_Body_RNN/outputs/validation/output_$i.log" 2>&1 &
done

wait

# Now call collect.py after all the jobs are finished

/home/gilada/miniconda3/envs/TBRNN_env/bin/python /home/gilada/Three_Body_RNN/training_scripts/validation/collect.py