#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_PATH")")")"
N=30  # Total number of jobs
START=1
MAX_JOBS_PER_GPU=2
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
MAX_CONCURRENT=$((MAX_JOBS_PER_GPU * NUM_GPUS))

mkdir -p "$ROOT/outputs/multifate_inference" # ensure output directory exists

for i in $(seq $START $((START + N - 1))); do
  while [ $(jobs -r | wc -l) -ge $MAX_CONCURRENT ]; do
    sleep 5
  done

  GPU_IDX=$(( (i - 1) % NUM_GPUS ))
  GPU_ID=${GPUS[$GPU_IDX]}

  CUDA_VISIBLE_DEVICES=$GPU_ID nohup python \
    "$ROOT/training_scripts/multifate_inference/train_multifate_inference.py" \
    --run_name "$(printf "%03d" $i)" \
    > "$ROOT/outputs/multifate_inference/output_$i.log" 2>&1 &
done

wait

# Now call collect data after all the jobs are finished

python "$ROOT/training_scripts/multifate_inference/collect_data.py"