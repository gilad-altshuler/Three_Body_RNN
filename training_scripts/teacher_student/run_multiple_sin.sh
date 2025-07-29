#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_PATH")")")"
N=30  # Total number of jobs    
START=1 
MAX_JOBS_PER_GPU=4 
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
MAX_CONCURRENT=$((MAX_JOBS_PER_GPU * NUM_GPUS))
TASK_NAME="sin_task"
RANKS=(1 6)

mkdir -p "$ROOT/outputs/teacher_student/sin_task"  # ensure output directory exists

for i in $(seq $START $((START + N - 1))); do
  while [ $(jobs -r | wc -l) -ge $MAX_CONCURRENT ]; do
    sleep 5
  done

  GPU_IDX=$(( (i - 1) % NUM_GPUS ))
  GPU_ID=${GPUS[$GPU_IDX]}

  CUDA_VISIBLE_DEVICES=$GPU_ID nohup python \
    "$ROOT/training_scripts/teacher_student/train_teacher_student.py" \
    --run_name "${TASK_NAME}/$(printf "%03d" $i)" --t_rank 2 --ranks "${RANKS[@]}" --rates --w_out_bias \
    > "$ROOT/outputs/teacher_student/sin_task/output_$i.log" 2>&1 &
done

wait
