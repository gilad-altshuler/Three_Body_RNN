#!/bin/bash

N=30  # Total number of jobs    
START=1 
MAX_JOBS_PER_GPU=3 
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
MAX_CONCURRENT=$((MAX_JOBS_PER_GPU * NUM_GPUS))
KS=(1 2 3)
RANKS=(1 6)

mkdir -p Three_Body_RNN/outputs/teacher_student  # ensure output directory exists

for K in "${KS[@]}"; do
  TASK_NAME="K_Bit_Flipflop_task"
  LOG_NAME="${K}_Bit_Flipflop_task"

  for i in $(seq $START $((START + N - 1))); do
    while [ $(jobs -r | wc -l) -ge $MAX_CONCURRENT ]; do
      sleep 5
    done

    GPU_IDX=$(( (i - 1) % NUM_GPUS ))
    GPU_ID=${GPUS[$GPU_IDX]}

    LOG_DIR="Three_Body_RNN/outputs/teacher_student/${LOG_NAME}"
    mkdir -p "$LOG_DIR"

    CUDA_VISIBLE_DEVICES=$GPU_ID nohup /home/gilada/miniconda3/envs/TBRNN_env/bin/python \
      /home/gilada/Three_Body_RNN/training_scripts/teacher_student/train_teacher_student.py \
      --run_name "${TASK_NAME}/${K}/$(printf "%03d" $i)" \
      --t_rank "$K" \
      --ranks "${RANKS[@]}" \
      --input_size "$K" \
      --output_size "$K" \
      --lr 1e-2 1e-3 \
      --lint_lr 1e-2 5e-3 \
      --lint_epochs 80000 \
      --sched_epochs 20000 \
      > "${LOG_DIR}/output_${i}.log" 2>&1 &
  done
done

wait
# End of script