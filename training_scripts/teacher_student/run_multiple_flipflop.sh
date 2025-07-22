#!/bin/bash

N=30
START=1
MAX_JOBS_PER_GPU=3
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
KS=(1 2 3)
RANKS=(1 6)

mkdir -p Three_Body_RNN/outputs/teacher_student

# Track jobs using PID->GPU assignment
declare -A JOB_GPU_COUNT

# Function to count active jobs on a given GPU
count_gpu_jobs() {
  local gpu_id=$1
  local count=0
  for pid in "${!JOB_GPU_COUNT[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      [[ "${JOB_GPU_COUNT[$pid]}" == "$gpu_id" ]] && ((count++))
    else
      unset JOB_GPU_COUNT["$pid"]  # Clean up finished jobs
    fi
  done
  echo "$count"
}

for K in "${KS[@]}"; do
  TASK_NAME="K_Bit_Flipflop_task"
  LOG_NAME="${K}_Bit_Flipflop_task"

  for i in $(seq $START $((START + N - 1))); do

    while true; do
      for gpu in "${GPUS[@]}"; do
        count=$(count_gpu_jobs "$gpu")
        if [ "$count" -lt "$MAX_JOBS_PER_GPU" ]; then
          GPU_ID=$gpu
          break 2
        fi
      done
      sleep 5
    done

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

    pid=$!
    JOB_GPU_COUNT["$pid"]=$GPU_ID
    echo "Started job $i on GPU $GPU_ID (pid $pid)"
  done
done

wait
