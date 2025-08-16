# -------- paths --------
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)"   # repo root
OVERLAY_REL="training_scripts/reach_inference/overlay"          
OVERLAY="$ROOT/$OVERLAY_REL"                                    
REPO="$HOME/ext/smc_rnns"                                       
TARGET="$REPO/data_untracked/dandi"

# -------- Run the directory using the "rnn.py" file --------
(
  cd "$REPO" || exit 1
  N=30  # Total number of jobs
  START=1
  MAX_JOBS_PER_GPU=3
  GPUS=(0 1 2 3)
  NUM_GPUS=${#GPUS[@]}
  MAX_CONCURRENT=$((MAX_JOBS_PER_GPU * NUM_GPUS))

  LOG_DIR="$ROOT/outputs/reach_inference/rnn/r_5"
  mkdir -p "$LOG_DIR"

  for i in $(seq "$START" "$((START + N - 1))"); do
    while [ "$(jobs -r | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 5; done

    GPU_IDX=$(( (i - 1) % NUM_GPUS ))
    GPU_ID=${GPUS[$GPU_IDX]}
    RUN=$(printf "%03d" "$i")

    # Env vars BEFORE nohup
    CUDA_VISIBLE_DEVICES="$GPU_ID" RNN_IMPL=rnn PYTHONPATH="$OVERLAY:$REPO:$PYTHONPATH" \
    nohup conda run -n smc_rnn_env \
      python -m train_scripts.macaque_reach.train_single_conditioning \
      --run_name "reach_conditioning/$RUN" \
      vae_params=default vae_params.dim_z=5 \
      > "$LOG_DIR/output_${RUN}.log" 2>&1 &
  done

  wait

)

# Now call collect data after all the jobs are finished

# conda run -n smc_rnn_env python -m "$ROOT/training_scripts/mante_inference/collect_data.py"