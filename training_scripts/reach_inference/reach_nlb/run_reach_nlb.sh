#!/usr/bin/env bash
set -euo pipefail

# ---------- paths ----------
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)"
OVERLAY="$ROOT/training_scripts/reach_inference/overlay"   # overlay ROOT
REPO="$HOME/ext/smc_rnns"

# ---------- scheduler ----------
N=30
START=1
MAX_JOBS_PER_GPU=3
GPUS=(0 )

# ---------- configs ----------
# "DIM_Z,rnn"  OR  "RNN_DIM,TBRNN_DIM,hornn"
CONFIGS=(
  "36,rnn"
  "34,1,hornn"
  "35,1,hornn"
  "36,1,hornn"
)

# ---------- helpers ----------
declare -A RUNNING PID_GPU
for g in "${GPUS[@]}"; do RUNNING[$g]=0; done

reap_one() {
  if wait -p wpid -n 2>/dev/null; then
    local gpu=${PID_GPU[$wpid]:-}
    [[ -n $gpu ]] && RUNNING[$gpu]=$(( RUNNING[$gpu]-1 )) && unset PID_GPU[$wpid]
    return 0
  fi
  for p in "${!PID_GPU[@]}"; do
    if ! kill -0 "$p" 2>/dev/null; then
      local gpu=${PID_GPU[$p]}
      RUNNING[$gpu]=$(( RUNNING[$gpu]-1 ))
      unset PID_GPU[$p]
      return 0
    fi
  done
  sleep 1
  return 1
}

pick_gpu() {
  while :; do
    local best="" best_load=9999
    for g in "${GPUS[@]}"; do
      local load=${RUNNING[$g]}
      if (( load < MAX_JOBS_PER_GPU )) && (( load < best_load )); then
        best="$g"; best_load=$load
      fi
    done
    [[ -n $best ]] && { echo "$best"; return; }
    reap_one
  done
}

# ---------- run ----------
mkdir -p "$ROOT/outputs/reach_inference/reach_nlb"
PIDFILE="$ROOT/outputs/reach_inference/reach_nlb/pids.txt"
: > "$PIDFILE"

(
  cd "$REPO" || exit 1

  for ((i = START + N - 1; i >= START; i--)); do
    for cfg in "${CONFIGS[@]}"; do
      IFS=',' read -r -a F <<<"$cfg"

      case ${#F[@]} in
        2)  DIM_Z=$((10#${F[0]})); MODEL=${F[1]}; TAG="r_${DIM_Z}" ;;
        3)  RNN_DIM=$((10#${F[0]})); TBRNN_DIM=$((10#${F[1]})); MODEL=${F[2]}
            DIM_Z=$(( RNN_DIM + TBRNN_DIM )); TAG="r_${RNN_DIM}_r_${TBRNN_DIM}" ;;
        *)  echo "Bad CONFIG: $cfg" >&2; exit 1 ;;
      esac

      GPU_ID=$(pick_gpu)
      RUN=$(printf "%03d" "$i")
      RUN_NAME="reach_nlb/${TAG}_${MODEL}/$RUN"
      RUN_DIR_REPO="$HOME/ext/runs/$RUN_NAME"

      if [[ -d "$RUN_DIR_REPO" ]]; then
        echo "[SKIP] exist: $RUN_NAME"
        continue
      fi

      LOG_DIR="$ROOT/outputs/reach_inference/reach_nlb/$MODEL/$TAG"
      mkdir -p "$LOG_DIR"

      # Build command as ONE array starting with `env` (no line breaks)
      cmd=(env
           OVERLAY="$OVERLAY" REPO="$REPO" TRAIN_SCRIPT="reach_nlb"
           RNN_IMPL="$MODEL"
           CUDA_VISIBLE_DEVICES="$GPU_ID")

      # hornn-only envs
      if [[ "$MODEL" == hornn ]]; then
        cmd+=(RNN_DIM="$RNN_DIM" TBRNN_DIM="$TBRNN_DIM")
      fi

      # append program & Hydra overrides
      cmd+=(conda run -n smc_rnn_env
           python "$ROOT/training_scripts/reach_inference/overlay/with_overlay.py"
           --run_name "$RUN_NAME"
           dataset=mc_maze_20ms_val_nlb
           vae_params=default "vae_params.dim_z=$DIM_Z")

      # header then launch
      {
        echo "[LAUNCH] i=$i cfg='$cfg' GPU=$GPU_ID RUN=$RUN MODEL=$MODEL DIM_Z=$DIM_Z ${RNN_DIM:+RNN_DIM=$RNN_DIM }${TBRNN_DIM:+TBRNN_DIM=$TBRNN_DIM }"
      } >> "$LOG_DIR/output_${RUN}.log"

      "${cmd[@]}" >> "$LOG_DIR/output_${RUN}.log" 2>&1 &

      pid=$!
      echo "$pid" >> "$PIDFILE"
      PID_GPU[$pid]=$GPU_ID
      RUNNING[$GPU_ID]=$(( RUNNING[$GPU_ID]+1 ))
    done
  done

  # drain
  while ((${#PID_GPU[@]})); do reap_one; done
)

python "$ROOT/training_scripts/reach_inference/reach_nlb/collect_data.py"
