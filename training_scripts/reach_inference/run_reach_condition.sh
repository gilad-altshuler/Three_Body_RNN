# -------- paths --------
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)"   # repo root
OVERLAY_REL="training_scripts/reach_inference/overlay"          
OVERLAY="$ROOT/$OVERLAY_REL"                                    
REPO="$HOME/ext/smc_rnns"                                       
TARGET="$REPO/data_untracked/dandi"

# -------- Run the directory using the "rnn.py" file --------
(
  cd "$REPO" || exit 1
  RNN_IMPL=hornn PYTHONPATH="$OVERLAY:$REPO:$PYTHONPATH" \
  conda run -n smc_rnn_env python -m train_scripts.macaque_reach.train_single_conditioning \
    --run_name reach_conditioning/001
)