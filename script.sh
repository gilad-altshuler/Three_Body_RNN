# ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)"   # repo root
# # Make sure PYTHONPATH points to the OVERLAY **root**, not .../overlay/vi_rnn
# OVERLAY_REL="training_scripts/reach_inference/overlay"
# OVERLAY="$ROOT/$OVERLAY_REL"


# # QUICK IMPORT TEST (should raise immediately if 6+1 != 5)
# env RNN_IMPL=hornn RNN_DIM=6 TBRNN_DIM=1 DIM_Z=5 \
# PYTHONPATH="$OVERLAY:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
# conda run -n smc_rnn_env \
# python -c "import vi_rnn.rnn; print('Imported overlay hornn OK')"


# Must exist:
#   $OVERLAY/vi_rnn/__init__.py
#   $OVERLAY/vi_rnn/rnn.py        (router -> .hornn or .orig_rnn)
#   $OVERLAY/vi_rnn/hornn.py
#   $OVERLAY/vi_rnn/orig_rnn.py   (copy of their vi_rnn/rnn.py with relative imports)

# conda run -n smc_rnn_env python -c "
# import importlib, inspect, sys
# sys.path[:0] = ['$OVERLAY', '$REPO']
# m = importlib.import_module('vi_rnn.rnn')
# print('Resolved to:', inspect.getfile(m))
# "


# OVERLAY_REL="training_scripts/reach_inference/overlay"           
# OVERLAY="$ROOT/$OVERLAY_REL"
# REPO="$HOME/ext/smc_rnns"
# env RNN_IMPL=hornn RNN_DIM=6 TBRNN_DIM=1 \
# PYTHONPATH="$OVERLAY:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
# conda run -n smc_rnn_env python - <<'PY'
# import importlib
# m = importlib.import_module('vi_rnn.rnn')
# print("Resolved to:", m.__file__)
# PY


pgrep -af '/training_scripts/reach_inference/with_overlay\.py'
pgrep -af 'train_scripts\.macaque_reach\.train_single_conditioning'
# graceful
pgrep -f '/training_scripts/reach_inference/with_overlay\.py'                | xargs -r kill -TERM
pgrep -f 'train_scripts\.macaque_reach\.train_single_conditioning'           | xargs -r kill -TERM
sleep 2
# force any leftovers
pgrep -f '/training_scripts/reach_inference/with_overlay\.py'                | xargs -r kill -KILL
pgrep -f 'train_scripts\.macaque_reach\.train_single_conditioning'           | xargs -r kill -KILL
# nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -TERM
# sleep 2
# nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -KILL

