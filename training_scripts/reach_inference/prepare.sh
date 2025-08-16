# -------- paths --------
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)"   # repo root
OVERLAY_REL="training_scripts/reach_inference/overlay"          
OVERLAY="$ROOT/$OVERLAY_REL"                                    
REPO="$HOME/ext/smc_rnns"                                       
TARGET="$REPO/data_untracked/dandi"

# -------- clone --------
git clone https://github.com/mackelab/smc_rnns.git "$REPO"

# -------- conda env from YAML --------
conda env create -f "$REPO/smc_rnn_env.yml" -n smc_rnn_env

# -------- DANDI in env --------
mkdir -p "$TARGET"
conda run -n smc_rnn_env python -m pip install -U dandi
conda run -n smc_rnn_env dandi download DANDI:000128/0.220113.0400 \
  -o "$TARGET" --existing skip --download assets -J 8

# -------- make tensors --------
(
  cd "$REPO" || exit 1
  conda run -n smc_rnn_env python train_scripts/macaque_reach/make_tensors_conditioning.py --binsize 20
  # conda run -n smc_rnn_env python train_scripts/macaque_reach/make_tensors_nlb.py --binsize 20
)