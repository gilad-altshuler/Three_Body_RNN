import os, sys, runpy
OVERLAY      = os.environ["OVERLAY"]
REPO         = os.environ["REPO"]
TRAIN_SCRIPT = os.environ["TRAIN_SCRIPT"]
# put overlay first, then their repo
sys.path[:0] = [OVERLAY, REPO]
# (optional) visibility
sys.stderr.write(f"[with_overlay] sys.path[:2]={sys.path[:2]}\n")
# run their training module as __main__
if TRAIN_SCRIPT == "reach_condition":
    runpy.run_module("train_scripts.macaque_reach.train_single_conditioning", run_name="__main__")
elif TRAIN_SCRIPT == "reach_nlb":
    runpy.run_module("train_scripts.macaque_reach.train_single_nlb", run_name="__main__")
else:
    sys.exit(f"Unknown TRAIN_SCRIPT: {TRAIN_SCRIPT}")

