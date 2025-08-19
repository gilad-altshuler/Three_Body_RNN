import os, sys, runpy
OVERLAY = os.environ["OVERLAY"]
REPO    = os.environ["REPO"]
# put overlay first, then their repo
sys.path[:0] = [OVERLAY, REPO]
# (optional) visibility
sys.stderr.write(f"[with_overlay] sys.path[:2]={sys.path[:2]}\n")
# run their training module as __main__
runpy.run_module("train_scripts.macaque_reach.train_single_conditioning", run_name="__main__")
