import os, importlib

_impl = os.getenv("RNN_IMPL", "rnn")  # "hornn" or "rnn"

# Map to modules inside this overlay package
target = ".hornn" if _impl == "hornn" else ".orig_rnn"
_mod = importlib.import_module(target, __package__)

# Re-export
for k in dir(_mod):
    if not k.startswith("_"):
        globals()[k] = getattr(_mod, k)
__all__ = [k for k in globals() if not k.startswith("_")]