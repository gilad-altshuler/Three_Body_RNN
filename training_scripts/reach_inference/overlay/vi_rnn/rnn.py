import os, importlib

_impl = os.getenv("RNN_IMPL", "rnn")  # "tbrnn" | "hornn" | "rnn"

# Map to modules inside this overlay package
target = ".orig_rnn" if _impl == "rnn" else f".{_impl}"
_mod = importlib.import_module(target, __package__)

# Re-export
for k in dir(_mod):
    if not k.startswith("_"):
        globals()[k] = getattr(_mod, k)
__all__ = [k for k in globals() if not k.startswith("_")]