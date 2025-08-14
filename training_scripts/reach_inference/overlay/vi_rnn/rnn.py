import os, importlib
_impl = os.getenv("RNN_IMPL", "rnn")  # choose "tbrnn" or "hornn" (or "rnn" to fall back)
_mod = importlib.import_module("." + _impl, __package__)
for k in dir(_mod):
    if not k.startswith("_"):
        globals()[k] = getattr(_mod, k)
__all__ = [k for k in globals() if not k.startswith("_")]