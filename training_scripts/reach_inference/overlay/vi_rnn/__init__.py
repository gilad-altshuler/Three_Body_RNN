# $OVERLAY/vi_rnn/__init__.py
import os

# Allow vi_rnn.* to resolve from the original repo as a fallback
_repo = os.environ.get("REPO")
if _repo:
    other = os.path.join(_repo, "vi_rnn")
    if os.path.isdir(other) and other not in __path__:
        __path__.append(other)

# (optional) also support pkgutil namespace merging
try:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)
except Exception:
    pass
