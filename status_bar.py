"""
Status bar bridge

App sets `_var` in _build_statusbar(), everything else calls status() from here
"""

_var = None


def status(msg: str):
    if _var is not None:
        _var.set(msg)
