"""
Optional Visdom wrapper.
If the Visdom server is unreachable the entire codebase keeps working —
all viz calls become silent no-ops instead of crashing at import time.
"""
import logging

_logger = logging.getLogger(__name__)
_viz = None
_loss_window = None
_grad_window = None


def get_viz():
    global _viz
    if _viz is None:
        try:
            import visdom
            candidate = visdom.Visdom(raise_exceptions=True)
            _viz = candidate
        except Exception:
            _logger.warning("Visdom server not reachable — visualizations disabled.")
            _viz = _NoopViz()
    return _viz


def get_loss_window():
    global _loss_window
    if _loss_window is None:
        import torch as th
        _loss_window = get_viz().line(
            Y=th.zeros((1)).cpu(),
            X=th.zeros((1)).cpu(),
            opts=dict(xlabel="step", ylabel="Loss", title="Training loss"),
        )
    return _loss_window


def get_grad_window():
    global _grad_window
    if _grad_window is None:
        import torch as th
        _grad_window = get_viz().line(
            Y=th.zeros((1)).cpu(),
            X=th.zeros((1)).cpu(),
            opts=dict(xlabel="step", ylabel="amplitude", title="Gradient norm"),
        )
    return _grad_window


class _NoopViz:
    """Drop-in replacement when Visdom is unavailable."""

    def image(self, *args, **kwargs):
        return None

    def line(self, *args, **kwargs):
        return None

    def update_window_opts(self, *args, **kwargs):
        return None
