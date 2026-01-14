"""DS-RPC Optimizers - DSPy-style optimization via RPC."""

from .teleprompt import RPCTeleprompter
from .vanilla import LabeledFewShot
from .bootstrap_rpc import BootstrapFewShotRPC

__all__ = [
    "RPCTeleprompter",
    "LabeledFewShot",
    "BootstrapFewShotRPC",
]
