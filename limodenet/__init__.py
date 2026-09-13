"""
LIMODENet (LinearMix-ODENet): an information-preserving, softmax-/QKV-free
convolutional backbone for restoring channel-degraded satellite imagery
under a hard onboard/neuromorphic-deployment constraint.

    from limodenet import family, recon, snn, snn_recon

See the paper and README for details. `snn`/`snn_recon` require `snntorch`.
"""
__version__ = "1.0.0"

from . import family, recon  # noqa: F401  (no extra deps)

__all__ = ["family", "recon"]

try:
    from . import snn, snn_recon  # noqa: F401  (require snntorch)
    __all__ += ["snn", "snn_recon"]
except ImportError:
    pass
