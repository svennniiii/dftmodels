from importlib.metadata import version
from .stats import cramer_rao_bound

__version__ = version("dftmodels")
from .dft import (
    DFTConfig,
    NormType,
    DFTRange,
    WindowType,
    DFTCorrection,
    DFTCorrectionMode,
    SignalSeries,
    FourierSeries,
)
from .models import (
    FourierModelBase,
    Sinusoid,
    SineFourier,
    CompositeModel,
    ModelBase,
)

__all__ = [
    "__version__",
    # Stats
    "cramer_rao_bound",
    # DFT
    "DFTConfig",
    "NormType",
    "DFTRange",
    "WindowType",
    "DFTCorrection",
    "DFTCorrectionMode",
    "SignalSeries",
    "FourierSeries",
    # Models
    "FourierModelBase",
    "ModelBase",
    "Sinusoid",
    "SineFourier",
    "CompositeModel",
]
