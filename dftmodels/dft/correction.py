from dataclasses import dataclass
from enum import IntFlag, auto


class DFTCorrectionMode(IntFlag):
    NONE          = 0
    BASELINE_ONLY = auto()
    SAMPLING_ONLY = auto()
    WINDOW_ONLY   = auto()

    WINDOW        = WINDOW_ONLY
    BASELINE      = WINDOW_ONLY | BASELINE_ONLY
    ALL           = WINDOW_ONLY | BASELINE_ONLY | SAMPLING_ONLY


@dataclass(frozen=True)
class DFTCorrection:
    mode: DFTCorrectionMode
    order: int = 10
