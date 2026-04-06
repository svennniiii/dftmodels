from dataclasses import dataclass, replace, field
from enum import Enum
from functools import cached_property

import numpy as np
from numpy.typing import NDArray


class WindowType(str, Enum):
    # Cosine-sum windows
    RECTANGULAR      = "rectangular"
    HAMMING          = "hamming"
    HANN             = "hann"
    DIRICHLET        = "rectangular"  # alias for RECTANGULAR
    BLACKMAN         = "blackman"
    NUTTAL           = "nuttal"
    BLACKMAN_NUTTAL  = "blackman-nuttal"
    BLACKMAN_HARRIS  = "blackman-harris"
    FLAT_TOP         = "flat-top"

    # For Debug Only
    # COS4N            = "cos-4n"
    # COS6N            = "cos-6n"
    # COS8N            = "cos-8n"

    # Other
    EXPONENTIAL_ASYM = "exponential-asymmetric"
    BARTLETT         = "bartlett"



def get_window_array(window: WindowType, n: int, **params) -> NDArray[np.floating]:
    match window:
        case WindowType.RECTANGULAR:
            return np.ones(n)
        case WindowType.HAMMING:
            return 25/46 - 21/46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
        case WindowType.HANN:
            return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
        case WindowType.BLACKMAN:
            return (7938 
                - 9240 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
                + 1430 * np.cos(4 * np.pi * np.arange(n) / (n - 1))
            ) / 18608
        case WindowType.NUTTAL:
            return (355_768 \
                - 487_396 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
                + 144_232 * np.cos(4 * np.pi * np.arange(n) / (n - 1))
                -  12_604 * np.cos(6 * np.pi * np.arange(n) / (n - 1))
            ) / 1_000_000
        case WindowType.BLACKMAN_NUTTAL:
            return (3_635_819 \
                - 4_891_775 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
                + 1_365_995 * np.cos(4 * np.pi * np.arange(n) / (n - 1))
                -   106_411 * np.cos(6 * np.pi * np.arange(n) / (n - 1))
            ) / 10_000_000
        case WindowType.BLACKMAN_HARRIS:
            return (4_243_801 \
                - 4_973_406 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
                + 782_793 * np.cos(4 * np.pi * np.arange(n) / (n - 1))
            ) / 10_000_000
        case WindowType.FLAT_TOP:
            return (215_578_950 \
                - 416_631_580 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
                + 277_263_158 * np.cos(4 * np.pi * np.arange(n) / (n - 1))
                -  83_578_947 * np.cos(6 * np.pi * np.arange(n) / (n - 1))
                +   6_947_368 * np.cos(8 * np.pi * np.arange(n) / (n - 1))
            ) / 1_000_000_000
        case WindowType.BARTLETT:
            return np.bartlett(n)
        # case WindowType.COS6N:
        #     return .5 - .5 * np.cos(6 * np.pi * np.arange(n) / (n - 1))
        # case WindowType.COS8N:
        #     return .5 - .5 * np.cos(8 * np.pi * np.arange(n) / (n - 1))
        case WindowType.EXPONENTIAL_ASYM:
            return np.exp(-np.arange(n) * params.get("alpha", 1.0) / (n - 1))
        case _:
            raise ValueError(f"Unsupported window type: {window}")


class NormType(str, Enum):
    CFT     = "continuous_fourier_transform"
    ASD     = "amplitude_spectral_density"
    ASD_ABS = "amplitude_spectral_density_absolute"
    PSD     = "power_spectral_density"


class DFTRange(str, Enum):
    DOUBLE_SIDED = "double_sided"
    SINGLE_SIDED = "single_sided"


@dataclass(frozen=True)
class DFTConfig:
    number_of_samples: int
    sample_rate: float
    pad: float = 1.0
    window: WindowType = WindowType.RECTANGULAR
    window_params: dict = field(default_factory=dict)  # <--- NEW
    norm_type: NormType = NormType.ASD
    dft_range: DFTRange = DFTRange.DOUBLE_SIDED

    def __post_init__(self):
        if self.number_of_samples <= 0:
            raise ValueError("number_of_samples must be a positive integer")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.pad < 1.0:
            raise ValueError("pad cannot be less than 1.0")
        if self.window not in WindowType:
            raise ValueError(f"Unsupported window type: {self.window}")
        if self.norm_type not in NormType:
            raise ValueError(f"Unsupported norm type: {self.norm_type}")
        if self.dft_range not in DFTRange:
            raise ValueError(f"Unsupported DFT range: {self.dft_range}")

    @cached_property
    def number_of_samples_fft(self) -> int:
        return round(self.number_of_samples * self.pad)

    @cached_property
    def window_array(self) -> NDArray[np.floating]:
        return get_window_array(self.window, self.number_of_samples, **self.window_params)

    @cached_property
    def norm_factor(self) -> float:
        if self.norm_type in (NormType.ASD, NormType.ASD_ABS):
            return self._asd_norm()
        elif self.norm_type == NormType.PSD:
            return self._asd_norm() ** 2
        elif self.norm_type == NormType.CFT:
            return self._cft_norm()
        raise ValueError(f"Unsupported norm type: {self.norm_type}")

    def _cft_norm(self) -> float:
        wa = self.window_array
        if wa[0] == 0.0:
            raise ValueError('Zero-valued window is incompatible with CFT normalization')
        factor = 2.0 if self.dft_range == DFTRange.SINGLE_SIDED else 1.0
        return factor / self.sample_rate / wa[0]

    def _asd_norm(self) -> float:
        wa = self.window_array
        n = self.number_of_samples
        window_rms = np.sqrt(np.sum(wa ** 2) / n)
        factor = np.sqrt(2.0) if self.dft_range == DFTRange.SINGLE_SIDED else 1.0
        return factor / np.sqrt(self.sample_rate * n) / window_rms

    @cached_property
    def frequency_step(self) -> float:
        return self.sample_rate / self.number_of_samples_fft

    @cached_property
    def frequency_min(self) -> float:
        if self.dft_range == DFTRange.SINGLE_SIDED:
            return 0.0
        n = self.number_of_samples_fft
        df = self.frequency_step
        return -(self.sample_rate + df) / 2 if n % 2 == 0 else -self.sample_rate / 2

    @cached_property
    def frequency_max(self) -> float:
        if self.dft_range == DFTRange.SINGLE_SIDED:
            return self.sample_rate / 2
        n = self.number_of_samples_fft
        df = self.frequency_step
        return (self.sample_rate - df) / 2 if n % 2 == 0 else self.sample_rate / 2

    def copy(self) -> 'DFTConfig':
        return replace(self)
