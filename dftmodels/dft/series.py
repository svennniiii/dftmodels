from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from .config import DFTConfig, DFTRange, NormType, WindowType

@dataclass
class DataSeries:
    x: NDArray[np.floating]
    y: NDArray

@dataclass
class SignalSeries(DataSeries):
    def __post_init__(self):
        x = np.asarray(self.x)
        if np.iscomplexobj(x):
            raise ValueError("x must be real-valued")
        self.x = x.astype(float)
        self.y = np.asarray(self.y)

        if self.x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")

    def __len__(self) -> int:
        return len(self.x)

    @property
    def sample_rate(self) -> float:
        if len(self.x) < 2:
            raise ValueError("Need at least 2 samples to determine sample rate")
        return 1.0 / (self.x[1] - self.x[0])

    def copy(self) -> SignalSeries:
        return SignalSeries(
            x=self.x.copy(),
            y=self.y.copy(),
        )

    def calculate_rms(self) -> float:
        return float(np.sqrt(np.mean(np.abs(self.y) ** 2)))

    def calculate_dft(
        self,
        pad: float = 1.0,
        norm: NormType = NormType.CFT,
        window: WindowType = WindowType.RECTANGULAR,
        window_params: None | dict = None,
        dft_range: DFTRange = DFTRange.DOUBLE_SIDED,
    ) -> FourierSeries:
        if len(self) == 0:
            raise ValueError("Signal is empty")

        if dft_range == DFTRange.SINGLE_SIDED and np.iscomplexobj(self.y):
            raise ValueError("Single-sided DFT requires a real-valued signal")

        n = len(self)
        sr = self.sample_rate

        if window_params is None:
            window_params = dict()

        dft_config = DFTConfig(
            number_of_samples=n,
            sample_rate=sr,
            pad=pad,
            window=window,
            window_params=window_params,
            norm_type=norm,
            dft_range=dft_range,
        )

        windowed = self.y * dft_config.window_array
        n_fft = dft_config.number_of_samples_fft
        t_step = 1.0 / sr

        if dft_range == DFTRange.DOUBLE_SIDED:
            amplitude = np.fft.fftshift(np.fft.fft(windowed, n=n_fft))
            frequency = np.fft.fftshift(np.fft.fftfreq(n_fft, t_step))
        else:
            amplitude = np.fft.rfft(windowed, n=n_fft)
            frequency = np.fft.rfftfreq(n_fft, t_step)

        if norm == NormType.PSD:
            amplitude = np.abs(amplitude) ** 2
        elif norm == NormType.ASD_ABS:
            amplitude = np.abs(amplitude)

        amplitude = amplitude * dft_config.norm_factor

        return FourierSeries(
            x=frequency,
            y=amplitude,
            dft_config=dft_config,
        )


@dataclass
class FourierSeries(DataSeries):
    dft_config: DFTConfig

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y)

        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")

    def __len__(self) -> int:
        return len(self.x)

    def copy(self) -> FourierSeries:
        return FourierSeries(
            x=self.x.copy(),
            y=self.y.copy(),
            dft_config=self.dft_config.copy(),
        )

    @property
    def real(self) -> FourierSeries:
        result = self.copy()
        result.y = np.real(self.y).copy()
        return result

    @property
    def imag(self) -> FourierSeries:
        result = self.copy()
        result.y = np.imag(self.y).copy()
        return result

    @property
    def abs(self) -> FourierSeries:
        result = self.copy()
        result.y = np.abs(self.y).copy()
        return result

    def calculate_integral(
        self,
        f_min: float | None = None,
        f_max: float | None = None,
    ):
        """Integrate the spectrum using the rectangular (midpoint) rule.

        The rectangular rule is exact for DFT spectra: Parseval's theorem
        holds precisely as ``df * sum(PSD) == signal_power``.

        When f_min/f_max are given, boundary bins are weighted by their
        fractional overlap with the integration window rather than included
        or excluded wholesale.
        """
        x, y = self.x, self.y

        if len(x) == 0:
            return 0.0
        if len(x) == 1:
            return y[0]

        df = x[1] - x[0]

        if f_min is None and f_max is None and self.dft_config.dft_range == DFTRange.SINGLE_SIDED:
            lo = x[0]
            hi = x[-1]
        else:
            lo = f_min if f_min is not None else x[0] - df / 2
            hi = f_max if f_max is not None else x[-1] + df / 2

        bin_starts = x - df / 2
        bin_ends = x + df / 2

        overlap = np.maximum(0.0, np.minimum(bin_ends, hi) - np.maximum(bin_starts, lo))
        proportions = overlap / df

        return df * np.sum(proportions * y)

    def calculate_idft(self, remove_padding: bool = True, remove_window: bool = True) -> SignalSeries:
        cfg = self.dft_config
        n_fft = cfg.number_of_samples_fft
        n = cfg.number_of_samples

        window_array = cfg.window_array
        if np.any(window_array == 0.0):
            raise ValueError("Zero-valued window is incompatible with IDFT")

        if cfg.norm_type in (NormType.ASD_ABS, NormType.PSD):
            raise ValueError("Cannot compute IDFT of ASD_ABS or PSD data")

        expected_len = n_fft if cfg.dft_range == DFTRange.DOUBLE_SIDED else n_fft // 2 + 1
        if len(self) != expected_len:
            raise ValueError(
                f"Expected {expected_len} frequency bins for IDFT, got {len(self)}"
            )

        amplitude = self.y / cfg.norm_factor

        if cfg.dft_range == DFTRange.DOUBLE_SIDED:
            value = np.fft.ifft(np.fft.ifftshift(amplitude), n=n_fft)
        else:
            value = np.fft.irfft(amplitude, n=n_fft)

        if remove_padding:
            value = value[:n]

        if remove_window:
            value[:n] = value[:n] / window_array

        sr = cfg.sample_rate
        n_out = n if remove_padding else n_fft
        time = np.linspace(0, n_out / sr, n_out, endpoint=False)

        return SignalSeries(x=time, y=value)

    def convert_to_psd(self) -> FourierSeries:
        if self.dft_config.norm_type not in (NormType.ASD, NormType.ASD_ABS):
            raise ValueError("convert_to_psd requires ASD or ASD_ABS normalization")

        cfg = DFTConfig(
            number_of_samples=self.dft_config.number_of_samples,
            sample_rate=self.dft_config.sample_rate,
            pad=self.dft_config.pad,
            window=self.dft_config.window,
            window_params=self.dft_config.window_params,
            norm_type=NormType.PSD,
            dft_range=self.dft_config.dft_range,
        )
        
        return FourierSeries(
            x=self.x.copy(),
            y=np.abs(self.y) ** 2,
            dft_config=cfg,           
        )

    def convert_to_asd(self) -> FourierSeries:
        if self.dft_config.norm_type != NormType.PSD:
            raise ValueError("convert_to_asd requires PSD normalization")

        cfg = DFTConfig(
            number_of_samples=self.dft_config.number_of_samples,
            sample_rate=self.dft_config.sample_rate,
            pad=self.dft_config.pad,
            window=self.dft_config.window,
            window_params=self.dft_config.window_params,
            norm_type=NormType.ASD_ABS,
            dft_range=self.dft_config.dft_range,
        )
        return FourierSeries(
            x=self.x.copy(),
            y=np.sqrt(np.abs(self.y)),
            dft_config=cfg,
        )