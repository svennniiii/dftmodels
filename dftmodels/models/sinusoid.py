from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
from numpy.typing import NDArray
import lmfit

from dftmodels.dft.config import WindowType
from dftmodels.dft.correction import DFTCorrectionMode
from dftmodels.utils.math import sinc
from .base import FourierModelBase, _parse_weights


# ---------------------------------------------------------------------------
# Time-domain model
# ---------------------------------------------------------------------------

class Sinusoid:
    """Time-domain sinusoid: A_i*cos(2π f x) + A_q*sin(2π f x)."""

    PARAM_NAMES = ("amplitude_i", "amplitude_q", "frequency")

    @staticmethod
    def eval(
        x: NDArray,
        amplitude_i: float,
        amplitude_q: float,
        frequency: float,
        decay: float = 0.0,
    ) -> NDArray[np.floating]:
        return (
            amplitude_i * np.cos(2 * np.pi * frequency * x)
            + amplitude_q * np.sin(2 * np.pi * frequency * x)
        ) * np.exp(-decay * x)

    @staticmethod
    def make_params(
        amplitude_i: float = 0.0,
        amplitude_q: float = 0.0,
        frequency: float = 0.0,
        decay: float | None = None,
        frequency_min: float | None = None,
        frequency_max: float | None = None,
        decay_min: float | None = 0.0,
        decay_max: float | None = None,

    ) -> lmfit.Parameters:
        return _sine_params(
            amplitude_i=amplitude_i, amplitude_q=amplitude_q, frequency=frequency, decay=decay,
            frequency_min=frequency_min, frequency_max=frequency_max, decay_min=decay_min,
            decay_max=decay_max,
        )

    @staticmethod
    def amplitude(p: lmfit.Parameters) -> float:
        return float(np.abs(p["amplitude_i"].value + 1j * p["amplitude_q"].value))

    @classmethod  # TODO
    def crb_backup(cls, params: lmfit.Parameters, t: NDArray, noise_std: float) -> dict[str, float]:
        """
        Cramér–Rao bounds (σ) for all parameters and derived quantities.

        Pass the true noise_std for benchmarking against theory; pass a
        residual-estimated σ for practical per-measurement uncertainty.

        Returns
        -------
        dict with keys: amplitude_i, amplitude_q, amplitude, phase, frequency.
        """
        from dftmodels.stats import cramer_rao_bound
        _, Cov = cramer_rao_bound(Sinusoid.eval, params, t, noise_std, return_cov=True)
        free  = [n for n, p in params.items() if p.vary]
        i     = free.index("amplitude_i")
        q     = free.index("amplitude_q")
        f_idx = free.index("frequency")
        ai = float(params["amplitude_i"].value)
        aq = float(params["amplitude_q"].value)
        A  = Sinusoid.amplitude(params)
        return {
            "amplitude_i": float(np.sqrt(Cov[i, i])),
            "amplitude_q": float(np.sqrt(Cov[q, q])),
            "amplitude": float(np.sqrt(
                (ai/A)**2 * Cov[i, i] + (aq/A)**2 * Cov[q, q] + 2*(ai/A)*(aq/A)*Cov[i, q]
            )),
            "phase": float(np.sqrt(
                (aq/A**2)**2 * Cov[i, i] + (ai/A**2)**2 * Cov[q, q]
                + 2*(aq/A**2)*(ai/A**2)*Cov[i, q]
            )),
            "frequency": float(np.sqrt(Cov[f_idx, f_idx])),
        }

    @classmethod
    def crb(cls, params: lmfit.Parameters, t: NDArray, noise_std: float) -> dict[str, float]:
        """
        Cramér–Rao bounds (σ) for all parameters and derived quantities.

        Pass the true noise_std for benchmarking against theory; pass a
        residual-estimated σ for practical per-measurement uncertainty.

        Returns
        -------
        dict with keys: amplitude_i, amplitude_q, amplitude, phase, frequency,
        decay, fwhm.
        """
        from dftmodels.stats import cramer_rao_bound
        _, Cov = cramer_rao_bound(Sinusoid.eval, params, t, noise_std, return_cov=True)
        free  = [n for n, p in params.items() if p.vary]
        i     = free.index("amplitude_i")
        q     = free.index("amplitude_q")
        f_idx = free.index("frequency")        
        ai = float(params["amplitude_i"].value)
        aq = float(params["amplitude_q"].value)
        A  = Sinusoid.amplitude(params)
        if "decay" in free:
            d_idx = free.index("decay")
            σ_decay = float(np.sqrt(Cov[d_idx, d_idx]))
        else:
            σ_decay = np.nan
        return {
            "amplitude_i": float(np.sqrt(Cov[i, i])),
            "amplitude_q": float(np.sqrt(Cov[q, q])),
            "amplitude": float(np.sqrt(
                (ai/A)**2 * Cov[i, i] + (aq/A)**2 * Cov[q, q] + 2*(ai/A)*(aq/A)*Cov[i, q]
            )),
            "phase": float(np.sqrt(
                (aq/A**2)**2 * Cov[i, i] + (ai/A**2)**2 * Cov[q, q]
                + 2*(aq/A**2)*(ai/A**2)*Cov[i, q]
            )),
            "frequency": float(np.sqrt(Cov[f_idx, f_idx])),
            "decay":     σ_decay,
            "fwhm":      σ_decay / np.pi,
        }

    def fit(
        self,
        x: NDArray,
        data: NDArray,
        params: lmfit.Parameters,
        weights: NDArray | float | None = None,
        **minimizer_kwargs,
    ) -> lmfit.minimizer.MinimizerResult:
        n = len(x)
        w = _parse_weights(weights, n)

        def residual(p: lmfit.Parameters) -> NDArray:
            model = self.eval(
                x,
                p["amplitude_i"].value,
                p["amplitude_q"].value,
                p["frequency"].value,
                p["decay"].value,
            )
            diff = (data - model) * np.sqrt(w)
            if np.iscomplexobj(diff):
                return np.concatenate([np.real(diff), np.imag(diff)])
            return diff

        minimizer = lmfit.Minimizer(residual, params)
        return minimizer.minimize(**minimizer_kwargs)


# ---------------------------------------------------------------------------
# Fourier-domain models
# ---------------------------------------------------------------------------

class SineFourier(FourierModelBase):
    """Analytic DFT of a rectangular-windowed sinusoid."""

    PARAM_NAMES = ("amplitude_i", "amplitude_q", "frequency", "decay")

    WINDOW_CORRECTED_BASE: dict[WindowType, Callable]
    BASELINE: dict[WindowType, Callable | None]

    def _compile(self) -> Callable:
        norm = self._dft_config.sample_rate / 2
        t_last = (self._dft_config.number_of_samples - 1) / self._dft_config.sample_rate

        if not DFTCorrectionMode.WINDOW_ONLY in self._dft_correction.mode:
            def compiled(x, amplitude_i, amplitude_q, frequency, decay): # type: ignore
                return _no_window_base(x, amplitude_i, amplitude_q, frequency, decay, t_last) * norm
            return compiled

        window = self.dft_config.window
        if window not in self.WINDOW_CORRECTED_BASE:
            supported = ", ".join(w.name for w in self.WINDOW_CORRECTED_BASE)
            raise ValueError(
                f"Window '{window.value}' is not supported by SineFourier. "
                f"Supported windows: {supported}"
            )
        window_params = self._dft_config.window_params
        base_function = self.WINDOW_CORRECTED_BASE[window]
        return partial(base_function, t_last=t_last, norm=norm, **window_params)

    def _baseline_correction(self, x: NDArray, *p: float) -> NDArray[np.complexfloating]:
        t_last = (self._dft_config.number_of_samples - 1) / self._dft_config.sample_rate
        window_params = self._dft_config.window_params
        window = self.dft_config.window

        if window not in self.BASELINE:
            supported = ", ".join(w.name for w in self.BASELINE)
            raise ValueError(
                f"Baseline correction for window '{window.value}' is not supported by SineFourier. "
                f"Supported windows: {supported}"
            )

        baseline = self.BASELINE[window]

        if baseline is not None:
            return baseline(x, *p, t_last=t_last, **window_params)
        else:
            return super()._baseline_correction(x, *p)

    @staticmethod
    def make_params(
        amplitude_i: float = 0.0,
        amplitude_q: float = 0.0,
        frequency: float = 0.0,
        decay: float | None = None,
        frequency_min: float | None = None,
        frequency_max: float | None = None,
        decay_min: float | None = 0.0,
        decay_max: float | None = None,
        **_,
    ) -> lmfit.Parameters:
        return _sine_params(
            amplitude_i=amplitude_i, amplitude_q=amplitude_q, frequency=frequency, decay=decay,
            frequency_min=frequency_min, frequency_max=frequency_max, decay_min=decay_min,
            decay_max=decay_max,
        )
    
    @staticmethod
    def amplitude(p: lmfit.Parameters) -> float:
        return float(np.abs(p["amplitude_i"].value + 1j * p["amplitude_q"].value))
    
    @classmethod
    def amplitude_stderr(cls, p: lmfit.Parameters) -> float:
        ai_f = float(p["amplitude_i"].value) 
        aq_f = float(p["amplitude_q"].value)
        A    = cls.amplitude(p)
        sai  = p["amplitude_i"].stderr
        saq  = p["amplitude_q"].stderr
        return float(np.sqrt((ai_f / A) ** 2 * sai ** 2 + (aq_f / A) ** 2 * saq ** 2))

    def amplitude_fourier(self, p: lmfit.Parameters) -> float:
        c = self.center(p)
        return np.abs(self.eval(p, np.array([c])))[0]

    def phase(self, p: lmfit.Parameters) -> float:
        return float(np.angle(p["amplitude_i"].value + 1j * p["amplitude_q"].value))

    @classmethod
    def phase_stderr(cls, p: lmfit.Parameters) -> float:
        ai_f = float(p["amplitude_i"].value)
        aq_f = float(p["amplitude_q"].value)
        A    = cls.amplitude(p)
        sai  = p["amplitude_i"].stderr
        saq  = p["amplitude_q"].stderr
        return float(np.sqrt((aq_f / A**2) ** 2 * sai ** 2 + (ai_f / A**2) ** 2 * saq ** 2))

    @staticmethod
    def fwhm(p: lmfit.Parameters) -> float:
        return float(p["decay"].value / np.pi)

    @classmethod
    def fwhm_stderr(cls, p: lmfit.Parameters) -> float:
        return p["decay"].stderr / np.pi

    @staticmethod
    def center(params: lmfit.Parameters) -> float:
        return float(params["frequency"].value)  # type: ignore

    @classmethod
    def center_stderr(cls, p: lmfit.Parameters) -> float:
        return p["frequency"].stderr


# ---------------------------------------------------------------------------
# Pure math functions (no state, all parameters explicit)
# ---------------------------------------------------------------------------

def triangular(x):
    x = np.array(x)
    return np.maximum((1 - np.abs(x)), 0)

def _no_window_base(
    x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
) -> NDArray[np.complexfloating]:
    if decay != 0.0:
        return _lorentzian(x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
            frequency=frequency, decay=decay)

    ampl = amplitude_i - 1j * amplitude_q
    phase = np.angle(ampl)
    f0 = frequency

    decay_term = decay / (2 * np.pi)

    d_pos = (f0 - x) + 1j * decay_term
    d_neg = (f0 + x) - 1j * decay_term

    phase_pos = np.exp(1j * (phase + np.pi * d_pos * t_last))
    phase_neg = np.exp(-1j * (phase + np.pi * d_neg * t_last))
    
    return t_last * (
            phase_pos * abs(ampl) * np.sqrt(triangular(d_pos * t_last))
            + phase_neg * abs(ampl) * np.sqrt(triangular(d_neg * t_last))
    )  # type: ignore 


def _rect_base(
    x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    norm: float,
) -> NDArray[np.complexfloating]:
    ampl = amplitude_i - 1j * amplitude_q
    phase = np.angle(ampl)
    f0 = frequency

    decay_term = decay / (2 * np.pi)

    d_pos = (f0 - x) + 1j * decay_term
    d_neg = (f0 + x) - 1j * decay_term

    phase_pos = np.exp(1j * (phase + np.pi * d_pos * t_last))
    phase_neg = np.exp(-1j * (phase + np.pi * d_neg * t_last))
    
    return t_last * norm * (
            phase_pos * abs(ampl) * np.sinc(d_pos * t_last)
            + phase_neg * abs(ampl) * np.sinc(d_neg * t_last)
    )  # type: ignore 

def _rect_baseline(
    x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    norm: float=1.0,
) -> NDArray[np.complexfloating]:
    last = Sinusoid.eval(np.array([t_last]), amplitude_i, amplitude_q, frequency, decay)[0]
    last_term = last * np.exp(-2j * np.pi * x * t_last)
    return (amplitude_i + last_term) * 0.5 * norm


def _hamming_base(
    x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    norm: float,
) -> NDArray[np.complexfloating]:
    rect =  _rect_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last, norm=norm)
    cos2n = _cos2n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)
    
    return (4.0 * rect + 42.0 * cos2n) / 46.0
    

def _cos2n_base(
    x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    norm: float,
) -> NDArray[np.complexfloating]:
    ampl = amplitude_i - 1j * amplitude_q
    phase = np.angle(ampl)
    f0 = frequency
    d = 1.0 / t_last

    decay_term = decay / (2 * np.pi)

    d_pos = (f0 - x) + 1j * decay_term
    d_neg = (f0 + x) - 1j * decay_term

    denom_pos = d_pos ** 2 - d ** 2 
    denom_neg = d_neg ** 2 - d ** 2

    phase_pos = np.exp(1j * (phase + np.pi * d_pos * t_last))
    phase_neg = np.exp(-1j * (phase + np.pi * d_neg * t_last))

    # Hann window = (1 - cos(2πt/T)) / 2
    # Using exp(j2πy) - 1 = exp(jπy) * 2j * sin(πy)
    # sin(π*d_pos/d) / (f0 - x) = (π/d) * sinc(d_pos/d)

    term_pos = -(phase_pos / denom_pos) * np.sinc(d_pos * t_last)
    term_neg = -(phase_neg / denom_neg) * np.sinc(d_neg * t_last)

    return .5 * abs(ampl) * d * norm * (term_pos + term_neg) # type: ignore


def _cos4n_base(
    x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    norm: float,
) -> NDArray[np.complexfloating]:
    t_middle = .5 * t_last 
    cos2n_1 = _cos2n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_middle,  norm=norm)
    
    ampl_compl = amplitude_i + 1j * amplitude_q
    ampl_compl *= np.exp(-2j * np.pi * frequency * t_middle)

    cos2n_2 = _cos2n_base(
        x=x, amplitude_i=np.real(ampl_compl), amplitude_q=np.imag(ampl_compl),
        frequency=frequency, decay=decay, t_last=t_middle,  norm=norm)
    
    return cos2n_1 + cos2n_2 * np.exp(-2j * np.pi * x * t_middle)

def _cos6n_base(
    x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    norm: float,
) -> NDArray[np.complexfloating]:
    t1 = t_last  / 3.
    cos2n_1 = _cos2n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t1,  norm=norm)
    
      
    ampl_compl = amplitude_i + 1j * amplitude_q
    ampl_compl *= np.exp(-2j * np.pi * frequency * t1)

    cos2n_2 = _cos2n_base(
        x=x, amplitude_i=np.real(ampl_compl), amplitude_q=np.imag(ampl_compl),
        frequency=frequency, decay=decay, t_last=t1,  norm=norm)

    t2 = 2*t_last  / 3.  

    ampl_compl = amplitude_i + 1j * amplitude_q
    ampl_compl *= np.exp(-2j * np.pi * frequency * t2)

    cos2n_3 = _cos2n_base(
        x=x, amplitude_i=np.real(ampl_compl), amplitude_q=np.imag(ampl_compl),
        frequency=frequency, decay=decay, t_last=t1,  norm=norm)
    
    return cos2n_1 \
        + cos2n_2 * np.exp(-2j * np.pi * x * t1) \
        + cos2n_3 * np.exp(-2j * np.pi * x * t2)

def _cos8n_base(
    x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    norm: float,
) -> NDArray[np.complexfloating]:
    t1 = t_last  / 4.
    cos2n_1 = _cos2n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t1,  norm=norm)
    
      
    ampl_compl = amplitude_i + 1j * amplitude_q
    ampl_compl *= np.exp(-2j * np.pi * frequency * t1)

    cos2n_2 = _cos2n_base(
        x=x, amplitude_i=np.real(ampl_compl), amplitude_q=np.imag(ampl_compl),
        frequency=frequency, decay=decay, t_last=t1,  norm=norm)

    t2 = 2*t_last  / 4.  

    ampl_compl = amplitude_i + 1j * amplitude_q
    ampl_compl *= np.exp(-2j * np.pi * frequency * t2)

    cos2n_3 = _cos2n_base(
        x=x, amplitude_i=np.real(ampl_compl), amplitude_q=np.imag(ampl_compl),
        frequency=frequency, decay=decay, t_last=t1,  norm=norm)

    t3 = 3*t_last  / 4.  

    ampl_compl = amplitude_i + 1j * amplitude_q
    ampl_compl *= np.exp(-2j * np.pi * frequency * t3)

    cos2n_4 = _cos2n_base(
        x=x, amplitude_i=np.real(ampl_compl), amplitude_q=np.imag(ampl_compl),
        frequency=frequency, decay=decay, t_last=t1,  norm=norm)
    
    return cos2n_1 \
        + cos2n_2 * np.exp(-2j * np.pi * x * t1) \
        + cos2n_3 * np.exp(-2j * np.pi * x * t2) \
        + cos2n_4 * np.exp(-2j * np.pi * x * t3) \

def _blackman_base(x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    norm: float,
) -> NDArray[np.complexfloating]:
    rect =  _rect_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last, norm=norm)
    cos2n = _cos2n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)
    cos4n = _cos4n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)
    
    return (128 * rect + 18480 * cos2n - 2860 * cos4n) / 18608

def _nuttall_base(x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    norm: float,
) -> NDArray[np.complexfloating]:
    cos2n = _cos2n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)
    cos4n = _cos4n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)
    cos6n = _cos6n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)
    
    return (974_792 * cos2n - 288_464 * cos4n + 25_208 * cos6n) / 1_000_000

def _blackman_nuttall_base(x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    norm: float,
) -> NDArray[np.complexfloating]:
    rect =  _rect_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last, norm=norm)
    cos2n = _cos2n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)
    cos4n = _cos4n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)
    cos6n = _cos6n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)
    
    return (3_628 * rect + 9_783_550 * cos2n - 2_731_990 * cos4n + 212_822 * cos6n) / 10_000_000

def _blackman_harris_base(x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    norm: float,
) -> NDArray[np.complexfloating]:
    rect =  _rect_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last, norm=norm)
    cos2n = _cos2n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)
    cos4n = _cos4n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)

    
    return (53_188 * rect + 9_946_812 * cos2n - 1_565_586 * cos4n) / 10_000_000

def _flat_top_base(x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    norm: float,
) -> NDArray[np.complexfloating]:
    rect =  _rect_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last, norm=norm)
    cos2n = _cos2n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)
    cos4n = _cos4n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)
    cos6n = _cos6n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)
    cos8n = _cos8n_base(
        x=x, amplitude_i=amplitude_i, amplitude_q=amplitude_q,
        frequency=frequency, decay=decay, t_last=t_last,  norm=norm)

    
    return (
        -421_051 * rect
        + 833_263_160 * cos2n
        - 554_526_316 * cos4n
        + 167_157_894 * cos6n
        - 13_894_736 * cos8n
    ) / 1_000_000_000

def _bartlett_base(
    x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    norm: float,
) -> NDArray[np.complexfloating]:
    ampl = amplitude_i - 1j * amplitude_q
    phase = np.angle(ampl)
    f0 = frequency

    decay_term = decay / (2 * np.pi)

    d_pos = (f0 - x) + 1j * decay_term
    d_neg = (f0 + x) - 1j * decay_term

    phase_pos = np.exp(1j * (phase + np.pi * d_pos * t_last))
    phase_neg = np.exp(-1j * (phase + np.pi * d_neg * t_last))
    
    return (t_last / 2) * norm * (
            phase_pos * abs(ampl) * np.sinc(d_pos * t_last/2)**2
            + phase_neg * abs(ampl) * np.sinc(d_neg * t_last/2)**2
    )  # type: ignore


def _lorentzian(
    x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
) -> NDArray[np.complexfloating]:
    if decay <= 0.0:
        raise ValueError(f"Parameter 'decay' ({decay}) must be > 0.")

    common = decay + 2j * np.pi * x
    return (
        (amplitude_i - 1j * amplitude_q) / (common - 2j * np.pi * frequency)
        + (amplitude_i + 1j * amplitude_q) / (common + 2j * np.pi * frequency)
    )


def _exponential_base(
    x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    alpha: float,
    norm: float,
) -> NDArray[np.complexfloating]:
    base = _lorentzian(x, amplitude_i, amplitude_q, frequency, decay + alpha / t_last)

    start = amplitude_i - 1j * amplitude_q
    end = start * np.exp((2j * np.pi * frequency - decay) * t_last - alpha)
    correction = (
        _lorentzian(x, np.real(end), -np.imag(end), frequency, decay + alpha / t_last)
        * np.exp(-2j * np.pi * x * t_last)
    )
    return(base - correction) * norm

def _exponential_baseline(
    x: NDArray,
    amplitude_i: float,
    amplitude_q: float,
    frequency: float,
    decay: float,
    t_last: float,
    alpha: float,
) -> NDArray[np.complexfloating]:
    last = (
        Sinusoid.eval(
            np.array([t_last]), amplitude_i, amplitude_q, frequency, decay + alpha / t_last
        )[0]
        * np.exp(-2j * np.pi * x * t_last)
    )
    return (amplitude_i + last) / 2


# ---------------------------------------------------------------------------
# Assign windows
# ---------------------------------------------------------------------------

SineFourier.WINDOW_CORRECTED_BASE = {
    WindowType.DIRICHLET: _rect_base,
    WindowType.HAMMING:  _hamming_base,
    WindowType.HANN: _cos2n_base,
    WindowType.BLACKMAN: _blackman_base,
    WindowType.NUTTAL: _nuttall_base,
    WindowType.BLACKMAN_NUTTAL: _blackman_nuttall_base,
    WindowType.BLACKMAN_HARRIS: _blackman_harris_base,
    WindowType.FLAT_TOP: _flat_top_base,
    WindowType.BARTLETT: _bartlett_base,
    WindowType.EXPONENTIAL_ASYM: _exponential_base,

    # WindowType.COS6N: _cos6n_base,
    # WindowType.COS8N: _cos8n_base,
}

SineFourier.BASELINE = {
    WindowType.DIRICHLET: _rect_baseline,
    WindowType.HAMMING: partial(_rect_baseline, norm=4/46),
    WindowType.HANN: None,
    WindowType.BLACKMAN: partial(_rect_baseline, norm=128/18608),
    WindowType.NUTTAL: None,
    WindowType.BLACKMAN_NUTTAL: partial(_rect_baseline, norm=3_628/10_000_000),
    WindowType.BLACKMAN_HARRIS: partial(_rect_baseline, norm=53_188/10_000_000),
    WindowType.FLAT_TOP: partial(_rect_baseline, norm=-421_051/1_000_000_000),
    WindowType.BARTLETT: None,
    WindowType.EXPONENTIAL_ASYM: _exponential_baseline,

    # WindowType.COS6N: None,
    # WindowType.COS8N: None,
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sine_params(
    amplitude_i: float = 0.0,
    amplitude_q: float = 0.0,
    frequency: float = 0.0,
    decay: float | None = None,
    frequency_min: float | None = None,
    frequency_max: float | None = None,
    decay_min: float | None = 0.0,
    decay_max: float | None = None,
    **_,
) -> lmfit.Parameters:
    params = lmfit.Parameters()
    params.add("amplitude_i", value=amplitude_i)
    params.add("amplitude_q", value=amplitude_q)
    params.add(
        "frequency",
        value=frequency,
        min=-np.inf if frequency_min is None else frequency_min,
        max=np.inf if frequency_max is None else frequency_max,
    )
    if decay is None:
        params.add(
            "decay",
            value=0.0,
            min=-np.inf if decay_min is None else decay_min,
            max=np.inf if decay_max is None else decay_max,
            vary=False,
        )
    else:
        params.add(
            "decay",
            value=decay,
            min=-np.inf if decay_min is None else decay_min,
            max=np.inf if decay_max is None else decay_max,
        )
    return params


