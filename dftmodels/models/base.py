from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable
import inspect

import numpy as np
from numpy.typing import NDArray
import lmfit

from dftmodels.dft.config import DFTConfig, NormType
from dftmodels.dft.correction import DFTCorrection, DFTCorrectionMode
from dftmodels.dft.series import FourierSeries, DataSeries


def _parse_weights(weights: NDArray | float | None, n: int) -> NDArray:
    if weights is None:
        return np.ones(n)
    if isinstance(weights, (int, float)):
        return np.full(n, float(weights))
    return np.asarray(weights, dtype=float)


class ModelBase(ABC):
    PARAM_NAMES: tuple[str, ...]

    @classmethod
    def build_model(cls, fun: Callable) -> ModelBase:
        sig = inspect.signature(fun)

        # Assume first arg is x, rest are parameters
        param_names = tuple(
            p.name for p in sig.parameters.values()
            if p.name != "x"
        )

        class GeneratedModel(cls):
            PARAM_NAMES = param_names

            @staticmethod
            def make_params(**kwargs) -> lmfit.Parameters:
                params = lmfit.Parameters()
                for name in param_names:
                    params.add(name, value=kwargs.get(name, 1.0))  # default value
                return params

            def eval(self, params: lmfit.Parameters, x: NDArray) -> NDArray:
                kwargs = {name: params[name].value for name in self.PARAM_NAMES}
                return fun(x, **kwargs)

        return GeneratedModel()

    @staticmethod
    @abstractmethod
    def make_params(**kwargs) -> lmfit.Parameters:
        """Create lmfit Parameters with sensible defaults and optional initial values."""

    @abstractmethod
    def eval(self, params: lmfit.Parameters, x: NDArray) -> NDArray:
        pass
    
    def fit(
        self,
        series: DataSeries,
        params: lmfit.Parameters,
        mask: NDArray[np.bool_] | None = None,
        weights: NDArray | float | None = None,
        baseline: Callable[[NDArray], NDArray] | None = None,
        **minimizer_kwargs,
    ) -> lmfit.minimizer.MinimizerResult:
        """Fit the model to a FourierSeries using least-squares.

        Parameters
        ----------
        series:
            The data to fit.
        params:
            Initial lmfit Parameters (from ``make_params``).
        mask:
            Boolean array selecting which frequency bins to include.
        weights:
            Per-point weights (or scalar). Residuals are multiplied by sqrt(weights).
        baseline:
            Callable ``f(x) -> y`` subtracted from data before fitting.
        **minimizer_kwargs:
            Passed to ``lmfit.Minimizer.minimize()``.

        Returns
        -------
        lmfit.minimizer.MinimizerResult
            Contains fitted ``.params``, ``.covar``, ``.residual``, etc.
        """
        x = series.x
        data = series.y
        w = _parse_weights(weights, len(x))

        if mask is not None:
            x = x[mask]
            data = data[mask]
            w = w[mask]

        _baseline = baseline if baseline is not None else lambda _x: 0.0

        def residual(p: lmfit.Parameters) -> NDArray:
            model = self.eval(p, x)
            diff = (data - _baseline(x) - model) * np.sqrt(w)
            if np.iscomplexobj(diff):
                return np.concatenate([np.real(diff), np.imag(diff)])
            return diff

        minimizer = lmfit.Minimizer(residual, params)
        return minimizer.minimize(**minimizer_kwargs)


class FourierBase(ABC):
    @property
    @abstractmethod
    def dft_config(self) -> DFTConfig:
        pass

    @property
    @abstractmethod
    def dft_correction(self) -> DFTCorrection:
        pass
            

class FourierModelBase(ModelBase, FourierBase):
    """Base class for analytic Fourier-domain models.

    Subclasses implement the closed-form expression for a windowed signal
    in the DFT, including optional corrections for sampling artifacts,
    baseline discontinuities, and window edge effects.

    Fitting uses lmfit.Minimizer. Complex residuals are handled by stacking
    real and imaginary parts into a single real residual vector.
    """

    def __init__(
        self,
        dft_config: DFTConfig,
        dft_correction: DFTCorrection = DFTCorrection(DFTCorrectionMode.WINDOW_ONLY),
    ) -> None:
        self._dft_config = dft_config
        self._dft_correction = dft_correction
        self._compiled = self._compile()

    @property
    def dft_config(self) -> DFTConfig:
        return self._dft_config

    @property
    def dft_correction(self) -> DFTCorrection:
        return self._dft_correction

    @abstractmethod
    def _compile(self) -> Callable[..., NDArray]:
        """Return the compiled evaluation callable with DFT config baked in.

        The callable signature must be ``f(x, *param_values)`` where
        param values are ordered as ``PARAM_NAMES``.
        """

    def _sampling_correction(
        self, x: NDArray, *p: float
    ) -> NDArray[np.complexfloating]:
        result = np.zeros_like(x, dtype=np.complex128)
        sr = self._dft_config.sample_rate
        for i in range(1, self._dft_correction.order + 1):
            result += self._compiled(x + i * sr, *p)
            result += self._compiled(x - i * sr, *p)
        return result

    def _baseline_correction(
        self, x: NDArray, *p: float
    ) -> NDArray[np.complexfloating]:
        return np.zeros_like(x, dtype=np.complex128)

    def eval(self, params: lmfit.Parameters, x: NDArray) -> NDArray:
        if self.dft_config.norm_type in (NormType.ASD_ABS, NormType.PSD):
            raise RuntimeError("Absolute norms are currently not supported for Fourier models.")

        p = [params[name].value for name in self.PARAM_NAMES]
        result = self._compiled(x, *p)

        if DFTCorrectionMode.SAMPLING_ONLY in self._dft_correction.mode:
            result = result + self._sampling_correction(x, *p)

        if DFTCorrectionMode.BASELINE_ONLY in self._dft_correction.mode:
            result = result + self._baseline_correction(x, *p)

        result *= self.dft_config.norm_factor

        if self._dft_config.norm_type == NormType.ASD_ABS:
            return np.abs(result)
        elif self._dft_config.norm_type == NormType.PSD:
            return np.abs(result) ** 2
        return result

