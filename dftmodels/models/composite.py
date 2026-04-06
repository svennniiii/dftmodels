from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import lmfit

from dftmodels.dft.series import DataSeries
from .base import ModelBase, _parse_weights


class CompositeModel:
    """Sum of multiple :class:`ModelBase` components fitted simultaneously.

    Each component is identified by a string prefix.  Parameter names are
    scoped as ``"<prefix>_<param_name>"`` so that components with identical
    parameter names (e.g. two Lorentzians both having ``frequency``) do not
    collide.

    Example::

        cfg = DFTConfig(...)
        correction = DFTCorrection(DFTCorrectionMode.ALL)

        model = CompositeModel([
            ("peak1", SineFourier(cfg, correction)),
            ("peak2", SineFourier(cfg, correction)),
            ("bg",    LinearComplexModel(cfg)),
        ])

        params = model.make_params(
            peak1=dict(frequency=10.0, decay=0.5),
            peak2=dict(frequency=12.0, decay=0.5),
        )
        result = model.fit(data, params)
    """

    def __init__(self, components: list[tuple[str, ModelBase]]) -> None:
        if not components:
            raise ValueError("CompositeModel requires at least one component")
        self._components = list(components)

    @property
    def components(self) -> list[tuple[str, ModelBase]]:
        return self._components

    def make_params(self, **per_prefix_params: dict[str, lmfit.Parameters]) -> lmfit.Parameters:
        """Build merged lmfit Parameters for all components.

        Parameters
        ----------
        **per_prefix_params:
            Params keyed by component prefix.

        Returns
        -------
        lmfit.Parameters
            All parameters with names ``"<prefix>_<param>"``
        """
        merged = lmfit.Parameters()
        for prefix, component in self._components:
            sub_params = per_prefix_params.get(prefix, component.make_params())
            for name, param in sub_params.items():
                merged.add(
                    f"{prefix}_{name}",
                    value=param.value,  # type: ignore
                    min=param.min,  # type: ignore
                    max=param.max,  # type: ignore
                    vary=param.vary,  # type: ignore
                )
        return merged

    def eval(self, params: lmfit.Parameters, x: NDArray) -> NDArray:
        """Evaluate the sum of all components at frequencies ``x``."""
        evals = []
        for prefix, component in self._components:
            sub_params = lmfit.Parameters()
            for name in component.PARAM_NAMES:
                p = params[f"{prefix}_{name}"]
                sub_params.add(name, value=p.value)
            evals.append(component.eval(sub_params, x))
        return np.sum(evals, axis=0)  # type: ignore[return-value]

    def fit(
        self,
        data_series: DataSeries,
        params: lmfit.Parameters,
        mask: NDArray[np.bool_] | None = None,
        weights: NDArray | float | None = None,
        **minimizer_kwargs,
    ) -> lmfit.minimizer.MinimizerResult:
        """Fit the composite model to a :class:`DataSeries`.

        Parameters
        ----------
        fourier:
            Data to fit.
        params:
            Initial parameters from :meth:`make_params`.
        mask:
            Boolean array selecting which frequency bins to include.
        weights:
            Per-point weights (or scalar).
        **minimizer_kwargs:
            Passed to ``lmfit.Minimizer.minimize()``.
        """
        x = data_series.x
        data = data_series.y
        w = _parse_weights(weights, len(x))

        if mask is not None:
            x = x[mask]
            data = data[mask]
            w = w[mask]

        def residual(p: lmfit.Parameters) -> NDArray:
            model = self.eval(p, x)
            diff = (data - model) * np.sqrt(w)
            if np.iscomplexobj(diff):
                return np.concatenate([np.real(diff), np.imag(diff)])
            return diff

        minimizer = lmfit.Minimizer(residual, params)
        return minimizer.minimize(**minimizer_kwargs)
