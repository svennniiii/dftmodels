from __future__ import annotations

from typing import Callable, Literal, overload

import numpy as np
from numpy.typing import NDArray
import lmfit


@overload
def cramer_rao_bound(
    model_fn: Callable,
    params: lmfit.Parameters,
    t: NDArray,
    noise_std: float,
    *,
    return_cov: Literal[False] = ...,
) -> dict[str, float]: ...

@overload
def cramer_rao_bound(
    model_fn: Callable,
    params: lmfit.Parameters,
    t: NDArray,
    noise_std: float,
    *,
    return_cov: Literal[True],
) -> tuple[dict[str, float], NDArray]: ...

def cramer_rao_bound(
    model_fn: Callable,
    params: lmfit.Parameters,
    t: NDArray,
    noise_std: float,
    *,
    return_cov: bool = False,
) -> dict[str, float] | tuple[dict[str, float], NDArray]:
    """
    Cramér–Rao lower bound via numerical Jacobian.

    Parameters
    ----------
    model_fn : callable
        f(t, param1, param2, ...) — time-domain model evaluation.
        Positional args must match the insertion order of `params`.
    params : lmfit.Parameters
        Parameter values. Only free (vary=True) parameters are included in the
        Fisher information matrix; fixed parameters are held constant.
    t : NDArray
        Time grid at which the signal is evaluated.
    noise_std : float
        White noise standard deviation σ.
    return_cov : bool, optional
        If True, also return the full covariance matrix (inverse FIM) as an
        NDArray of shape (n_free, n_free), rows/columns ordered as free params.

    Returns
    -------
    crb : dict[str, float]
        Cramér–Rao bound (variance σ² lower bound) keyed by parameter name.
        Take the square root to obtain the standard-deviation bound.
    cov : NDArray, only when return_cov=True
        Full covariance matrix FIM⁻¹.
    """
    entries = [(name, p.value, p.vary) for name, p in params.items()]

    all_values = np.array([v for _, v, _ in entries])
    free_names = [n for n, _, f in entries if f]
    free_indices = [i for i, (_, _, f) in enumerate(entries) if f]

    h = 1e-6 * np.maximum(np.abs(all_values[free_indices]), 1e-3)

    J = np.zeros((len(t), len(free_names)))
    for k, (idx, hk) in enumerate(zip(free_indices, h)):
        pp, pm = all_values.copy(), all_values.copy()
        pp[idx] += hk
        pm[idx] -= hk
        J[:, k] = (model_fn(t, *pp) - model_fn(t, *pm)) / (2 * hk)

    FIM = J.T @ J / noise_std**2
    Cov = np.linalg.inv(FIM)

    crb = {name: float(Cov[k, k]) for k, name in enumerate(free_names)}
    if return_cov:
        return crb, Cov
    return crb
