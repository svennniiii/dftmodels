# dftmodels

Analytical DFT models for spectral parameter estimation.

## Overview

Fitting peaks in DFT spectra with continuous Fourier transform (CFT) models (sinc functions, Lorentzians, and their window-convolved variants) introduces systematic errors in the estimated amplitude, frequency, phase, and linewidth. Three effects are responsible: the DFT sum gives full weight to the boundary samples where the trapezoidal-rule approximation gives half weight (boundary mismatch); sampling makes the spectrum periodic, introducing aliased copies at f ± k·f_s (aliasing); and the finite window modifies the spectral shape in a way that depends on both the window coefficients and the signal frequency (windowing). Each effect produces a deterministic, structured residual in the DFT.

`dftmodels` provides closed-form analytical models that include these effects as additive correction terms: WINDOW (the CFT of the windowed signal), BASELINE (the boundary weight mismatch), and SAMPLING (aliased spectral copies up to a configurable order). Combined with least-squares fitting, these models form estimators whose variance can approach the Cramér–Rao bound (CRB).

**When corrections matter.** Systematic bias is only the limiting factor when SNR is high and records are short. The SAMPLING correction becomes negligible above a crossover acquisition length that depends on window and noise level (see [Notebook 02](https://github.com/svennniiii/dftmodels/blob/main/examples/02_dft_corrections.ipynb)). For long records or when noise dominates, a CFT-based sinc or Lorentzian model is adequate. The gains shown in the examples below are large by design — the signal parameters were chosen so that systematic bias is the limiting factor, not noise. The package is most relevant when that condition holds.

## Installation

```bash
pip install dftmodels
```

Dependencies: Python ≥ 3.12, `numpy` ≥ 2.0, `scipy` ≥ 1.15, `lmfit` ≥ 1.0.

## Examples

Six notebooks in [examples/](https://github.com/svennniiii/dftmodels/tree/main/examples) work through the main topics in sequence.

| Notebook | Topic |
|---|---|
| [01 — Normalizations](https://github.com/svennniiii/dftmodels/blob/main/examples/01_dft_normalization.ipynb) | ASD, PSD, and CFT normalizations; Parseval's theorem |
| [02 — DFT corrections](https://github.com/svennniiii/dftmodels/blob/main/examples/02_dft_corrections.ipynb) | Origin and convergence of the three correction terms |
| [03 — Precision fitting](https://github.com/svennniiii/dftmodels/blob/main/examples/03_precision_fitting.ipynb) | Sinusoid parameter estimation; bias elimination; Cramér–Rao bound |
| [04 — Decaying signals](https://github.com/svennniiii/dftmodels/blob/main/examples/04_decaying_signals.ipynb) | Lorentzian fitting; exponential windows; line broadening |
| [05 — Composite models](https://github.com/svennniiii/dftmodels/blob/main/examples/05_composite_model.ipynb) | Simultaneous multi-peak fitting; spectral background |
| [06 — Window comparison](https://github.com/svennniiii/dftmodels/blob/main/examples/06_window_comparison.ipynb) | Window choice: Fisher information loss vs. sidelobe suppression |

---

### Normalizations

Three normalizations are provided. **ASD** and **PSD** satisfy Parseval's theorem — ∫|ASD(f)|² df = ∫PSD(f) df = RMS² — and are appropriate for stationary signals. The normalization factor includes the window RMS (w_rms) to compensate for amplitude attenuation; this correction is exact in the limit N → ∞ and introduces a residual error of order 1/N. **CFT** normalization satisfies ∫X_CFT(f) df = x(0) and is appropriate for decaying signals, where total power depends on acquisition length and ASD peak height grows with T.

![CFT vs ASD normalization](https://raw.githubusercontent.com/svennniiii/dftmodels/main/examples/figures/01_dft_normalization_fig01.svg)

*Top: CFT normalization — the spectral integral converges to x(0) = A for both decaying (left cluster, 8–12 Hz) and stationary sinusoids (right cluster, 18–22 Hz), independent of T. Bottom: ASD normalization — for stationary sinusoids √∫|ASD|² = RMS is stable; for decaying sinusoids the integral grows with T.*

*[Notebook 01](https://github.com/svennniiii/dftmodels/blob/main/examples/01_dft_normalization.ipynb). Implementation: `NormType`, `DFTConfig.norm_factor` in [dft/config.py](https://github.com/svennniiii/dftmodels/blob/main/dftmodels/dft/config.py).*

---

### DFT corrections

An off-bin sinusoid produces a deterministic, structured residual in the DFT that is not captured by the WINDOW-only model. The figure below shows the residual reduction as corrections are applied in sequence. Without corrections, the RMS residual for a rectangular window at N = 100 samples is ~7×10⁻² V/√Hz; with all corrections at order 100 it drops to ~3×10⁻⁵ V/√Hz. For windows that taper to zero at both boundaries (Hann, Nuttall, Bartlett), the BASELINE correction is identically zero. The SAMPLING correction converges faster for windows with steeper sidelobe roll-off; rectangular and Bartlett windows require more orders than Hann or Nuttall (see window sweep table in Notebook 02).

![Correction convergence](https://raw.githubusercontent.com/svennniiii/dftmodels/main/examples/figures/02_dft_corrections_fig01.svg)

The corrections become irrelevant above a crossover acquisition length where the noise floor dominates. The figure below sweeps N at fixed f_s and shows the normalized model error for each correction level alongside reference noise floors at fixed σ/A ratios. Below the crossover, the fit is model-limited; above it, noise-limited.

![Corrections vs acquisition length](https://raw.githubusercontent.com/svennniiii/dftmodels/main/examples/figures/02_dft_corrections_fig02.svg)

*[Notebook 02](https://github.com/svennniiii/dftmodels/blob/main/examples/02_dft_corrections.ipynb). Implementation: `DFTCorrection`, `DFTCorrectionMode` in [dft/correction.py](https://github.com/svennniiii/dftmodels/blob/main/dftmodels/dft/correction.py).*

---

### Precision fitting — sinusoid

300 noise realizations of a rectangular-windowed sinusoid at f₀ = 10.42 Hz (0.42 bins from the nearest bin), T = 1 s, σ = 0.01 V. The table below compares four correction levels; the violin plots show the full error distribution.

| | freq RMSE (mHz) | amp RMSE (mV) | phase RMSE (mrad) |
|---|---|---|---|
| CRB | 0.39 | 1.41 | 0.98 |
| ★ None | 9.90 | 117.4 | 27.18 |
| ★ All (order 10) | 0.41 | 1.47 | 1.43 |

Frequency and amplitude RMSE approach the CRB once corrections eliminate the systematic bias.

![Precision fitting violin plots](https://raw.githubusercontent.com/svennniiii/dftmodels/main/examples/figures/03_precision_fitting_fig01.svg)

*[Notebook 03](https://github.com/svennniiii/dftmodels/blob/main/examples/03_precision_fitting.ipynb). Implementation: `SineFourier` in [models/sinusoid.py](https://github.com/svennniiii/dftmodels/blob/main/dftmodels/models/sinusoid.py).*

---

### Decaying signals

For a decaying sinusoid x(t) = A·cos(2πf₀t + φ)·e^{−γt}, the CFT near f₀ is a complex Lorentzian: X(f) ≈ (Ae^{jφ}/2) / (γ + 2πj(f − f₀)). `SineFourier` with a nonzero `decay` parameter implements the exact analytical DFT expression and fits four parameters (frequency f₀, quadrature amplitudes Aᵢ = A cos φ and A_q = A sin φ, and decay rate γ); amplitude A, phase φ, and FWHM = γ/π are derived quantities. The Fisher information matrix for a decaying sinusoid has non-zero off-diagonal coupling between f and γ — the two parameters are not decoupled, unlike the pure sinusoid case.

300 realizations at σ = 0.05 V, T = 5 s, γ = 0.5 s⁻¹:

| | amp RMSE (mV) | phase RMSE (mrad) | freq RMSE (mHz) | decay RMSE (ms⁻¹) |
|---|---|---|---|---|
| CRB | 10.20 | 5.13 | 0.62 | 3.87 |
| ★ None | 59.98 | 5.93 | 0.67 | 34.41 |
| ★ All (order 10) | 10.03 | 5.14 | 0.61 | 3.83 |

![Lorentzian bare vs corrected](https://raw.githubusercontent.com/svennniiii/dftmodels/main/examples/figures/04_decaying_signals_fig02.svg)

Applying an exponential window w(t_n) = e^{−α·t_n/T} before the DFT shifts the effective decay to γ_eff = γ + α/T (line broadening). `SineFourier` with `WindowType.EXPONENTIAL_ASYM` corrects for the window-induced decay shift and recovers the true γ from the broadened spectrum. A bare fit converges to γ_eff, biased by α/T.

*[Notebook 04](https://github.com/svennniiii/dftmodels/blob/main/examples/04_decaying_signals.ipynb). Implementation: `SineFourier` in [models/sinusoid.py](https://github.com/svennniiii/dftmodels/blob/main/dftmodels/models/sinusoid.py).*

---

### Composite models

`CompositeModel` fits a weighted sum of named components simultaneously, with all parameters coupled through a shared covariance matrix. For overlapping spectral lines, sequential peak-by-peak fitting propagates subtraction errors; joint fitting avoids this. Each component's parameters are prefixed by name (`p1_frequency`, `p2_decay`, ...).

A J-coupled quartet — four Lorentzian lines with 1:3:3:1 binomial amplitudes at spacing J = 2.5 Hz — illustrates the setup. A linear complex background b₀ + b₁f is included as a fifth component; fitting without it forces the peaks to absorb the baseline, distorting amplitude ratios.

![Composite fit with decomposition](https://raw.githubusercontent.com/svennniiii/dftmodels/main/examples/figures/05_composite_model_fig01.svg)

*Top: magnitude spectrum (grey), total fit (dashed), individual peak contributions (dotted), and background (dash-dot). Bottom: residuals with and without a background component in the model.*

The corrected composite estimator approaches the CRB; outer peaks (A = 1 V) have proportionally larger uncertainty than inner peaks (A = 3 V).

*[Notebook 05](https://github.com/svennniiii/dftmodels/blob/main/examples/05_composite_model.ipynb). Implementation: `CompositeModel` in [models/composite.py](https://github.com/svennniiii/dftmodels/blob/main/dftmodels/models/composite.py).*

---

### Window choice

The CRB is determined by the raw, unweighted time-domain data and is independent of the window. Any non-rectangular window downweights samples near the record boundaries, discarding Fisher information. This loss cannot be recovered by corrections. For an isolated peak, the rectangular window is the only window whose RMSE approaches the CRB; all others inflate variance, with the inflation scaling approximately with their equivalent noise bandwidth.

![Window RMSE comparison](https://raw.githubusercontent.com/svennniiii/dftmodels/main/examples/figures/06_window_comparison_fig00.svg)

*RMSE for frequency (left) and amplitude (right) estimation across six windows at two fractional bin offsets δ = 0 and δ = 0.5. The CRB (dotted line) is the same for both offsets and all windows. N = 1000, f_s = 1000 Hz, σ = 0.05 V, 300 realizations.*

The tradeoff reverses when a nearby peak is absent from the model. Its spectral sidelobes leak into the fit region; the rectangular window has the slowest-decaying sidelobes (∝ 1/f), so its estimator is most sensitive to this. At worst-case interferer offset (δ = 0.484, +5 bins), the rectangular window has a frequency RMSE of 13.5 mHz vs. 2.3 mHz for Hamming.

Window choice therefore depends on context: isolated peaks favour rectangular; peaks with unmodeled nearby components favour windowed estimators.

*[Notebook 06](https://github.com/svennniiii/dftmodels/blob/main/examples/06_window_comparison.ipynb).*

---

## Usage

```python
import numpy as np
from dftmodels import (
    SignalSeries, NormType, WindowType, DFTRange,
    DFTCorrection, DFTCorrectionMode, SineFourier,
)

t  = np.arange(0, 1.0, 1 / 100.0)
y  = 2.0 * np.cos(2 * np.pi * 10.42 * t + 0.5)
y += np.random.normal(scale=0.01, size=len(t))

fourier = SignalSeries(x=t, y=y).calculate_dft(
    norm=NormType.ASD, window=WindowType.RECTANGULAR,
    dft_range=DFTRange.SINGLE_SIDED, pad=10.0,
)

model  = SineFourier(fourier.dft_config, DFTCorrection(DFTCorrectionMode.ALL, order=10))
params = model.make_params(
    amplitude_i=2.0, amplitude_q=0.0, frequency=10.0,
    frequency_min=8.0, frequency_max=13.0,
)
result = model.fit(fourier, params, mask=(fourier.x >= 8.0) & (fourier.x <= 13.0))

print(f"Amplitude : {model.amplitude(result.params):.6f} V")
print(f"Frequency : {model.center(result.params):.6f} Hz")
print(f"Phase     : {model.phase(result.params):.6f} rad")
```

For decaying signals use `SineFourier` with `NormType.CFT` and a nonzero `decay` parameter. For simultaneous multi-peak fitting use `CompositeModel`. See the notebooks for detailed examples of each.

## Models

| Signal type | Model class | Windows |
|---|---|---|
| Sinusoid | `SineFourier` | All cosine-sum windows, Bartlett |
| Decaying sinusoid | `SineFourier` (nonzero `decay`) | All cosine-sum windows, Bartlett, Exponential |
| Time-domain sinusoid / decaying sinusoid | `Sinusoid` | N/A |
| Custom | `ModelBase.build_model(fn)` | N/A |

## Limitations

**Scope of corrections.** The correction terms reduce systematic bias from the finite, sampled, windowed DFT. An alternative that achieves the same result is to define the signal model in the time domain and compute the DFT numerically at each evaluation. The analytical corrections are a closed-form shortcut to that approach; they are most efficient when the signal model is simple and the number of evaluations is large. For short-duration signals with few samples, the time-domain approach may be simpler.

**Noise model.** The Cramér–Rao bounds in the examples assume additive white Gaussian noise. Colored noise or non-Gaussian distributions will produce different variance floors; the model residuals remain unbiased but the optimality guarantee does not carry over.

**Model vs. estimator.** The classes in this package are models, not estimators. Combined with a least-squares fitting procedure (`lmfit.Minimizer`), they form estimators. The CRB is a property of the estimator (model + fitting algorithm + noise model), not of the model alone.

## License

MIT License. See [LICENSE](LICENSE) for details.
