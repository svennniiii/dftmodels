import unittest
from itertools import product

import numpy as np

from dftmodels.dft.config import DFTRange, NormType, WindowType
from dftmodels.dft.correction import DFTCorrection, DFTCorrectionMode
from dftmodels.dft.series import SignalSeries
from dftmodels.models.sinusoid import Sinusoid, SineFourier


def _make_sinusoid_signal(
    amplitude: float = 2.0,
    frequency: float = 10.37,
    phase: float = 0.5,
    n: int = 400,
    duration: float = 4.0,
) -> tuple[SignalSeries, float, float]:
    x = np.linspace(0, duration, n, endpoint=False)
    ai = amplitude * np.cos(phase)
    aq = amplitude * np.sin(phase)
    y = Sinusoid.eval(x, ai, aq, frequency)
    return SignalSeries(x=x, y=y), ai, aq


class TestSinusoidEval(unittest.TestCase):
    def test_pure_cosine(self):
        x = np.linspace(0, 1, 1000, endpoint=False)
        y = Sinusoid.eval(x, amplitude_i=1.0, amplitude_q=0.0, frequency=5.0)
        np.testing.assert_array_almost_equal(y, np.cos(2 * np.pi * 5.0 * x))

    def test_pure_sine(self):
        x = np.linspace(0, 1, 1000, endpoint=False)
        y = Sinusoid.eval(x, amplitude_i=0.0, amplitude_q=1.0, frequency=5.0)
        np.testing.assert_array_almost_equal(y, np.sin(2 * np.pi * 5.0 * x))


class TestSinusoidFit(unittest.TestCase):
    def setUp(self):
        self.signal, self.ai, self.aq = _make_sinusoid_signal()
        self.frequency = 10.37
        rng = np.random.default_rng(seed=42)
        self.signal.y += rng.normal(scale=1e-3, size=len(self.signal))

    def test_fit_recovers_params(self):
        model = Sinusoid()
        params = model.make_params(
            amplitude_i=self.ai * 1.05,
            amplitude_q=self.aq * 1.05,
            frequency=self.frequency + 0.005,  # small offset, within one frequency bin
        )
        result = model.fit(self.signal.x, self.signal.y, params)
        self.assertTrue(result.success)  # type: ignore[union-attr]
        self.assertAlmostEqual(result.params["frequency"].value, self.frequency, places=5)  # type: ignore[union-attr]


class TestSineFourierCorrections(unittest.TestCase):
    def setUp(self):
        self.signal, self.ai, self.aq = _make_sinusoid_signal()
        self.frequency = 10.37

    def test_corrections_reduce_error(self):
        corrections = [
            DFTCorrection(mode=DFTCorrectionMode.WINDOW),
            DFTCorrection(mode=DFTCorrectionMode.BASELINE),
            DFTCorrection(mode=DFTCorrectionMode.ALL, order=10),
            DFTCorrection(mode=DFTCorrectionMode.ALL, order=100),
        ]
        pads = (1.0, 2.0, 4.5)
        norms = (NormType.CFT, NormType.ASD)
        ranges = tuple(DFTRange)
        windows = (WindowType.RECTANGULAR, WindowType.HAMMING, WindowType.HANN)

        for pad, norm, window, dft_range in product(pads, norms, windows, ranges):
            if window == WindowType.HANN and norm == NormType.CFT:
                continue  # CFT + Hann is invalid
            fourier = self.signal.calculate_dft(
                pad=pad, norm=norm, window=window, dft_range=dft_range
            )
            
            previous_error = np.inf

            for correction in corrections:
                model = SineFourier(
                    dft_config=fourier.dft_config,
                    dft_correction=correction,
                )
                params = model.make_params(
                    amplitude_i=self.ai,
                    amplitude_q=self.aq,
                    frequency=self.frequency,
                )
                predicted = model.eval(params, fourier.x)
                error =  float(np.sum(np.abs(fourier.y - predicted)))

                if correction.mode ==  DFTCorrectionMode.BASELINE and window == WindowType.HANN:
                    # Hann has no baseline correction, error should be unchanged
                    self.assertAlmostEqual(error, previous_error, places=6,
                        msg=f"Hann baseline changed unexpectedly: pad={pad}, norm={norm}, range={dft_range}")
                else:
                    self.assertGreater(
                        previous_error, error,
                        msg=f"Error did not decrease: pad={pad}, norm={norm}, window={window}, "
                            f"range={dft_range}, correction={correction}"
                    )
                previous_error = error


class TestSineFourierFit(unittest.TestCase):
    def setUp(self):
        n, duration = 400, 4.0
        amplitude, frequency, phase = 2.0, 10.37, 0.5
        x = np.linspace(0, duration, n, endpoint=False)
        ai = amplitude * np.cos(phase)
        aq = amplitude * np.sin(phase)
        rng = np.random.default_rng(seed=2023)
        y = Sinusoid.eval(x, ai, aq, frequency) + rng.normal(scale=1e-3, size=n)
        self.signal = SignalSeries(x=x, y=y)

        # Slightly perturbed initial params
        rng2 = np.random.default_rng(seed=99)
        self.ai0 = ai + rng2.normal(scale=0.12)
        self.aq0 = aq + rng2.normal(scale=0.13)
        self.f0 = frequency + rng2.normal(scale=0.05)

    def test_fit_cost_decreases_with_corrections(self):
        corrections = [
            DFTCorrection(mode=DFTCorrectionMode.WINDOW),
            DFTCorrection(mode=DFTCorrectionMode.BASELINE),
            DFTCorrection(mode=DFTCorrectionMode.ALL, order=10),
            DFTCorrection(mode=DFTCorrectionMode.ALL, order=100),
        ]
        pads = (1.0, 2.0, 4.5)
        ranges = tuple(DFTRange)
        windows = (WindowType.RECTANGULAR, WindowType.HAMMING, WindowType.HANN)

        for pad, window, dft_range in product(pads, windows, ranges):
            fourier = self.signal.calculate_dft(
                pad=pad, norm=NormType.ASD, window=window, dft_range=dft_range
            )

            previous_cost = np.inf

            for correction in corrections:
                model = SineFourier(
                    dft_config=fourier.dft_config,
                    dft_correction=correction,
                )
                params = model.make_params(
                    amplitude_i=self.ai0,
                    amplitude_q=self.aq0,
                    frequency=self.f0,
                )
                result = model.fit(fourier, params)
                cost = result.chisqr

                if correction.mode == DFTCorrectionMode.BASELINE and window == WindowType.HANN:
                    self.assertAlmostEqual(cost, previous_cost, places=6)
                else:
                    self.assertGreater(
                        previous_cost, cost,
                        msg=f"Cost did not decrease: pad={pad}, window={window}, "
                            f"range={dft_range}, correction={correction}"
                    )
                previous_cost = cost


class TestSineFourierAmplitudePhase(unittest.TestCase):
    AMPLITUDE = 2.0
    FREQUENCY = 10.37
    PHASE = 0.5
    N = 400
    DURATION = 4.0

    def setUp(self):
        x = np.linspace(0, self.DURATION, self.N, endpoint=False)
        self.ai = self.AMPLITUDE * np.cos(self.PHASE)
        self.aq = self.AMPLITUDE * np.sin(self.PHASE)
        y = Sinusoid.eval(x, self.ai, self.aq, self.FREQUENCY)
        self.signal = SignalSeries(x=x, y=y)
        self.correction = DFTCorrection(mode=DFTCorrectionMode.ALL, order=50)

    def _fit(self, window, norm):
        fourier = self.signal.calculate_dft(pad=2.0, norm=norm, window=window)
        model = SineFourier(dft_config=fourier.dft_config, dft_correction=self.correction)
        params = model.make_params(
            amplitude_i=self.ai * 1.02,
            amplitude_q=self.aq * 1.02,
            frequency=self.FREQUENCY * 1.001,
        )
        result = model.fit(fourier, params)
        return model, result

    def test_rectangular_amplitude(self):
        model, result = self._fit(WindowType.RECTANGULAR, NormType.CFT)
        self.assertAlmostEqual(model.amplitude(result.params), self.AMPLITUDE, places=4)  # type: ignore

    def test_rectangular_phase(self):
        model, result = self._fit(WindowType.RECTANGULAR, NormType.CFT)
        self.assertAlmostEqual(model.phase(result.params), self.PHASE, places=4)  # type: ignore

    def test_rectangular_center(self):
        model, result = self._fit(WindowType.RECTANGULAR, NormType.CFT)
        self.assertAlmostEqual(model.center(result.params), self.FREQUENCY, places=5)  # type: ignore

    def test_hann_amplitude(self):
        model, result = self._fit(WindowType.HANN, NormType.ASD)
        self.assertAlmostEqual(model.amplitude(result.params), self.AMPLITUDE, places=7)  # type: ignore

    def test_hann_phase(self):
        model, result = self._fit(WindowType.HANN, NormType.ASD)
        self.assertAlmostEqual(model.phase(result.params), self.PHASE, places=7)  # type: ignore

    def test_hann_center(self):
        model, result = self._fit(WindowType.HANN, NormType.ASD)
        self.assertAlmostEqual(model.center(result.params), self.FREQUENCY, places=7)  # type: ignore

    def test_hamming_amplitude(self):
        # Hamming window has ~0.3 % systematic leakage; use 1 % tolerance
        model, result = self._fit(WindowType.HAMMING, NormType.CFT)
        self.assertAlmostEqual(model.amplitude(result.params), self.AMPLITUDE, places=6)  # type: ignore

    def test_hamming_phase(self):
        model, result = self._fit(WindowType.HAMMING, NormType.CFT)
        self.assertAlmostEqual(model.phase(result.params), self.PHASE, places=5)  # type: ignore

    def test_hamming_center(self):
        model, result = self._fit(WindowType.HAMMING, NormType.CFT)
        self.assertAlmostEqual(model.center(result.params), self.FREQUENCY, places=7)  # type: ignore


if __name__ == "__main__":
    unittest.main()
