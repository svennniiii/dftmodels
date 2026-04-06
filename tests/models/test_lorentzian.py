import unittest
from itertools import product

import numpy as np

from dftmodels.dft.config import DFTRange, NormType, WindowType
from dftmodels.dft.correction import DFTCorrection, DFTCorrectionMode
from dftmodels.dft.series import SignalSeries
from dftmodels.models.sinusoid import Sinusoid, SineFourier 


def _make_decaying_signal(
    amplitude: float = 2.0,
    frequency: float = 10.37,
    phase: float = 0.5,
    decay: float = 0.5,
    n: int = 1000,
    duration: float = 10.0,
) -> tuple[SignalSeries, float, float]:
    x = np.linspace(0, duration, n, endpoint=False)
    ai = amplitude * np.cos(phase)
    aq = amplitude * np.sin(phase)
    y = Sinusoid.eval(x, ai, aq, frequency, decay)
    return SignalSeries(x=x, y=y), ai, aq


class TestSinusoidWithDecayEval(unittest.TestCase):
    def test_zero_decay_matches_sinusoid(self):
        from dftmodels.models.sinusoid import Sinusoid
        x = np.linspace(0, 1, 500, endpoint=False)
        ai, aq, freq, decay = 2.0, 1.0, 5.0, 0.0
        y_decaying = Sinusoid.eval(x, ai, aq, freq, decay)
        y_sinusoid = Sinusoid.eval(x, ai, aq, freq)
        np.testing.assert_array_almost_equal(y_decaying, y_sinusoid)

    def test_decay_reduces_amplitude(self):
        x = np.linspace(0, 10, 1000, endpoint=False)
        y = Sinusoid.eval(x, 1.0, 0.0, 1.0, 1.0)
        # Amplitude at t=5 should be much less than at t=0
        self.assertLess(abs(y[-1]), abs(y[0]))


class TestSinusoidWithDecayFit(unittest.TestCase):
    def setUp(self):
        self.signal, self.ai, self.aq = _make_decaying_signal()
        self.frequency = 10.37
        self.decay = 0.5
        rng = np.random.default_rng(seed=42)
        self.signal.y += rng.normal(scale=1e-4, size=len(self.signal))

    def test_fit_recovers_frequency_and_decay(self):
        model = Sinusoid()
        params = model.make_params(
            amplitude_i=self.ai * 1.05,
            amplitude_q=self.aq * 1.05,
            frequency=self.frequency * 1.02,
            decay=self.decay * 1.1,
        )
        result = model.fit(self.signal.x, self.signal.y, params)
        self.assertTrue(result.success)  # type: ignore[union-attr]
        self.assertAlmostEqual(result.params["frequency"].value, self.frequency, places=2)  # type: ignore[union-attr]
        self.assertAlmostEqual(result.params["decay"].value, self.decay, places=2)  # type: ignore[union-attr]


class TestSineFourierWithDecayCorrections(unittest.TestCase):
    def setUp(self):
        self.frequency = 10.37
        self.decay = 0.5
        self.signal, self.ai, self.aq = _make_decaying_signal(
            frequency=self.frequency, decay=self.decay, duration=10.0
        )

    def _error(self, fourier, dft_correction):
        model = SineFourier(
            dft_config=fourier.dft_config,
            dft_correction=dft_correction,
        )
        params = model.make_params(
            amplitude_i=self.ai,
            amplitude_q=self.aq,
            frequency=self.frequency,
            decay=self.decay,
        )
        predicted = model.eval(params, fourier.x)
        return float(np.sum(np.abs(fourier.y - predicted)))

    def test_window_correction_reduces_error(self):
        no_correction = DFTCorrection(mode=DFTCorrectionMode.NONE, order=0)
        with_window = DFTCorrection(mode=DFTCorrectionMode.ALL, order=10)

        pads = (1.0, 2.0)
        norms = (NormType.CFT, NormType.ASD)
        ranges = tuple(DFTRange)

        for pad, norm, dft_range in product(pads, norms, ranges):
            fourier = self.signal.calculate_dft(
                pad=pad, norm=norm,
                window=WindowType.RECTANGULAR,
                dft_range=dft_range,
            )

            error_none = self._error(fourier, no_correction)
            error_window = self._error(fourier, with_window)
            
            self.assertGreater(
                error_none, error_window,
                msg=f"WINDOW correction did not reduce error: pad={pad}, norm={norm}, "
                    f"range={dft_range}"
            )


class TestSineFourierWithDecayAttributes(unittest.TestCase):
    def setUp(self):
        signal, ai, aq = _make_decaying_signal(amplitude=2.0, decay=0.5)
        fourier = signal.calculate_dft(
            pad=2.0, norm=NormType.CFT, window=WindowType.RECTANGULAR
        )
        model = SineFourier(
            dft_config=fourier.dft_config,
            dft_correction=DFTCorrection(mode=DFTCorrectionMode.ALL, order=10),
        )
        params = model.make_params(
            amplitude_i=ai * 1.05,
            amplitude_q=aq * 1.05,
            frequency=10.37 * 1.02,
            decay=0.5 * 1.1,
        )
        self.result = model.fit(fourier, params)
        self.model = model

    def test_fwhm_is_decay_over_pi(self):
        decay = self.result.params["decay"].value  # type: ignore[union-attr]
        fwhm = self.model.fwhm(self.result.params)  # type: ignore
        self.assertAlmostEqual(fwhm, decay / np.pi, places=10)

    def test_center_matches_frequency(self):
        freq = self.result.params["frequency"].value  # type: ignore[union-attr]
        self.assertAlmostEqual(self.model.center(self.result.params), freq, places=10)  # type: ignore


class TestSineFourierWithDecayFitAccuracy(unittest.TestCase):
    AMPLITUDE = 2.0
    FREQUENCY = 10.37
    PHASE = 0.5
    DECAY = 0.5

    def setUp(self):
        self.ai = self.AMPLITUDE * np.cos(self.PHASE)
        self.aq = self.AMPLITUDE * np.sin(self.PHASE)
        signal, _, _ = _make_decaying_signal(
            amplitude=self.AMPLITUDE,
            frequency=self.FREQUENCY,
            phase=self.PHASE,
            decay=self.DECAY,
            n=128,
            duration=4.0,
        )
        fourier = signal.calculate_dft(
            pad=2.0, norm=NormType.CFT, window=WindowType.RECTANGULAR
        )
        correction = DFTCorrection(mode=DFTCorrectionMode.ALL, order=50)
        self.model = SineFourier(
            dft_config=fourier.dft_config,
            dft_correction=correction,
        )
        params = self.model.make_params(
            amplitude_i=self.ai * 1.02,
            amplitude_q=self.aq * 1.02,
            frequency=self.FREQUENCY * 1.005,
            decay=self.DECAY * 1.05,
        )
        self.result = self.model.fit(fourier, params)

    def test_fit_success(self):
        self.assertTrue(self.result.success)  # type: ignore[union-attr]

    def test_frequency_recovery(self):
        self.assertAlmostEqual(
            self.result.params["frequency"].value, self.FREQUENCY, places=3  # type: ignore[union-attr]
        )

    def test_decay_recovery(self):
        self.assertAlmostEqual(
            self.result.params["decay"].value, self.DECAY, places=3  # type: ignore[union-attr]
        )

    def test_amplitude_recovery(self):
        self.assertAlmostEqual(self.model.amplitude(self.result.params), self.AMPLITUDE, places=3)  # type: ignore

    def test_phase_recovery(self):
        self.assertAlmostEqual(self.model.phase(self.result.params), self.PHASE, places=3)  # type: ignore


class TestFourierModelBaseFitOptions(unittest.TestCase):
    def setUp(self):
        signal, ai, aq = _make_decaying_signal(amplitude=2.0, decay=0.5, n=1000, duration=10.0)
        fourier = signal.calculate_dft(
            pad=2.0, norm=NormType.ASD, window=WindowType.RECTANGULAR
        )
        self.fourier = fourier
        self.ai = ai
        self.aq = aq
        self.model = SineFourier(
            dft_config=fourier.dft_config,
            dft_correction=DFTCorrection(mode=DFTCorrectionMode.ALL, order=10),
        )

    def _base_params(self):
        return self.model.make_params(
            amplitude_i=self.ai * 1.02,
            amplitude_q=self.aq * 1.02,
            frequency=10.37 * 1.005,
            decay=0.5 * 1.05,
        )

    def test_fit_with_mask(self):
        mask = (self.fourier.x >= 9.0) & (self.fourier.x <= 12.0)
        result = self.model.fit(self.fourier, self._base_params(), mask=mask)
        self.assertTrue(result.success)  # type: ignore[union-attr]
        self.assertAlmostEqual(result.params["frequency"].value, 10.37, places=2)  # type: ignore[union-attr]

    def test_fit_with_scalar_weights(self):
        result = self.model.fit(self.fourier, self._base_params(), weights=2.0)
        self.assertTrue(result.success)  # type: ignore[union-attr]

    def test_fit_with_array_weights(self):
        weights = np.ones(len(self.fourier))
        weights[len(self.fourier) // 2 :] = 0.1  # down-weight upper half
        result = self.model.fit(self.fourier, self._base_params(), weights=weights)
        self.assertTrue(result.success)  # type: ignore[union-attr]

    def test_fit_with_baseline(self):
        # 1. Create a significant baseline (approx 25% of signal amplitude)
        baseline_val = 0.5 + 0.2j
        baseline = lambda x: baseline_val * np.ones_like(x)  # noqa: E731

        # 2. Add it to the data
        self.fourier.y += baseline(self.fourier.x)

        # 3. Fit with the baseline
        result = self.model.fit(self.fourier, self._base_params(), baseline=baseline)

        # 4. Verify parameter recovery
        self.assertTrue(result.success)  # type: ignore[union-attr]
        self.assertAlmostEqual(result.params["frequency"].value, 10.37, places=2)  # type: ignore[union-attr]
        self.assertAlmostEqual(result.params["decay"].value, 0.5, places=2)  # type: ignore[union-attr]
        self.assertAlmostEqual(result.params["amplitude_i"].value, self.ai, places=2)  # type: ignore[union-attr]
        self.assertAlmostEqual(result.params["amplitude_q"].value, self.aq, places=2)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
