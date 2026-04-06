"""End-to-end integration tests: signal synthesis → DFT → fit → parameter recovery."""
import unittest

import numpy as np

from dftmodels.dft.config import NormType, WindowType, DFTRange
from dftmodels.dft.correction import DFTCorrection, DFTCorrectionMode
from dftmodels.dft.series import SignalSeries
from dftmodels.models.sinusoid import Sinusoid, SineFourier
from dftmodels.models.composite import CompositeModel
from dftmodels.models.base import ModelBase


class TestSinusoidHannEndToEnd(unittest.TestCase):
    AMPLITUDE = 2.0
    FREQUENCY = 10.37
    PHASE = 0.5

    def setUp(self):
        ai = self.AMPLITUDE * np.cos(self.PHASE)
        aq = self.AMPLITUDE * np.sin(self.PHASE)
        rng = np.random.default_rng(seed=7)
        t = np.linspace(0, 4.0, 2000, endpoint=False)
        y = (
            Sinusoid.eval(t, ai, aq, self.FREQUENCY)
            + rng.normal(scale=0.02, size=len(t))
        )
        signal = SignalSeries(x=t, y=y)
        fourier = signal.calculate_dft(
            norm=NormType.ASD,
            window=WindowType.HANN,
            dft_range=DFTRange.SINGLE_SIDED,
            pad=2.0,
        )
        correction = DFTCorrection(DFTCorrectionMode.ALL, order=10)
        self.model = SineFourier(fourier.dft_config, correction)
        params = self.model.make_params(
            amplitude_i=ai * 1.05,
            amplitude_q=aq * 1.05,
            frequency=self.FREQUENCY * 1.01,
        )
        self.result = self.model.fit(fourier, params)

    def test_fit_success(self):
        self.assertTrue(self.result.success)  # type: ignore[attr-defined]

    def test_frequency_recovery(self):
        self.assertAlmostEqual(self.model.center(self.result.params), self.FREQUENCY, places=2)  # type: ignore 

    def test_amplitude_recovery(self):
        self.assertAlmostEqual(
            self.model.amplitude(self.result.params), self.AMPLITUDE, places=2  # type: ignore
        )

    def test_phase_recovery(self):
        self.assertAlmostEqual(self.model.phase(self.result.params), self.PHASE, places=2)  # type: ignore


class TestSinusoidHammingEndToEnd(unittest.TestCase):
    AMPLITUDE = 2.0
    FREQUENCY = 10.37
    PHASE = 0.5

    def setUp(self):
        ai = self.AMPLITUDE * np.cos(self.PHASE)
        aq = self.AMPLITUDE * np.sin(self.PHASE)
        rng = np.random.default_rng(seed=42)
        t = np.linspace(0, 4.0, 2000, endpoint=False)
        y = (
            Sinusoid.eval(t, ai, aq, self.FREQUENCY)
            + rng.normal(scale=0.02, size=len(t))
        )
        signal = SignalSeries(x=t, y=y)
        fourier = signal.calculate_dft(
            norm=NormType.ASD,
            window=WindowType.HAMMING,
            dft_range=DFTRange.SINGLE_SIDED,
            pad=2.0,
        )
        correction = DFTCorrection(DFTCorrectionMode.ALL, order=10)
        self.model = SineFourier(fourier.dft_config, correction)
        params = self.model.make_params(
            amplitude_i=ai * 1.05,
            amplitude_q=aq * 1.05,
            frequency=self.FREQUENCY * 1.01,
        )
        self.result = self.model.fit(fourier, params)

    def test_fit_success(self):
        self.assertTrue(self.result.success)  # type: ignore[attr-defined]

    def test_frequency_recovery(self):
        self.assertAlmostEqual(self.model.center(self.result.params), self.FREQUENCY, places=4)  # type: ignore

    def test_amplitude_recovery(self):
        self.assertAlmostEqual(
            self.model.amplitude(self.result.params), self.AMPLITUDE, places=2  # type: ignore
        )

    def test_phase_recovery(self):
        self.assertAlmostEqual(self.model.phase(self.result.params), self.PHASE, places=3)  # type: ignore


class TestSinusoidRectEndToEnd(unittest.TestCase):
    AMPLITUDE = 2.0
    FREQUENCY = 10.37
    PHASE = 0.5

    def setUp(self):
        ai = self.AMPLITUDE * np.cos(self.PHASE)
        aq = self.AMPLITUDE * np.sin(self.PHASE)
        t = np.linspace(0, 5.0, 2500, endpoint=False)
        y = Sinusoid.eval(t, ai, aq, self.FREQUENCY)
        signal = SignalSeries(x=t, y=y)
        fourier = signal.calculate_dft(
            norm=NormType.ASD,
            window=WindowType.RECTANGULAR,
            dft_range=DFTRange.DOUBLE_SIDED,
            pad=2.0,
        )

        self.model = SineFourier(fourier.dft_config)
        params = self.model.make_params(
            amplitude_i=ai * 1.05,
            amplitude_q=aq * 1.05,
            frequency=self.FREQUENCY * 1.01,
        )
        self.result = self.model.fit(fourier, params)

    def test_fit_success(self):
        self.assertTrue(self.result.success)  # type: ignore[attr-defined]

    def test_frequency_recovery(self):
        self.assertAlmostEqual(self.model.center(self.result.params), self.FREQUENCY, places=4)  # type: ignore

    def test_amplitude_recovery(self):
        self.assertAlmostEqual(
            self.model.amplitude(self.result.params), self.AMPLITUDE, places=3  # type: ignore
        )


class TestLorentzianEndToEnd(unittest.TestCase):
    AMPLITUDE = 2.0
    FREQUENCY = 10.37
    PHASE = 0.5
    DECAY = 0.5

    def setUp(self):
        ai = self.AMPLITUDE * np.cos(self.PHASE)
        aq = self.AMPLITUDE * np.sin(self.PHASE)
        t = np.linspace(0, 50.0, 5000, endpoint=False)
        y = Sinusoid.eval(t, ai, aq, self.FREQUENCY, self.DECAY)
        signal = SignalSeries(x=t, y=y)
        fourier = signal.calculate_dft(
            norm=NormType.ASD,
            window=WindowType.RECTANGULAR,
            dft_range=DFTRange.DOUBLE_SIDED,
            pad=2.0,
        )
        correction = DFTCorrection(DFTCorrectionMode.ALL, order=20)
        self.model = SineFourier(fourier.dft_config, correction)
        params = self.model.make_params(
            amplitude_i=ai * 1.05,
            amplitude_q=aq * 1.05,
            frequency=self.FREQUENCY * 1.02,
            decay=self.DECAY * 1.1,
        )
        self.result = self.model.fit(fourier, params)

    def test_fit_success(self):
        self.assertTrue(self.result.success)  # type: ignore[attr-defined]

    def test_frequency_recovery(self):
        self.assertAlmostEqual(self.model.center(self.result.params), self.FREQUENCY, places=5)  # type: ignore

    def test_amplitude_recovery(self):
        self.assertAlmostEqual(
            self.model.amplitude(self.result.params), self.AMPLITUDE, places=3  # type: ignore
        )

    def test_decay_recovery(self):
        self.assertAlmostEqual(
            self.result.params["decay"].value, self.DECAY, places=4  # type: ignore[attr-defined]
        )

    def test_phase_recovery(self):
        self.assertAlmostEqual(self.model.phase(self.result.params), self.PHASE, places=4)  # type: ignore


class TestCompositeEndToEnd(unittest.TestCase):
    F1, A1, D1 = 8.0, 2.0, 0.4
    F2, A2, D2 = 12.0, 1.5, 0.6

    def setUp(self):
        t = np.linspace(0, 30.0, 3000, endpoint=False)
        y = Sinusoid.eval(t, self.A1, 0.0, self.F1, self.D1) + Sinusoid.eval(
            t, self.A2, 0.0, self.F2, self.D2
        )
        signal = SignalSeries(x=t, y=y)
        fourier = signal.calculate_dft(
            norm=NormType.ASD,
            window=WindowType.RECTANGULAR,
            dft_range=DFTRange.DOUBLE_SIDED,
            pad=1.0,
        )
        cfg = fourier.dft_config
        correction = DFTCorrection(DFTCorrectionMode.ALL, order=10)
        from dftmodels.utils.math import linear_complex
        linear_complex_model = ModelBase.build_model(linear_complex)
        self.model = CompositeModel([
            ("p1", SineFourier(cfg, correction)),
            ("p2", SineFourier(cfg, correction)),
            ("bg", linear_complex_model),
        ])
        params = self.model.make_params(
            p1=SineFourier.make_params(
                amplitude_i=self.A1 * 1.05,
                amplitude_q=0.0,
                frequency=self.F1 * 1.02,
                decay=self.D1 * 1.1,
            ),
            p2=SineFourier.make_params(
                amplitude_i=self.A2 * 1.05,
                amplitude_q=0.0,
                frequency=self.F2 * 1.02,
                decay=self.D2 * 1.1,
            ),
            bg=linear_complex_model.make_params(),
        )
        mask = (np.abs(fourier.x - self.F1) <= 3.0) | (
            np.abs(fourier.x - self.F2) <= 3.0
        )
        self.result = self.model.fit(fourier, params, mask=mask)

    def test_fit_success(self):
        self.assertTrue(self.result.success)  # type: ignore[attr-defined]

    def test_peak1_frequency(self):
        self.assertAlmostEqual(
            self.result.params["p1_frequency"].value, self.F1, places=7  # type: ignore[attr-defined]
        )

    def test_peak2_frequency(self):
        self.assertAlmostEqual(
            self.result.params["p2_frequency"].value, self.F2, places=7  # type: ignore[attr-defined]
        )

    def test_peak1_decay(self):
        self.assertAlmostEqual(
            self.result.params["p1_decay"].value, self.D1, places=7  # type: ignore[attr-defined]
        )

    def test_peak2_decay(self):
        self.assertAlmostEqual(
            self.result.params["p2_decay"].value, self.D2, places=7  # type: ignore[attr-defined]
        )


class TestParsevalWithCorrections(unittest.TestCase):
    def test_parseval_single_and_double_sided(self):
        amplitude = 2.0
        frequency = 10.37
        decay = 0.5
        t = np.linspace(0, 10.0, 1000, endpoint=False)
        y = Sinusoid.eval(t, amplitude, 0.0, frequency, decay)
        signal = SignalSeries(x=t, y=y)
        true_ms = float(np.mean(y**2))

        for dft_range in [DFTRange.SINGLE_SIDED, DFTRange.DOUBLE_SIDED]:
            fourier = signal.calculate_dft(
                norm=NormType.ASD,
                window=WindowType.RECTANGULAR,
                dft_range=dft_range,
                pad=1.0,
            )
            correction = DFTCorrection(DFTCorrectionMode.ALL, order=10)
            model = SineFourier(fourier.dft_config, correction)
            params = model.make_params(
                amplitude_i=amplitude,
                amplitude_q=0.0,
                frequency=frequency,
                decay=decay,
            )
            
            model_y = model.eval(params, fourier.x)
            model_fourier = fourier.copy()
            model_fourier.y = model_y
            
            psd = model_fourier.convert_to_psd()
            model_power = psd.calculate_integral()
            
            self.assertAlmostEqual(model_power, true_ms, places=3)


if __name__ == "__main__":
    unittest.main()
