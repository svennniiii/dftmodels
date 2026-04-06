import unittest
import numpy as np
import lmfit

from dftmodels.dft.config import NormType
from dftmodels.dft.correction import DFTCorrection, DFTCorrectionMode
from dftmodels.dft.series import SignalSeries
from dftmodels.models.base import ModelBase
from dftmodels.models.sinusoid import Sinusoid, SineFourier
from dftmodels.models.composite import CompositeModel
from dftmodels.utils.math import linear, linear_complex
    
class GeneralLorentzianModel(ModelBase):
    PARAM_NAMES = ("amplitude", "center", "sigma")

    @staticmethod
    def make_params(
        amplitude: float = 1.0, center: float = 0.0, sigma: float = 1.0, **_
    ) -> lmfit.Parameters:
        params = lmfit.Parameters()
        params.add("amplitude", value=amplitude, min=0)
        params.add("center", value=center)
        params.add("sigma", value=sigma, min=0.001)
        return params

    def eval(self, params: lmfit.Parameters, x: np.ndarray) -> np.ndarray:
        amp = params["amplitude"].value
        cen = params["center"].value
        sig = params["sigma"].value
        return (amp / np.pi) * (sig / ((x - cen) ** 2 + sig**2))


def _make_two_peak_signal(
    freq1: float = 8.0,
    freq2: float = 12.0,
    amp1: float = 2.0,
    amp2: float = 1.5,
    decay1: float = 0.4,
    decay2: float = 0.6,
    n: int = 1500,
    duration: float = 15.0,
) -> tuple[SignalSeries, dict]:
    x = np.linspace(0, duration, n, endpoint=False)
    y = (
        Sinusoid.eval(x, amp1, 0.0, freq1, decay1)
        + Sinusoid.eval(x, amp2, 0.0, freq2, decay2)
    )
    truth = dict(freq1=freq1, freq2=freq2, amp1=amp1, amp2=amp2, decay1=decay1, decay2=decay2)
    return SignalSeries(x=x, y=y), truth


class TestCompositeModelMakeParams(unittest.TestCase):
    def setUp(self):
        signal, _ = _make_two_peak_signal()
        fourier = signal.calculate_dft(pad=2.0, norm=NormType.ASD)
        cfg = fourier.dft_config
        correction = DFTCorrection(mode=DFTCorrectionMode.NONE)
        self.model = CompositeModel([
            ("p1", SineFourier(cfg, correction)),
            ("p2", SineFourier(cfg, correction)),
        ])

    def test_param_names_are_prefixed(self):
        params = self.model.make_params()
        for name in ("p1_amplitude_i", "p1_frequency", "p1_decay",
                     "p2_amplitude_i", "p2_frequency", "p2_decay"):
            self.assertIn(name, params)

    def test_per_prefix_init_values(self):
        params = self.model.make_params(
            p1=SineFourier.make_params(frequency=8.0, decay=0.4),
            p2=SineFourier.make_params(frequency=12.0, decay=0.6),
        )
        self.assertAlmostEqual(params["p1_frequency"].value, 8.0)
        self.assertAlmostEqual(params["p2_frequency"].value, 12.0)

    def test_requires_at_least_one_component(self):
        with self.assertRaises(ValueError):
            CompositeModel([])


class TestCompositeModelEval(unittest.TestCase):
    def setUp(self):
        signal, self.truth = _make_two_peak_signal()
        self.fourier = signal.calculate_dft(pad=2.0, norm=NormType.ASD)
        cfg = self.fourier.dft_config
        correction = DFTCorrection(mode=DFTCorrectionMode.ALL, order=10)
        self.model = CompositeModel([
            ("p1", SineFourier(cfg, correction)),
            ("p2", SineFourier(cfg, correction)),
        ])

    def test_eval_shape(self):
        params = self.model.make_params(
            p1=SineFourier.make_params(amplitude_i=self.truth["amp1"], frequency=self.truth["freq1"],
                    decay=self.truth["decay1"]),
            p2=SineFourier.make_params(amplitude_i=self.truth["amp2"], frequency=self.truth["freq2"],
                    decay=self.truth["decay2"]),
        )
        result = self.model.eval(params, self.fourier.x)
        self.assertEqual(result.shape, self.fourier.x.shape)

    def test_eval_with_background(self):
        cfg = self.fourier.dft_config
        model_with_bg = CompositeModel([
            ("p1", SineFourier(cfg, DFTCorrection(DFTCorrectionMode.ALL, order=10))),
            ("bg", ModelBase.build_model(linear_complex)),
        ])
        params = model_with_bg.make_params(
            p1=SineFourier.make_params(amplitude_i=self.truth["amp1"], frequency=self.truth["freq1"],
                    decay=self.truth["decay1"]),
        )
        result = model_with_bg.eval(params, self.fourier.x)
        self.assertEqual(result.shape, self.fourier.x.shape)


class TestCompositeModelFit(unittest.TestCase):
    def setUp(self):
        signal, self.truth = _make_two_peak_signal()
        self.fourier = signal.calculate_dft(pad=2.0, norm=NormType.ASD)
        cfg = self.fourier.dft_config
        correction = DFTCorrection(mode=DFTCorrectionMode.ALL, order=10)
        self.model = CompositeModel([
            ("p1", SineFourier(cfg, correction)),
            ("p2", SineFourier(cfg, correction)),
        ])

    def test_fit_recovers_two_frequencies(self):
        t = self.truth
        params = self.model.make_params(
            p1=SineFourier.make_params(amplitude_i=t["amp1"] * 1.05, frequency=t["freq1"] * 1.02,
                    decay=t["decay1"] * 1.05),
            p2=SineFourier.make_params(amplitude_i=t["amp2"] * 1.05, frequency=t["freq2"] * 1.02,
                    decay=t["decay2"] * 1.05),
        )
        # Restrict to a frequency window containing both peaks
        mask = (self.fourier.x >= 5.0) & (self.fourier.x <= 15.0)
        result = self.model.fit(self.fourier, params, mask=mask)

        self.assertTrue(result.success)  # type: ignore[union-attr]
        self.assertAlmostEqual(
            result.params["p1_frequency"].value, t["freq1"], places=1  # type: ignore[union-attr]
        )
        self.assertAlmostEqual(
            result.params["p2_frequency"].value, t["freq2"], places=1  # type: ignore[union-attr]
        )


class TestCompositeModelGeneral(unittest.TestCase):
    def test_fit_linear_plus_lorentzian(self):
        x = np.linspace(-10, 10, 200)
        slope_true = 0.5
        intercept_true = 2.0
        amp_true = 5.0
        center_true = 1.5
        sigma_true = 0.5

        rng = np.random.default_rng(seed=43)

        y_linear = slope_true * x + intercept_true
        y_lorentz = (amp_true / np.pi) * (sigma_true / ((x - center_true) ** 2 + sigma_true**2))
        y = y_linear + y_lorentz + rng.normal(0, 0.05, size=x.shape)

        from dftmodels.dft.series import DataSeries
        data = DataSeries(x=x, y=y)

        linear_model = ModelBase.build_model(linear)
        model = CompositeModel([
            ("lin", linear_model),
            ("peak", GeneralLorentzianModel()),
        ])

        params = model.make_params(
            lin=linear_model.make_params(slope=0.4, intercept=1.8),
            peak=GeneralLorentzianModel.make_params(amplitude=4.0, center=1.0, sigma=1.0),
        )

        result = model.fit(data, params)

        self.assertTrue(result.success)  # type: ignore[union-attr]
        self.assertAlmostEqual(result.params["lin_slope"].value, slope_true, places=2)  # type: ignore[union-attr]
        self.assertAlmostEqual(result.params["lin_intercept"].value, intercept_true, places=1)  # type: ignore[union-attr]
        self.assertAlmostEqual(result.params["peak_amplitude"].value, amp_true, places=0)  # type: ignore[union-attr]
        self.assertAlmostEqual(result.params["peak_center"].value, center_true, places=2)  # type: ignore[union-attr]
        self.assertAlmostEqual(result.params["peak_sigma"].value, sigma_true, places=2)  # type: ignore[union-attr]

    def test_mixed_fourier_and_general_model(self):
        signal, truth = _make_two_peak_signal(n=500, duration=5.0)
        fourier = signal.calculate_dft(pad=1.0, norm=NormType.ASD)
        cfg = fourier.dft_config
        
        slope_re = 0.1
        slope_im = -0.05
        intercept_re = 1.0
        intercept_im = 0.5
        
        bg = (intercept_re + 1j * intercept_im) + (slope_re + 1j * slope_im) * fourier.x
        fourier.y += bg

        linear_complex_model = ModelBase.build_model(linear_complex)   
        model = CompositeModel([
            ("p1", SineFourier(cfg, DFTCorrection(DFTCorrectionMode.NONE))),
            ("p2", SineFourier(cfg, DFTCorrection(DFTCorrectionMode.NONE))),
            ("bg", linear_complex_model),
        ])
        
        params = model.make_params(
            p1=SineFourier.make_params(amplitude_i=truth["amp1"], frequency=truth["freq1"], decay=truth["decay1"]),
            p2=SineFourier.make_params(amplitude_i=truth["amp2"], frequency=truth["freq2"], decay=truth["decay2"]),
            bg=linear_complex_model.make_params(real_slope=slope_re, imag_slope=slope_im, real_intercept=intercept_re, imag_intercept=intercept_im)
        )
        
        eval_result = model.eval(params, fourier.x)
        self.assertEqual(eval_result.shape, fourier.x.shape)
        self.assertTrue(np.iscomplexobj(eval_result))
        self.assertLess(np.sum(np.abs(fourier.y - model.eval(params, fourier.x)))/len(fourier.x), 1e-2)


if __name__ == "__main__":
    unittest.main()
