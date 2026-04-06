import unittest
import numpy as np
from dftmodels.utils.math import (
    sinc,
    lorentzian,
    gaussian,
    voigt,
    linear,
    linear_complex,
    exponential,
)


class TestSinc(unittest.TestCase):
    def test_peak_at_center(self):
        # sinc(center) = amplitude * sinc(0) = amplitude
        self.assertAlmostEqual(sinc(0.0, amplitude=3.0, center=0.0, width=1.0), 3.0)

    def test_peak_off_center(self):
        self.assertAlmostEqual(sinc(2.0, amplitude=3.0, center=2.0, width=1.0), 3.0)

    def test_zero_crossing(self):
        # sinc(x) = 0 at x = ±width from center (numpy sinc uses normalized sinc)
        result = sinc(1.0, amplitude=1.0, center=0.0, width=1.0)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_array_input(self):
        x = np.array([0.0, 1.0, -1.0])
        y = sinc(x, amplitude=2.0, center=0.0, width=1.0)
        self.assertEqual(y.shape, (3,))
        self.assertAlmostEqual(y[0], 2.0)
        self.assertAlmostEqual(y[1], 0.0, places=10)


class TestLorentzian(unittest.TestCase):
    def test_peak_at_center(self):
        # lorentzian(center) = amplitude / (pi * width)
        result = lorentzian(0.0, amplitude=np.pi, center=0.0, width=1.0)
        self.assertAlmostEqual(result, 1.0)

    def test_half_maximum_at_width(self):
        # at x = center ± width, value = half of peak
        peak = lorentzian(0.0, amplitude=1.0, center=0.0, width=1.0)
        half = lorentzian(1.0, amplitude=1.0, center=0.0, width=1.0)
        self.assertAlmostEqual(half, peak / 2.0)

    def test_symmetry(self):
        y_pos = lorentzian(2.0, amplitude=1.0, center=0.0, width=0.5)
        y_neg = lorentzian(-2.0, amplitude=1.0, center=0.0, width=0.5)
        self.assertAlmostEqual(y_pos, y_neg)

    def test_array_input(self):
        x = np.linspace(-5, 5, 100)
        y = lorentzian(x, amplitude=1.0, center=0.0, width=1.0)
        self.assertEqual(y.shape, (100,))
        self.assertTrue(np.all(y > 0))


class TestGaussian(unittest.TestCase):
    def test_peak_at_center(self):
        # gaussian(center) = amplitude / sqrt(2*pi*width^2)
        width = 1.0
        expected = 1.0 / np.sqrt(2 * np.pi * width ** 2)
        result = gaussian(0.0, amplitude=1.0, center=0.0, width=width)
        self.assertAlmostEqual(result, expected)

    def test_symmetry(self):
        y_pos = gaussian(1.0, amplitude=1.0, center=0.0, width=2.0)
        y_neg = gaussian(-1.0, amplitude=1.0, center=0.0, width=2.0)
        self.assertAlmostEqual(y_pos, y_neg)

    def test_integral_equals_amplitude(self):
        # ∫ gaussian dx = amplitude
        x = np.linspace(-50, 50, 100000)
        y = gaussian(x, amplitude=5.0, center=0.0, width=1.0)
        integral = np.trapezoid(y, x)
        self.assertAlmostEqual(integral, 5.0, places=3)


class TestVoigt(unittest.TestCase):
    def test_peak_positive(self):
        result = voigt(0.0, amplitude=1.0, center=0.0, width_lorentzian=0.5, width_gaussian=0.5)
        self.assertGreater(result, 0.0)

    def test_symmetry(self):
        y_pos = voigt(1.0, amplitude=1.0, center=0.0, width_lorentzian=0.5, width_gaussian=0.5)
        y_neg = voigt(-1.0, amplitude=1.0, center=0.0, width_lorentzian=0.5, width_gaussian=0.5)
        self.assertAlmostEqual(y_pos, y_neg)

    def test_pure_lorentzian_limit(self):
        # Very small gaussian width → approaches Lorentzian shape
        x = np.array([0.0, 1.0])
        y_voigt = voigt(x, amplitude=np.pi, center=0.0, width_lorentzian=1.0, width_gaussian=1e-6)
        y_lor = lorentzian(x, amplitude=np.pi, center=0.0, width=1.0)
        np.testing.assert_allclose(y_voigt, y_lor, rtol=1e-4)


class TestLinear(unittest.TestCase):
    def test_slope_and_intercept(self):
        self.assertAlmostEqual(linear(2.0, slope=3.0, intercept=1.0), 7.0)
        self.assertAlmostEqual(linear(0.0, slope=3.0, intercept=1.0), 1.0)

    def test_zero_slope(self):
        x = np.linspace(-5, 5, 20)
        y = linear(x, slope=0.0, intercept=4.0)
        np.testing.assert_array_almost_equal(y, 4.0)

    def test_array_input(self):
        x = np.array([0.0, 1.0, 2.0])
        y = linear(x, slope=2.0, intercept=1.0)
        np.testing.assert_array_almost_equal(y, [1.0, 3.0, 5.0])


class TestLinearComplex(unittest.TestCase):
    def test_at_origin(self):
        result = linear_complex(0.0, real_intercept=1.0, imag_intercept=2.0,
                                real_slope=0.0, imag_slope=0.0)
        self.assertAlmostEqual(result.real, 1.0)
        self.assertAlmostEqual(result.imag, 2.0)

    def test_slope_contribution(self):
        result = linear_complex(1.0, real_intercept=0.0, imag_intercept=0.0,
                                real_slope=3.0, imag_slope=-1.0)
        self.assertAlmostEqual(result.real, 3.0)
        self.assertAlmostEqual(result.imag, -1.0)

    def test_array_input(self):
        x = np.array([0.0, 1.0, 2.0])
        y = linear_complex(x, real_intercept=1.0, imag_intercept=0.0,
                           real_slope=1.0, imag_slope=0.0)
        np.testing.assert_array_almost_equal(y.real, [1.0, 2.0, 3.0])


class TestExponential(unittest.TestCase):
    def test_at_zero(self):
        self.assertAlmostEqual(exponential(0.0, amplitude=5.0, decay=2.0), 5.0)

    def test_decay(self):
        # amplitude * exp(decay * x): with negative decay, value decreases
        result = exponential(1.0, amplitude=1.0, decay=-1.0)
        self.assertAlmostEqual(result, np.exp(-1.0))

    def test_growth(self):
        result = exponential(2.0, amplitude=1.0, decay=1.0)
        self.assertAlmostEqual(result, np.exp(2.0))

    def test_array_input(self):
        x = np.array([0.0, 1.0, 2.0])
        y = exponential(x, amplitude=2.0, decay=0.0)
        np.testing.assert_array_almost_equal(y, [2.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
