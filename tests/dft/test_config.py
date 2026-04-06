import unittest
import numpy as np

from dftmodels.dft.config import (
    DFTConfig, NormType, DFTRange, WindowType, get_window_array
)


class TestWindowArray(unittest.TestCase):
    def test_rectangular_is_ones(self):
        w = get_window_array(WindowType.RECTANGULAR, 100)
        np.testing.assert_array_equal(w, np.ones(100))

    def test_hamming_length(self):
        w = get_window_array(WindowType.HAMMING, 64)
        self.assertEqual(len(w), 64)

    def test_hann_length(self):
        w = get_window_array(WindowType.HANN, 64)
        self.assertEqual(len(w), 64)

    def test_hann_zero_endpoints(self):
        # Hann window is zero at first and last sample
        w = get_window_array(WindowType.HANN, 64)
        self.assertAlmostEqual(w[0], 0.0, places=10)

    def test_dirichlet_is_rectangular_alias(self):
        # DIRICHLET is an alias for RECTANGULAR
        self.assertIs(WindowType.DIRICHLET, WindowType.RECTANGULAR)


class TestDFTConfigValidation(unittest.TestCase):
    def _valid(self):
        return DFTConfig(number_of_samples=100, sample_rate=1000.0)

    def test_valid_config(self):
        cfg = self._valid()
        self.assertEqual(cfg.number_of_samples, 100)
        self.assertEqual(cfg.sample_rate, 1000.0)

    def test_negative_samples_raises(self):
        with self.assertRaises(ValueError):
            DFTConfig(number_of_samples=-1, sample_rate=1000.0)

    def test_zero_sample_rate_raises(self):
        with self.assertRaises(ValueError):
            DFTConfig(number_of_samples=100, sample_rate=0.0)

    def test_pad_below_one_raises(self):
        with self.assertRaises(ValueError):
            DFTConfig(number_of_samples=100, sample_rate=1000.0, pad=0.5)

    def test_hann_cft_raises_via_norm_factor(self):
        # CFT norm with Hann window is invalid (first window value is 0)
        cfg = DFTConfig(
            number_of_samples=100,
            sample_rate=1000.0,
            window=WindowType.HANN,
            norm_type=NormType.CFT,
        )
        with self.assertRaises(ValueError):
            _ = cfg.norm_factor


class TestDFTConfigProperties(unittest.TestCase):
    def setUp(self):
        self.n = 1000
        self.sr = 1000.0
        self.cfg = DFTConfig(number_of_samples=self.n, sample_rate=self.sr)

    def test_number_of_samples_fft_no_pad(self):
        self.assertEqual(self.cfg.number_of_samples_fft, self.n)

    def test_number_of_samples_fft_with_pad(self):
        cfg = DFTConfig(number_of_samples=self.n, sample_rate=self.sr, pad=2.0)
        self.assertEqual(cfg.number_of_samples_fft, 2 * self.n)

    def test_frequency_step(self):
        self.assertAlmostEqual(self.cfg.frequency_step, self.sr / self.n)

    def test_double_sided_frequency_range(self):
        self.assertLess(self.cfg.frequency_min, 0)
        self.assertGreater(self.cfg.frequency_max, 0)

    def test_single_sided_frequency_range(self):
        cfg = DFTConfig(
            number_of_samples=self.n,
            sample_rate=self.sr,
            dft_range=DFTRange.SINGLE_SIDED,
        )
        self.assertAlmostEqual(cfg.frequency_min, 0.0)
        self.assertAlmostEqual(cfg.frequency_max, self.sr / 2)

    def test_norm_factor_cft_positive(self):
        self.assertGreater(self.cfg.norm_factor, 0)

    def test_norm_factor_asd_positive(self):
        cfg = DFTConfig(
            number_of_samples=self.n,
            sample_rate=self.sr,
            norm_type=NormType.ASD,
        )
        self.assertGreater(cfg.norm_factor, 0)

    def test_psd_norm_equals_asd_squared(self):
        asd_cfg = DFTConfig(
            number_of_samples=self.n, sample_rate=self.sr, norm_type=NormType.ASD
        )
        psd_cfg = DFTConfig(
            number_of_samples=self.n, sample_rate=self.sr, norm_type=NormType.PSD
        )
        self.assertAlmostEqual(psd_cfg.norm_factor, asd_cfg.norm_factor ** 2)

    def test_copy_is_equal(self):
        cfg2 = self.cfg.copy()
        self.assertEqual(cfg2.number_of_samples, self.cfg.number_of_samples)
        self.assertEqual(cfg2.sample_rate, self.cfg.sample_rate)
        self.assertEqual(cfg2.norm_type, self.cfg.norm_type)


if __name__ == "__main__":
    unittest.main()
