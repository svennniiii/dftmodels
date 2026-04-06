import unittest
from itertools import product

import numpy as np
from numpy.testing import assert_array_almost_equal

from dftmodels.dft.config import DFTConfig, DFTRange, NormType, WindowType, get_window_array
from dftmodels.dft.series import SignalSeries, FourierSeries


def _make_signal(n=100, sr=100.0, freq=10.0, amplitude=2.0):
    t = np.linspace(0, n / sr, n, endpoint=False)
    y = amplitude * np.cos(2 * np.pi * freq * t)
    return SignalSeries(x=t, y=y)


class TestSignalSeriesConstruction(unittest.TestCase):
    def test_basic(self):
        s = _make_signal()
        self.assertEqual(len(s), 100)

    def test_sample_rate(self):
        s = _make_signal(n=100, sr=500.0)
        self.assertAlmostEqual(s.sample_rate, 500.0)

    def test_complex_x_raises(self):
        with self.assertRaises(ValueError):
            SignalSeries(x=np.array([1 + 0j, 2 + 0j]), y=np.array([1.0, 2.0]))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            SignalSeries(x=np.arange(10, dtype=float), y=np.arange(5, dtype=float))

    def test_copy_independence(self):
        s = _make_signal()
        s2 = s.copy()
        s2.y[0] = 999.0
        self.assertNotEqual(s.y[0], 999.0)

    def test_rms(self):
        # Pure cosine: RMS = amplitude / sqrt(2)
        amplitude = 3.0
        t = np.linspace(0, 1, 10000, endpoint=False)
        y = amplitude * np.cos(2 * np.pi * 10 * t)
        s = SignalSeries(x=t, y=y)
        self.assertAlmostEqual(s.calculate_rms(), amplitude / np.sqrt(2), places=3)


class TestSignalSeriesEmptyDFT(unittest.TestCase):
    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            SignalSeries(x=np.array([]), y=np.array([])).calculate_dft()


class TestDFTPaddedLength(unittest.TestCase):
    def test_padded_length(self):
        s = _make_signal(n=100)
        f = s.calculate_dft(pad=2.0)
        self.assertEqual(len(f), 200)

    def test_single_sided_length(self):
        n = 100
        s = _make_signal(n=n)
        f = s.calculate_dft(dft_range=DFTRange.SINGLE_SIDED)
        self.assertEqual(len(f), n // 2 + 1)

    def test_complex_signal_single_sided_raises(self):
        t = np.linspace(0, 1, 100, endpoint=False)
        s = SignalSeries(x=t, y=t.astype(complex) + 1j)
        with self.assertRaises(ValueError):
            s.calculate_dft(dft_range=DFTRange.SINGLE_SIDED)


class TestASDNormalization(unittest.TestCase):
    def setUp(self):
        n = 100_000
        sr = 100.0
        t = np.linspace(0, n / sr, n, endpoint=False)
        y = 2.0 * np.cos(2 * np.pi * 10.37 * t)
        self.signal = SignalSeries(x=t, y=y)
        self.y = y
        self.n = n

    def test_parseval_windowed(self):
        """∫|ASD|² df equals the windowed signal power for all windows, pads, and ranges.

        This is the exact identity that the ASD normalization guarantees.  The
        windowed power is ∑(w[n]·x[n])²/N, which equals the original signal's RMS²
        only asymptotically (large N, stationary signal).  See
        test_parseval_original_rms_asymptotic for the approximate version.
        """
        window_configs = [
            (WindowType.RECTANGULAR,      {}),
            (WindowType.HAMMING,          {}),
            (WindowType.HANN,             {}),
            (WindowType.BLACKMAN,         {}),
            (WindowType.NUTTAL,           {}),
            (WindowType.BLACKMAN_NUTTAL,  {}),
            (WindowType.FLAT_TOP,         {}),
            (WindowType.BARTLETT,         {}),
            (WindowType.EXPONENTIAL_ASYM, {"alpha": 3.0}),
        ]
        norms  = (NormType.ASD, NormType.ASD_ABS)
        pads   = (1, 2, 4.5)
        ranges = tuple(DFTRange)

        for norm, (window, wp), pad, dft_range in product(norms, window_configs, pads, ranges):
            w_arr          = get_window_array(window, self.n, **wp)
            windowed_power = float(np.sum((w_arr * self.y) ** 2) / self.n)
            window_rms = np.sqrt(np.mean(w_arr**2))

            with self.subTest(norm=norm, window=window, pad=pad, range=dft_range):
                asd = self.signal.calculate_dft(
                    norm=norm, window=window, pad=pad, dft_range=dft_range,
                    window_params=wp if wp else None,
                )
                psd       = asd.convert_to_psd()
                power_psd = float(np.real(psd.calculate_integral()))
                self.assertAlmostEqual(
                    power_psd, windowed_power / window_rms**2, places=10,
                    msg=f"Parseval (windowed) failed: norm={norm}, window={window}, "
                        f"pad={pad}, range={dft_range}",
                )

    def test_parseval_original_rms_asymptotic(self):
        """For a long signal with many integer periods, ∫|ASD|² df approximates RMS².

        This is not an exact identity: the window_rms correction assumes the window
        and signal are uncorrelated, which holds only for stationary processes as N → ∞.
        The tolerance here is intentionally loose to reflect the asymptotic nature of
        the check.
        """

        original_power = float(np.sum(self.y ** 2) / self.n)   # = A²/2 = 2.0

        for window in tuple(WindowType):
            if window == WindowType.EXPONENTIAL_ASYM:
                continue
            with self.subTest(window=window):
                asd = self.signal.calculate_dft(
                    norm=NormType.ASD, window=window, pad=1.0,
                    dft_range=DFTRange.SINGLE_SIDED,
                )
                power_psd = float(np.real(asd.convert_to_psd().calculate_integral()))
                self.assertAlmostEqual(
                    power_psd, original_power, places=8,
                    msg=f"Asymptotic Parseval failed for {window}",
                )


class TestCFTNormalization(unittest.TestCase):
    def setUp(self):
        n = 99
        sr = 100.0
        t = np.linspace(0, n / sr, n, endpoint=False)
        y = 2.0 * np.cos(2 * np.pi * 10 * t)
        self.signal = SignalSeries(x=t, y=y)
        self.s0 = y[0]

    def test_cft_integral_all_windows_pads_ranges(self):
        # The identity ∫ S(f) df = s(0) is a two-sided transform property;
        # single-sided DFT is excluded here.
        windows = (
            WindowType.RECTANGULAR, WindowType.HAMMING,
            WindowType.BLACKMAN, WindowType.BLACKMAN_NUTTAL, WindowType.FLAT_TOP,
        )  # not HANN, BARTLETT or NUTTAL (zero first sample)
        pads = (1, 2, 4.5)

        for window, pad in product(windows, pads):
            with self.subTest(window=window, pad=pad):
                dft = self.signal.calculate_dft(
                    norm=NormType.CFT, window=window, pad=pad,
                    dft_range=DFTRange.DOUBLE_SIDED,
                )
                s0_dft = dft.real.calculate_integral()
                self.assertAlmostEqual(float(s0_dft), self.s0, places=10)

    def test_hann_cft_raises(self):
        with self.assertRaises(ValueError):
            self.signal.calculate_dft(norm=NormType.CFT, window=WindowType.HANN)


class TestIDFT(unittest.TestCase):
    def setUp(self):
        n = 100
        sr = 100.0
        t = np.linspace(0, n / sr, n, endpoint=False)
        y = 2.0 * np.sin(2 * np.pi * 10 * t)
        self.signal = SignalSeries(x=t, y=y)
        self.t = t
        self.y = y

    def test_idft_round_trip_all_windows_norms_pads(self):
        windows = (WindowType.RECTANGULAR, WindowType.HAMMING)
        norms = (NormType.CFT, NormType.ASD)
        pads = (1, 2)
        ranges = tuple(DFTRange)

        for window, norm, pad, dft_range in product(windows, norms, pads, ranges):
            with self.subTest(window=window, norm=norm, pad=pad, range=dft_range):
                dft = self.signal.calculate_dft(window=window, norm=norm, pad=pad, dft_range=dft_range)
                recovered = dft.calculate_idft()
                assert_array_almost_equal(np.real(recovered.y), self.y, decimal=10)
                assert_array_almost_equal(recovered.x, self.t, decimal=10)

    def test_idft_asd_abs_raises(self):
        dft = self.signal.calculate_dft(norm=NormType.ASD_ABS)
        with self.assertRaises(ValueError):
            dft.calculate_idft()

    def test_idft_psd_raises(self):
        dft = self.signal.calculate_dft(norm=NormType.PSD)
        with self.assertRaises(ValueError):
            dft.calculate_idft()

    def test_idft_hann_raises(self):
        dft = self.signal.calculate_dft(norm=NormType.ASD, window=WindowType.HANN)
        with self.assertRaises(ValueError):
            dft.calculate_idft()

    def test_idft_remove_padding_false(self):
        dft = self.signal.calculate_dft(pad=2.0)
        # Should return a signal of length 200 (padded length)
        recovered = dft.calculate_idft(remove_padding=False)
        self.assertEqual(len(recovered), 200)
        # The first 100 samples should match the original signal
        assert_array_almost_equal(np.real(recovered.y[:100]), self.y, decimal=10)
        # The remaining 100 samples should be approximately zero
        assert_array_almost_equal(np.real(recovered.y[100:]), np.zeros(100), decimal=10)

    def test_idft_remove_window_false(self):
        dft = self.signal.calculate_dft(window=WindowType.HAMMING)
        recovered = dft.calculate_idft(remove_window=False)
        self.assertEqual(len(recovered), 100)
        # The recovered signal should be the original signal multiplied by the window
        expected_y = self.y * dft.dft_config.window_array
        assert_array_almost_equal(np.real(recovered.y), expected_y, decimal=10)


class TestFourierSeriesProperties(unittest.TestCase):
    def setUp(self):
        self.signal = _make_signal()
        self.fft = self.signal.calculate_dft()

    def test_real_property_is_real(self):
        r = self.fft.real
        self.assertFalse(np.iscomplexobj(r.y))

    def test_abs_property_is_nonnegative(self):
        a = self.fft.abs
        self.assertTrue(np.all(a.y >= 0))

    def test_convert_to_psd_then_asd_round_trip(self):
        asd = self.signal.calculate_dft(norm=NormType.ASD)
        psd = asd.convert_to_psd()
        self.assertEqual(psd.dft_config.norm_type, NormType.PSD)

    def test_convert_non_asd_to_psd_raises(self):
        cft = self.signal.calculate_dft(norm=NormType.CFT)
        with self.assertRaises(ValueError):
            cft.convert_to_psd()

    def test_convert_non_psd_to_asd_raises(self):
        cft = self.signal.calculate_dft(norm=NormType.CFT)
        with self.assertRaises(ValueError):
            cft.convert_to_asd()

    def test_copy_independence(self):
        f2 = self.fft.copy()
        f2.y[0] = 999.0
        self.assertNotEqual(self.fft.y[0], 999.0)

    def test_integral_subrange_smaller_than_full(self):
        full = float(np.real(self.fft.abs.calculate_integral()))
        sub = float(np.real(self.fft.abs.calculate_integral(f_min=0.0, f_max=50.0)))
        self.assertLess(sub, full)


class TestIntegralBoundaryOverlap(unittest.TestCase):
    def test_flat_spectrum_partial_limits_equal_window_width(self):
        n, sr = 100, 100.0
        cfg = DFTConfig(number_of_samples=n, sample_rate=sr, norm_type=NormType.PSD,
                        dft_range=DFTRange.DOUBLE_SIDED)
        freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sr))
        c = 3.7
        psd = FourierSeries(x=freqs, y=np.full(n, c), dft_config=cfg)
        f1, f2 = -23.7, 34.6
        result = float(np.real(psd.calculate_integral(f_min=f1, f_max=f2)))
        self.assertAlmostEqual(result, c * (f2 - f1), places=8)

    def test_single_sided_parseval_with_dc(self):
        n, sr = 100, 100.0
        t = np.linspace(0, n / sr, n, endpoint=False)
        dc, amp, freq = 2.0, 1.0, 10.0
        y = dc + amp * np.cos(2 * np.pi * freq * t)
        signal = SignalSeries(x=t, y=y)
        power = signal.calculate_rms() ** 2  # dc^2 + amp^2/2

        psd = signal.calculate_dft(norm=NormType.PSD, dft_range=DFTRange.SINGLE_SIDED)
        result = float(np.real(psd.calculate_integral()))
        self.assertAlmostEqual(result, power, places=4)

    def test_single_sided_parseval_explicit_limits(self):
        n, sr = 100, 100.0
        t = np.linspace(0, n / sr, n, endpoint=False)
        dc, amp, freq = 2.0, 1.0, 10.0
        y = dc + amp * np.cos(2 * np.pi * freq * t)
        signal = SignalSeries(x=t, y=y)
        power = signal.calculate_rms() ** 2

        psd = signal.calculate_dft(norm=NormType.PSD, dft_range=DFTRange.SINGLE_SIDED)
        result = float(np.real(psd.calculate_integral(f_min=0.0, f_max=sr / 2)))
        self.assertAlmostEqual(result, power, places=4)


if __name__ == "__main__":
    unittest.main()
