import unittest
import numpy as np
import lmfit
from dftmodels.stats import cramer_rao_bound
from dftmodels.models.sinusoid import Sinusoid


class TestCramerRaoBoundSinusoid(unittest.TestCase):
    """
    Verify cramer_rao_bound against analytic Rife–Boorstyn bounds for a
    stationary sinusoid x_n = a_i cos(2π f t_n) + a_q sin(2π f t_n).

    Analytic results (N samples at rate fs, T = (N-1)/fs, A = sqrt(a_i²+a_q²)):
        σ²(a_i) = σ²(a_q) = 2σ²/N   (frequency known/fixed)
        σ²(f)   = 6σ²  / (π² A² T² N)
    """

    def setUp(self):
        fs = 200.0
        N  = 500
        self.t  = np.arange(N) / fs
        self.f0 = 17.3
        self.ai = 1.5
        self.aq = 0.8
        self.A  = np.sqrt(self.ai**2 + self.aq**2)
        self.σ  = 0.05
        self.N  = N
        self.T  = (N - 1) / fs

    def _params(self, **vary_overrides):
        p = lmfit.Parameters()
        p.add("amplitude_i", value=self.ai)
        p.add("amplitude_q", value=self.aq)
        p.add("frequency",   value=self.f0)
        for name, vary in vary_overrides.items():
            p[name].vary = vary
        return p

    def test_amplitude_variance_frequency_fixed(self):
        """With frequency known (fixed), σ²(a_i) = σ²(a_q) = 2σ²/N."""
        crb = cramer_rao_bound(Sinusoid.eval, self._params(frequency=False), self.t, self.σ)
        expected = 2 * self.σ**2 / self.N
        self.assertAlmostEqual(crb["amplitude_i"], expected, delta=expected * 0.02)
        self.assertAlmostEqual(crb["amplitude_q"], expected, delta=expected * 0.02)

    def test_frequency_variance_rife_boorstyn(self):
        """Frequency marginal variance matches Rife–Boorstyn with all params free."""
        crb = cramer_rao_bound(Sinusoid.eval, self._params(), self.t, self.σ)
        expected = 6 * self.σ**2 / (np.pi**2 * self.A**2 * self.T**2 * self.N)
        self.assertAlmostEqual(crb["frequency"], expected, delta=expected * 0.02)

    def test_return_cov_shape(self):
        crb, Cov = cramer_rao_bound(Sinusoid.eval, self._params(), self.t, self.σ, return_cov=True)
        self.assertEqual(Cov.shape, (3, 3))
        for k, name in enumerate(["amplitude_i", "amplitude_q", "frequency"]):
            self.assertAlmostEqual(float(Cov[k, k]), crb[name], places=15)

    def test_return_cov_false_returns_dict(self):
        result = cramer_rao_bound(Sinusoid.eval, self._params(), self.t, self.σ)
        self.assertIsInstance(result, dict)
        self.assertIn("frequency", result)

    def test_fixed_param_excluded(self):
        """vary=False parameters must not appear in the CRB dict."""
        crb = cramer_rao_bound(Sinusoid.eval, self._params(amplitude_q=False), self.t, self.σ)
        self.assertIn("amplitude_i", crb)
        self.assertNotIn("amplitude_q", crb)
        self.assertIn("frequency", crb)
        self.assertEqual(len(crb), 2)


class TestSineFourierCRB(unittest.TestCase):
    """Verify Sinusoid.crb() against manual delta-method computation."""

    def setUp(self):
        fs = 200.0
        N  = 500
        self.t  = np.arange(N) / fs
        self.f0 = 17.3
        self.ai = 1.5
        self.aq = 0.8
        self.σ  = 0.05

    def _params(self):
        p = lmfit.Parameters()
        p.add("amplitude_i", value=self.ai)
        p.add("amplitude_q", value=self.aq)
        p.add("frequency",   value=self.f0)
        return p

    def test_crb_matches_manual_delta_method(self):
        params = self._params()
        result = Sinusoid.crb(params, self.t, self.σ)

        _, Cov = cramer_rao_bound(Sinusoid.eval, params, self.t, self.σ, return_cov=True)
        ai, aq = self.ai, self.aq
        A = np.sqrt(ai**2 + aq**2)

        expected_amp = float(np.sqrt(
            (ai/A)**2 * Cov[0, 0] + (aq/A)**2 * Cov[1, 1] + 2*(ai/A)*(aq/A)*Cov[0, 1]
        ))
        expected_phase = float(np.sqrt(
            (aq/A**2)**2 * Cov[0, 0] + (ai/A**2)**2 * Cov[1, 1]
            + 2*(aq/A**2)*(ai/A**2)*Cov[0, 1]
        ))
        self.assertAlmostEqual(result["amplitude"], expected_amp,   places=12)
        self.assertAlmostEqual(result["phase"],     expected_phase, places=12)

    def test_crb_direct_params_match_diagonal(self):
        params = self._params()
        result = Sinusoid.crb(params, self.t, self.σ)

        _, Cov = cramer_rao_bound(Sinusoid.eval, params, self.t, self.σ, return_cov=True)
        self.assertAlmostEqual(result["amplitude_i"], float(np.sqrt(Cov[0, 0])), places=12)
        self.assertAlmostEqual(result["amplitude_q"], float(np.sqrt(Cov[1, 1])), places=12)
        self.assertAlmostEqual(result["frequency"],   float(np.sqrt(Cov[2, 2])), places=12)

    def test_crb_frequency_matches_standalone(self):
        params = self._params()
        crb_dict = cramer_rao_bound(Sinusoid.eval, params, self.t, self.σ)
        result = Sinusoid.crb(params, self.t, self.σ)
        self.assertAlmostEqual(result["frequency"], float(np.sqrt(crb_dict["frequency"])), places=12)


class TestSinusoidWithDecayCRB(unittest.TestCase):
    """Verify Sinusoid.crb() with decay against manual delta-method computation."""

    def setUp(self):
        fs = 200.0
        N  = 500
        self.t     = np.arange(N) / fs
        self.f0    = 17.3
        self.decay = 2.0
        self.ai    = 1.5
        self.aq    = 0.8
        self.σ     = 0.05

    def _params(self):
        p = lmfit.Parameters()
        p.add("amplitude_i", value=self.ai)
        p.add("amplitude_q", value=self.aq)
        p.add("frequency",   value=self.f0)
        p.add("decay",       value=self.decay)
        return p

    def test_crb_matches_manual_delta_method(self):
        params = self._params()
        result = Sinusoid.crb(params, self.t, self.σ)

        _, Cov = cramer_rao_bound(Sinusoid.eval, params, self.t, self.σ, return_cov=True)
        ai, aq = self.ai, self.aq
        A = np.sqrt(ai**2 + aq**2)

        expected_amp = float(np.sqrt(
            (ai/A)**2 * Cov[0, 0] + (aq/A)**2 * Cov[1, 1] + 2*(ai/A)*(aq/A)*Cov[0, 1]
        ))
        expected_phase = float(np.sqrt(
            (aq/A**2)**2 * Cov[0, 0] + (ai/A**2)**2 * Cov[1, 1]
            + 2*(aq/A**2)*(ai/A**2)*Cov[0, 1]
        ))
        self.assertAlmostEqual(result["amplitude"], expected_amp,   places=12)
        self.assertAlmostEqual(result["phase"],     expected_phase, places=12)

    def test_crb_direct_params_match_diagonal(self):
        params = self._params()
        result = Sinusoid.crb(params, self.t, self.σ)

        _, Cov = cramer_rao_bound(Sinusoid.eval, params, self.t, self.σ, return_cov=True)
        self.assertAlmostEqual(result["amplitude_i"], float(np.sqrt(Cov[0, 0])), places=12)
        self.assertAlmostEqual(result["amplitude_q"], float(np.sqrt(Cov[1, 1])), places=12)
        self.assertAlmostEqual(result["frequency"],   float(np.sqrt(Cov[2, 2])), places=12)
        self.assertAlmostEqual(result["decay"],       float(np.sqrt(Cov[3, 3])), places=12)

    def test_fwhm_crb_equals_decay_over_pi(self):
        params = self._params()
        result = Sinusoid.crb(params, self.t, self.σ)
        self.assertAlmostEqual(result["fwhm"], result["decay"] / np.pi, places=15)


if __name__ == "__main__":
    unittest.main()
