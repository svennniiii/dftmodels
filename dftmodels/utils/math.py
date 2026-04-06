import numpy as np
from scipy.special import wofz


def sinc(x, amplitude, center, width):
    return amplitude * np.sinc((x - center) / width)


def lorentzian(x, amplitude, center, width):
    return amplitude / (np.pi * width * (1 + ((x - center) / width) ** 2))


def gaussian(x, amplitude, center, width):
    return (
        amplitude
        * np.exp(-0.5 * ((x - center) / width) ** 2)
        / np.sqrt(2 * np.pi * width ** 2)
    )


def voigt(x, amplitude, center, width_lorentzian, width_gaussian):
    return (
        amplitude
        * np.real(wofz((x - center + 1j * width_lorentzian) / (width_gaussian * np.sqrt(2))))
        / (width_gaussian * np.sqrt(2 * np.pi))
    )


def linear(x, slope, intercept):
    return slope * x + intercept


def linear_complex(x, real_intercept, imag_intercept, real_slope, imag_slope):
    return (real_intercept + 1j * imag_intercept) + (real_slope + 1j * imag_slope) * x


def exponential(x, amplitude, decay):
    return amplitude * np.exp(decay * x)
