"""Linear prediction: autocorrelation method, lattice conversion, spectra.

Convention: A(z) = 1 + a1 z^-1 + ... + ap z^-p, stored as ``[1, a1, ..., ap]``.
The analysis filter A(z) is FIR; the synthesis filter 1/A(z) is IIR.
"""

import numpy as np


def autocorrelation(frame, order):
    """First ``order + 1`` autocorrelation lags of a Hamming-windowed frame."""
    windowed = frame * np.hamming(len(frame))
    full = np.correlate(windowed, windowed, mode="full")
    mid = len(windowed) - 1
    return full[mid:mid + order + 1]


def levinson(r, order):
    """Levinson-Durbin recursion.

    Returns ``(a, k)``: the predictor polynomial and the reflection
    (lattice) coefficients. For a positive definite autocorrelation all
    ``|k| < 1``, so 1/A(z) is stable. A silent frame (r[0] == 0) gives the
    trivial filter A(z) = 1 instead of a division by zero.
    """
    a = np.zeros(order + 1)
    a[0] = 1.0
    k = np.zeros(order)
    error = r[0]
    if error <= 0:
        return a, k
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1])
        k[i - 1] = -acc / error
        a[1:i + 1] = a[1:i + 1] + k[i - 1] * np.r_[a[i - 1:0:-1], 1.0]
        error *= 1.0 - k[i - 1] ** 2
        if error <= 0:  # numerically singular: stop here
            break
    return a, k


def lpc(frame, order):
    """LP coefficients of one frame (autocorrelation method, Hamming window)."""
    return levinson(autocorrelation(frame, order), order)[0]


def lpc_to_lattice(a):
    """Step-down recursion: direct form ``[1, a1..ap]`` -> reflection coefficients.

    Works on a copy, so the caller's coefficients are left untouched.
    """
    a = np.asarray(a, dtype=float).copy()
    order = len(a) - 1
    k = np.zeros(order)
    for i in range(order, 0, -1):
        k[i - 1] = a[i]
        if abs(k[i - 1]) >= 1:
            raise ValueError("filter is unstable (|k| >= 1)")
        prev = (a[1:i] - k[i - 1] * a[i - 1:0:-1]) / (1.0 - k[i - 1] ** 2)
        a[1:i] = prev
        a[i] = 0.0
    return k


def lattice_to_lpc(k):
    """Step-up recursion: reflection coefficients -> direct form ``[1, a1..ap]``."""
    order = len(k)
    a = np.zeros(order + 1)
    a[0] = 1.0
    for i in range(1, order + 1):
        a[1:i + 1] = a[1:i + 1] + k[i - 1] * np.r_[a[i - 1:0:-1], 1.0]
    return a


def synthesis_power_spectrum(a, n_fft=1024):
    """Power spectrum of 1/A(z) at bins 0..N/2."""
    response = np.fft.rfft(a, n_fft)
    return 1.0 / np.maximum(np.abs(response) ** 2, 1e-12)


def spectral_distortion(a_original, a_quantized, n_fft=1024):
    """Spectral distortion in dB (Paliwal and Atal, 1993), over bins 0..N/2."""
    p_o = synthesis_power_spectrum(a_original, n_fft)
    p_q = synthesis_power_spectrum(a_quantized, n_fft)
    diff = 10 * np.log10(p_o) - 10 * np.log10(p_q)
    return float(np.sqrt(np.mean(diff ** 2)))
