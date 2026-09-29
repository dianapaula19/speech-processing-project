from pathlib import Path

import numpy as np
import pytest

from lpcodec import CodecConfig, decode, encode, read_wav, segmental_snr_db
from lpcodec.filtering import lp_analysis, lp_synthesis
from lpcodec.lp import (autocorrelation, lattice_to_lpc, levinson, lpc_to_lattice,
                        spectral_distortion)
from lpcodec.quantize import quantize_lattice

ROOT = Path(__file__).resolve().parents[1]
MALE = ROOT / "Male_sentences_8kHz" / "si745_8kHz.wav"


def _ar_signal(n=4000, seed=0):
    """A stable AR(2) process: a toy stand-in for voiced speech."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.standard_normal(n) * 0.05
    for i in range(2, n):
        x[i] = 1.6 * x[i - 1] - 0.8 * x[i - 2] + e[i]
    return x / np.max(np.abs(x))


def test_levinson_matches_direct_solve():
    frame = _ar_signal(200)
    r = autocorrelation(frame, 10)
    a, k = levinson(r, 10)
    from scipy.linalg import solve_toeplitz
    direct = np.r_[1.0, solve_toeplitz(r[:-1], -r[1:])]
    np.testing.assert_allclose(a, direct, atol=1e-8)
    assert np.all(np.abs(k) < 1)


def test_lattice_round_trip_and_input_untouched():
    a, k = levinson(autocorrelation(_ar_signal(200, seed=3), 10), 10)
    original = a.copy()
    np.testing.assert_allclose(lpc_to_lattice(a), k, atol=1e-10)
    np.testing.assert_array_equal(a, original)  # the old version modified its input
    np.testing.assert_allclose(lattice_to_lpc(k), a, atol=1e-10)


def test_quantized_filters_are_stable():
    a, k = levinson(autocorrelation(_ar_signal(200, seed=1), 10), 10)
    for bits in (5, 4, 3):
        kq = quantize_lattice(k, bits)
        assert np.all(np.abs(kq) < 1)
        assert np.all(np.abs(np.roots(lattice_to_lpc(kq))) < 1)


def test_average_sd_grows_with_fewer_bits():
    # The old notebook printed the same average SD for 5, 4 and 3 bits.
    if not MALE.exists():
        pytest.skip("sentence files not present")
    signal, _ = read_wav(MALE)
    averages = []
    for bits in (5, 4, 3):
        sds = []
        for start in range(0, len(signal) - 200, 200):
            a, k = levinson(autocorrelation(signal[start:start + 200], 10), 10)
            sds.append(spectral_distortion(a, lattice_to_lpc(quantize_lattice(k, bits))))
        averages.append(np.mean(sds))
    assert averages[0] < averages[1] < averages[2]


def test_unquantized_analysis_synthesis_is_lossless():
    x = _ar_signal()
    residual, coeffs = lp_analysis(x, 200, 10)
    np.testing.assert_allclose(lp_synthesis(residual, 200, coeffs), x, atol=1e-9)


def test_bit_budget():
    config = CodecConfig()
    assert config.frame_length == 200
    assert config.n_pulses == 66  # Figure 1 of the brief
    assert config.bits_per_frame == 30 + 6 + 2 + 198
    assert config.bit_rate == 9440


def test_codec_round_trip_on_real_speech():
    if not MALE.exists():
        pytest.skip("sentence files not present")
    signal, rate = read_wav(MALE)
    bitstream, n = encode(signal)
    coded = decode(bitstream, n)
    assert len(coded) == len(signal)
    measured = len(bitstream) * 8 / (n / rate)
    assert abs(measured - CodecConfig().bit_rate) < 100  # only byte padding differs
    assert np.all(np.isfinite(coded))
    assert segmental_snr_db(signal, coded) > 0
