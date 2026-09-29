"""Scalar quantizers. Each has an index form (for the bitstream) and a value form."""

import numpy as np


def uniform_index(x, bits, low=-1.0, high=1.0):
    """Mid-rise uniform quantizer: index 0..2^bits-1 of the cell holding ``x``."""
    levels = 2 ** bits
    step = (high - low) / levels
    return np.clip(np.floor((np.asarray(x) - low) / step), 0, levels - 1).astype(int)


def uniform_value(index, bits, low=-1.0, high=1.0):
    """Centre of cell ``index``. Never reaches ``low``/``high`` themselves."""
    step = (high - low) / 2 ** bits
    return low + (np.asarray(index) + 0.5) * step


def quantize_uniform(x, bits, low=-1.0, high=1.0):
    return uniform_value(uniform_index(x, bits, low, high), bits, low, high)


def quantize_lattice(k, bits):
    """Quantize reflection coefficients uniformly on (-1, 1).

    Cell centres are at most ``1 - step/2`` in magnitude, so every quantized
    coefficient stays inside the unit interval and the quantized synthesis
    filter is stable by construction.
    """
    return quantize_uniform(k, bits, -1.0, 1.0)


# Frame gains (the maximum |residual| of a frame) span several decades, so they
# are quantized uniformly in log10 between these limits (signals in [-1, 1]).
GAIN_LOG_MIN, GAIN_LOG_MAX = -5.0, 1.0


def gain_index(gain, bits):
    log_gain = np.log10(max(gain, 10 ** GAIN_LOG_MIN))
    return int(uniform_index(log_gain, bits, GAIN_LOG_MIN, GAIN_LOG_MAX))


def gain_value(index, bits):
    return float(10 ** uniform_value(index, bits, GAIN_LOG_MIN, GAIN_LOG_MAX))
