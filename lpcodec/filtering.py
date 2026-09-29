"""Frame-wise LP analysis and synthesis with filter memory carried across frames."""

import numpy as np
from scipy.signal import lfilter, lfiltic

from .lp import lpc


def frames(signal, frame_length):
    """(start, end) of consecutive non-overlapping frames; the last may be short."""
    return [(s, min(s + frame_length, len(signal))) for s in range(0, len(signal), frame_length)]


def analysis_frame(frame, a, past_input):
    """Residual of one frame through A(z).

    A(z) is FIR, so its memory is simply the last p input samples: prepend
    them and drop the first p outputs.
    """
    p = len(a) - 1
    return lfilter(a, [1.0], np.r_[past_input[-p:], frame])[p:]


def synthesis_frame(residual, a, past_output):
    """Speech of one frame through 1/A(z).

    1/A(z) is IIR, so its memory is the last p *output* samples. They are
    turned into the initial state of the current frame's filter with
    ``lfiltic``; feeding them in as extra input would be wrong.
    """
    p = len(a) - 1
    zi = lfiltic([1.0], a, y=past_output[::-1][:p])
    out, _ = lfilter([1.0], a, residual, zi=zi)
    return out


def lp_analysis(signal, frame_length, order, coefficients=None):
    """Residual of a whole signal, frame by frame.

    ``coefficients`` optionally gives the filter to use for each frame
    (e.g. quantized ones); by default each frame's LPC is computed.
    Returns ``(residual, coefficients)``.
    """
    residual = np.zeros_like(signal, dtype=float)
    past = np.zeros(order)
    used = []
    for i, (start, end) in enumerate(frames(signal, frame_length)):
        frame = signal[start:end]
        a = coefficients[i] if coefficients is not None else lpc(frame, order)
        residual[start:end] = analysis_frame(frame, a, past)
        past = np.r_[past, frame][-order:]
        used.append(a)
    return residual, used


def lp_synthesis(residual, frame_length, coefficients):
    """Inverse of :func:`lp_analysis` given the same per-frame coefficients."""
    order = len(coefficients[0]) - 1
    output = np.zeros_like(residual, dtype=float)
    past = np.zeros(order)
    for (start, end), a in zip(frames(residual, frame_length), coefficients):
        output[start:end] = synthesis_frame(residual[start:end], a, past)
        past = np.r_[past, output[start:end]][-order:]
    return output
