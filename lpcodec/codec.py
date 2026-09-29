"""LP codec with a regular-pulse (decimated) residual, section 3.4 of the brief.

Per 25 ms frame (200 samples at 8 kHz) the encoder sends:

====================  =====================================  ============
field                 how                                    default bits
====================  =====================================  ============
LP filter             10 reflection coeffs, 3 bits each       30
gain                  max |low-passed residual|, log scale    6
grid                  which of the 4 decimated sequences      2
pulses                66 samples (every 3rd), uniform         66 x 3 = 198
====================  =====================================  ============

236 bits / 25 ms = 9.44 kbit/s. The decoder rebuilds the residual (pulses x
gain, zeros in between), rebuilds A(z) from the lattice indices and filters the
residual through 1/A(z). Everything is really packed into a bit string, so the
bit rate is measured, not estimated.
"""

from dataclasses import dataclass

import numpy as np

from .filtering import analysis_frame, frames, synthesis_frame
from .lp import lattice_to_lpc, levinson, autocorrelation
from .quantize import gain_index, gain_value, uniform_index, uniform_value

# Anti-aliasing low-pass FIR from Table 1 of the brief (zero phase, n = -5..5).
LOWPASS = np.array([-0.016357422, -0.045654297, 0.0, 0.25073242, 0.70080566, 1.0,
                    0.70080566, 0.25073242, 0.0, -0.045654297, -0.016357422])


@dataclass(frozen=True)
class CodecConfig:
    sample_rate: int = 8000
    frame_ms: float = 25.0
    order: int = 10
    lattice_bits: int = 3
    gain_bits: int = 6
    pulse_bits: int = 3
    decimation: int = 3
    n_grids: int = 4

    @property
    def frame_length(self):
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def n_pulses(self):
        """Pulses per grid; the same for every grid, see Figure 1 of the brief."""
        return (self.frame_length - (self.n_grids - 1) - 1) // self.decimation + 1

    @property
    def grid_bits(self):
        return int(np.ceil(np.log2(self.n_grids)))

    @property
    def bits_per_frame(self):
        return (self.order * self.lattice_bits + self.gain_bits + self.grid_bits
                + self.n_pulses * self.pulse_bits)

    @property
    def bit_rate(self):
        return self.bits_per_frame * 1000 / self.frame_ms

    def budget(self):
        return {
            "LP filter": self.order * self.lattice_bits,
            "gain": self.gain_bits,
            "grid": self.grid_bits,
            "pulses": self.n_pulses * self.pulse_bits,
        }


class BitWriter:
    def __init__(self):
        self.bits = []

    def write(self, value, n):
        self.bits.extend((int(value) >> i) & 1 for i in reversed(range(n)))

    def to_bytes(self):
        padded = self.bits + [0] * (-len(self.bits) % 8)
        return np.packbits(np.array(padded, dtype=np.uint8)).tobytes()


class BitReader:
    def __init__(self, data):
        self.bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        self.pos = 0

    def read(self, n):
        value = 0
        for bit in self.bits[self.pos:self.pos + n]:
            value = (value << 1) | int(bit)
        self.pos += n
        return value


def lowpass(residual):
    """Zero-phase anti-aliasing filter (symmetric FIR, centred)."""
    return np.convolve(residual, LOWPASS, mode="same")


def grid_positions(grid, config):
    return grid + config.decimation * np.arange(config.n_pulses)


def encode(signal, config=CodecConfig()):
    """Encode a signal in [-1, 1]. Returns ``(bitstream, n_samples)``."""
    writer = BitWriter()
    past_input = np.zeros(config.order)
    for start, end in frames(signal, config.frame_length):
        frame = np.zeros(config.frame_length)
        frame[:end - start] = signal[start:end]

        # LP filter: quantize the lattice form, analyse with the quantized filter
        _, k = levinson(autocorrelation(frame, config.order), config.order)
        k_idx = uniform_index(k, config.lattice_bits)
        a_q = lattice_to_lpc(uniform_value(k_idx, config.lattice_bits))
        residual = lowpass(analysis_frame(frame, a_q, past_input))
        past_input = frame[-config.order:]

        # Gain: normalise by the quantized maximum
        g_idx = gain_index(np.max(np.abs(residual)), config.gain_bits)
        normalised = np.clip(residual / gain_value(g_idx, config.gain_bits), -1, 1)

        # Grid: the decimated sequence with the most energy
        energies = [np.sum(normalised[grid_positions(g, config)] ** 2) for g in range(config.n_grids)]
        grid = int(np.argmax(energies))
        p_idx = uniform_index(normalised[grid_positions(grid, config)], config.pulse_bits)

        for idx in k_idx:
            writer.write(idx, config.lattice_bits)
        writer.write(g_idx, config.gain_bits)
        writer.write(grid, config.grid_bits)
        for idx in p_idx:
            writer.write(idx, config.pulse_bits)
    return writer.to_bytes(), len(signal)


def decode(bitstream, n_samples, config=CodecConfig()):
    reader = BitReader(bitstream)
    n_frames = -(-n_samples // config.frame_length)
    output = np.zeros(n_frames * config.frame_length)
    past_output = np.zeros(config.order)
    for f in range(n_frames):
        k_idx = [reader.read(config.lattice_bits) for _ in range(config.order)]
        gain = gain_value(reader.read(config.gain_bits), config.gain_bits)
        grid = reader.read(config.grid_bits)
        pulses = uniform_value([reader.read(config.pulse_bits) for _ in range(config.n_pulses)],
                               config.pulse_bits)

        residual = np.zeros(config.frame_length)
        residual[grid_positions(grid, config)] = pulses * gain
        a_q = lattice_to_lpc(uniform_value(np.array(k_idx), config.lattice_bits))
        start = f * config.frame_length
        out = synthesis_frame(residual, a_q, past_output)
        output[start:start + config.frame_length] = out
        past_output = out[-config.order:]
    return output[:n_samples]


def snr_db(reference, test):
    noise = np.sum((reference - test) ** 2)
    return float(10 * np.log10(np.sum(reference ** 2) / max(noise, 1e-20)))


def segmental_snr_db(reference, test, frame_length=200, floor=-10.0, ceil=35.0):
    """Mean per-frame SNR, clamped to [floor, ceil] dB; silent frames are skipped."""
    values = []
    for start, end in frames(reference, frame_length):
        ref = reference[start:end]
        if np.sum(ref ** 2) < 1e-8:
            continue
        values.append(np.clip(snr_db(ref, test[start:end]), floor, ceil))
    return float(np.mean(values)) if values else float("nan")
