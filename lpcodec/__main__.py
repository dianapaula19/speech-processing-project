"""Encode and decode every sentence, and build the DCR listening-test files.

    python -m lpcodec                      # all 20 sentences -> output/
    python -m lpcodec --pulse-bits 4       # try another bit allocation
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .codec import CodecConfig, decode, encode, segmental_snr_db, snr_db
from .io import read_wav, write_wav

INPUT_DIRS = ("Female_sentences_8kHz", "Male_sentences_8kHz")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="folder holding the sentence folders")
    parser.add_argument("--out", default="output")
    parser.add_argument("--lattice-bits", type=int, default=CodecConfig.lattice_bits)
    parser.add_argument("--gain-bits", type=int, default=CodecConfig.gain_bits)
    parser.add_argument("--pulse-bits", type=int, default=CodecConfig.pulse_bits)
    args = parser.parse_args(argv)

    config = CodecConfig(lattice_bits=args.lattice_bits, gain_bits=args.gain_bits,
                         pulse_bits=args.pulse_bits)
    out = Path(args.out)
    (out / "coded").mkdir(parents=True, exist_ok=True)
    (out / "dcr").mkdir(parents=True, exist_ok=True)

    print(f"Bit budget per {config.frame_ms:g} ms frame: {config.budget()} "
          f"= {config.bits_per_frame} bits = {config.bit_rate / 1000:.2f} kbit/s")

    results = []
    for folder in INPUT_DIRS:
        for path in sorted((Path(args.root) / folder).glob("*.wav")):
            signal, rate = read_wav(path)
            assert rate == config.sample_rate, f"{path}: expected {config.sample_rate} Hz"
            bitstream, n = encode(signal, config)
            coded = decode(bitstream, n, config)
            write_wav(out / "coded" / path.name, coded, rate)

            # DCR test sample: reference, 0.5 s of silence, coded version
            gap = np.zeros(rate // 2)
            write_wav(out / "dcr" / path.name, np.r_[signal, gap, coded], rate)

            results.append({
                "file": f"{folder}/{path.name}",
                "seconds": round(n / rate, 2),
                "bytes": len(bitstream),
                "kbit_per_s": round(len(bitstream) * 8 / (n / rate) / 1000, 2),
                "snr_db": round(snr_db(signal, coded), 2),
                "segsnr_db": round(segmental_snr_db(signal, coded), 2),
            })
            print(f"{results[-1]['file']:40s} SNR {results[-1]['snr_db']:6.2f} dB  "
                  f"segSNR {results[-1]['segsnr_db']:6.2f} dB")

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Mean segmental SNR: {np.mean([r['segsnr_db'] for r in results]):.2f} dB")


if __name__ == "__main__":
    main()
