"""LP-based speech codec (ELEC-E5522 Speech Processing Project, Aalto 2025)."""

from .codec import CodecConfig, decode, encode, segmental_snr_db, snr_db
from .io import read_wav, write_wav

__all__ = ["CodecConfig", "encode", "decode", "snr_db", "segmental_snr_db", "read_wav", "write_wav"]
