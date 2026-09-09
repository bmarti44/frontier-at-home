"""Native FlashInfer prebuilt-cache provider for this pinned Spark runtime."""
from pathlib import Path

__version__ = '0.6.18rc10'


def get_jit_cache_dir():
    return str(Path(__file__).resolve().parents[1] / 'state/.cache/flashinfer' /
               __version__ / '121a/cached_ops')
