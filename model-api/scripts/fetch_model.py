"""Download the trained model and print the scikit-learn version it was saved with.

Used at Docker build time:

    python scripts/fetch_model.py <MODEL_URL> fraud_model.pkl > sklearn_version.txt

A pickled scikit-learn estimator must be loaded with the same scikit-learn version
that saved it, so the image installs exactly that version. Each estimator stores it
in its `_sklearn_version` attribute, readable in the raw pickle bytes.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import re
import sys
import urllib.request
import zlib

FALLBACK_VERSION = "1.7.2"
VERSION_RE = re.compile(rb"_sklearn_version.{0,24}?(\d+\.\d+(?:\.\d+)?(?:rc\d+|\.post\d+)?)", re.DOTALL)


def _decompress(raw: bytes) -> bytes:
    """Undo joblib compression (zlib, gzip, xz, bz2) when present."""
    for magic, fn in (
        (b"\x1f\x8b", gzip.decompress),
        (b"\xfd7zXZ", lzma.decompress),
        (b"BZh", bz2.decompress),
        (b"\x78", zlib.decompress),
    ):
        if raw.startswith(magic):
            try:
                return fn(raw)
            except Exception:  # noqa: BLE001 - not actually compressed
                return raw
    return raw


def detect_sklearn_version(raw: bytes) -> str | None:
    # Pickle memoizes repeated strings, so only the first estimator's version is
    # written out in full; all estimators of one file share it anyway.
    match = VERSION_RE.search(_decompress(raw))
    return match.group(1).decode() if match else None


def main(url: str, dest: str) -> None:
    print(f"Downloading model from {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as resp:
        raw = resp.read()
    with open(dest, "wb") as f:
        f.write(raw)
    version = detect_sklearn_version(raw)
    if version is None:
        print(f"No _sklearn_version found, falling back to {FALLBACK_VERSION}", file=sys.stderr)
        version = FALLBACK_VERSION
    print(f"Model: {len(raw) / 1e6:.1f} MB, saved with scikit-learn {version}", file=sys.stderr)
    print(version)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
