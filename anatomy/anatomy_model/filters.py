"""Input filters: keep only images worth learning from.

Two filters, both ported from Tumblr-Image-Collector's existing code
(the learning pipeline reuses them as an *input* stage instead of a
saving stage):

- Quality: ``evaluate_image_quality`` in the collector judged an
  image by resolution, Shannon entropy and Laplacian-variance
  sharpness, with thresholds (800x600 minimum, entropy >= 4.0,
  sharpness > 100).  The same quantities and thresholds are used
  here, computed on the in-memory image instead of a file on disk;
  the Laplacian is evaluated with numpy (identical 4-neighbour
  kernel), so OpenCV is not needed for this stage.
- Duplicates: the collector compared perceptual hashes so reposted
  media could not enter twice (its ``IMAGE_HASH_THRESHOLD`` was a
  Hamming distance of 5 over a 64-bit pHash).  Duplicates here would
  bias the statistics, so the same rule is applied.  Hashes live in
  memory for the duration of one run only and are never persisted --
  a hash set written to disk would become a fingerprint registry of
  processed images, which this design forbids.
"""

from __future__ import annotations

import io
import math
from typing import Optional, Set, Tuple

import numpy as np
from PIL import Image

MIN_RESOLUTION = (800, 600)
MIN_ENTROPY = 4.0
MIN_SHARPNESS = 100.0
HASH_DISTANCE_THRESHOLD = 5  # collector's IMAGE_HASH_THRESHOLD


def quality_metrics(image_bytes: bytes) -> Optional[dict]:
    """Quality metrics for in-memory image bytes, or None if unreadable."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            gray = np.asarray(img.convert("L"), dtype=np.float64)
    except Exception:
        return None
    if gray.size == 0:
        return None
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist[hist > 0] / gray.size
    entropy = float(-np.sum(hist * np.log2(hist)))
    # 4-neighbour Laplacian variance (same statistic cv2.Laplacian
    # feeds in the collector's version).
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    sharpness = float(lap.var()) if lap.size else 0.0
    return {
        "width": width,
        "height": height,
        "entropy": entropy,
        "sharpness": sharpness,
        "ok": (
            width >= MIN_RESOLUTION[0]
            and height >= MIN_RESOLUTION[1]
            and entropy >= MIN_ENTROPY
            and sharpness > MIN_SHARPNESS
        ),
    }


def perceptual_hash(image_bytes: bytes) -> Optional[int]:
    """64-bit DCT perceptual hash (the classic pHash construction)."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            small = np.asarray(
                img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64
            )
    except Exception:
        return None
    # 2D DCT-II via separable cosine transform matrices.
    def dct_matrix(n: int) -> np.ndarray:
        k = np.arange(n)[:, None]
        m = np.arange(n)[None, :]
        mat = np.cos((2 * m + 1) * k * math.pi / (2 * n))
        mat[0, :] *= 1.0 / math.sqrt(2)
        return mat * math.sqrt(2.0 / n)

    d = dct_matrix(32)
    coeffs = d @ small @ d.T
    low = coeffs[:8, :8]
    median = np.median(low[1:, :])  # exclude the DC term from the median
    bits = (low > median).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class DuplicateFilter:
    """In-memory near-duplicate rejection (per run, never persisted)."""

    def __init__(self, threshold: int = HASH_DISTANCE_THRESHOLD):
        self._hashes: Set[int] = set()
        self._threshold = threshold

    def is_duplicate(self, image_hash: Optional[int]) -> bool:
        if image_hash is None:
            return False
        for existing in self._hashes:
            if hamming(existing, image_hash) <= self._threshold:
                return True
        self._hashes.add(image_hash)
        return False

    def __len__(self) -> int:
        return len(self._hashes)
