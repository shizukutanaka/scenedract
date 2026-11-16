"""Streaming aggregation: statistics that grow without storing data.

Each measured quantity gets a :class:`RunningStat`: Welford's online
algorithm for mean/variance plus a fixed-bin histogram for
percentiles.  Individual samples are folded in and forgotten -- the
aggregator's state is O(bins), independent of how many images were
processed, and contains nothing from which an individual image or
pose could be reconstructed.

A statistic is only *published* into the knowledge model once it has
at least ``min_samples`` samples (design section 2.2): below that, a
statistic could mirror an individual's body, and would be too weak
to validate against anyway.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

# (lo, hi, bin count) per statistic family.  Fixed bins keep the
# histogram -- and therefore every percentile -- deterministic.
_RATIO_BINS = (0.0, 0.8, 160)        # segment ratios
_ANGLE_BINS = (0.0, 180.0, 180)      # joint angles, degrees
_MARGIN_BINS = (-1.0, 2.0, 300)      # COM margin ratio
_FACE_BINS = (0.0, 1.0, 200)         # face ratios

BIN_SPECS: Dict[Tuple[str, str], Tuple[float, float, int]] = {}
for _seg in ("head", "upperArm", "forearm", "thigh", "shin", "torso"):
    BIN_SPECS[("proportions", _seg)] = _RATIO_BINS
for _joint in ("elbow", "knee", "shoulder", "hip"):
    BIN_SPECS[("jointAngles", _joint)] = _ANGLE_BINS
BIN_SPECS[("grounding", "comMarginRatio")] = _MARGIN_BINS
BIN_SPECS[("faceRelations", "eyeSpacingRatio")] = _FACE_BINS

ROUND_DIGITS = 6


def _r(x: float) -> float:
    return round(float(x), ROUND_DIGITS)


class RunningStat:
    __slots__ = ("n", "mean", "m2", "lo", "hi", "bins", "counts", "overflow", "underflow")

    def __init__(self, lo: float, hi: float, bins: int):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.lo, self.hi, self.bins = lo, hi, bins
        self.counts = [0] * bins
        self.overflow = 0
        self.underflow = 0

    def add(self, value: float) -> None:
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            return
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        width = (self.hi - self.lo) / self.bins
        if value < self.lo:
            self.underflow += 1
            idx = 0
        elif value >= self.hi:
            self.overflow += 1
            idx = self.bins - 1
        else:
            idx = int((value - self.lo) / width)
        self.counts[idx] += 1

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(max(self.m2, 0.0) / (self.n - 1))

    def percentile(self, p: float) -> float:
        """Percentile from the histogram (linear within the bin)."""
        if self.n == 0:
            return 0.0
        target = p * self.n
        width = (self.hi - self.lo) / self.bins
        seen = 0
        for i, c in enumerate(self.counts):
            if c and seen + c >= target:
                frac = (target - seen) / c if c else 0.0
                return self.lo + (i + frac) * width
            seen += c
        return self.hi

    def summary(self, percentiles: Iterable[float]) -> dict:
        out = {"mean": _r(self.mean), "std": _r(self.std), "n": self.n}
        for p in percentiles:
            out[f"p{int(round(p * 100)):02d}"] = _r(self.percentile(p))
        return out


class Aggregator:
    """Collects measurements dicts (see measurements.py) into stats."""

    def __init__(self):
        self._stats: Dict[Tuple[str, str], RunningStat] = {}
        self._group_images: Dict[str, int] = {}

    def _stat(self, group: str, name: str) -> RunningStat:
        key = (group, name)
        if key not in self._stats:
            lo, hi, bins = BIN_SPECS.get(key, (0.0, 1.0, 200))
            self._stats[key] = RunningStat(lo, hi, bins)
        return self._stats[key]

    def add(self, measurements: dict) -> None:
        """Fold one image's measurements in, then forget them.

        Called exactly once per processed image; a group counter is
        bumped once per image that contributed to that group.
        """
        for group, values in measurements.items():
            if not isinstance(values, dict):
                continue
            contributed = False
            for name, value in values.items():
                if isinstance(value, (list, tuple)):
                    for v in value:
                        self._stat(group, name).add(float(v))
                        contributed = True
                elif isinstance(value, (int, float)):
                    self._stat(group, name).add(float(value))
                    contributed = True
            if contributed:
                self._group_images[group] = self._group_images.get(group, 0) + 1

    def group_sample_counts(self) -> Dict[str, int]:
        """How many distinct images contributed to each group."""
        return dict(sorted(self._group_images.items()))

    def stat(self, group: str, name: str):
        return self._stats.get((group, name))

    def published_stats(self, min_samples: int):
        """{(group, name): RunningStat} meeting the sample floor."""
        return {
            key: st for key, st in self._stats.items() if st.n >= min_samples
        }
