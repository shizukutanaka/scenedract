"""Stream pipeline: fetch -> filter -> extract -> aggregate -> discard.

One image at a time is held in memory.  Nothing about an image -- not
its bytes, not its landmarks, not any tag attached to it -- is ever
written to disk; the pipeline's only outputs are the knowledge model
files and a counts-only summary.  This is the design's core invariant
(2026-11-15, sections 0 and 2.2) and the test-suite pins it.

Sources are pluggable.  Two are shipped:

- :class:`DirectorySource` -- image files from a local directory
  (the owner's own verification data; order is sorted, hence
  deterministic).
- :class:`UrlListSource` -- image URLs fetched over HTTP with the
  retry behaviour ported from Tumblr-Image-Collector's fetch layer
  (retry on 429/5xx, honouring ``Retry-After``).  A Tumblr blog
  source plugs into the same interface; it is deliberately not
  bundled because the collector's API client carries its own
  credentials handling and the terms question (design section 5) is
  the owner's call.

The run lock follows the collector's ``auto_runner`` pattern
(exclusive-create lock file with owner metadata) so two learning
runs never interleave.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from .aggregate import Aggregator
from .extractor import PoseExtractor, image_size
from .filters import DuplicateFilter, perceptual_hash, quality_metrics
from .measurements import measurements_from_landmarks
from .model import build_model, build_provenance, content_hash, derive_rules, write_knowledge_model

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MIN_SAMPLES = 100


class DirectorySource:
    """Yields image bytes from a local directory, in sorted order."""

    def __init__(self, root):
        self._root = Path(root)

    def __iter__(self) -> Iterator[bytes]:
        for path in sorted(self._root.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield path.read_bytes()


class UrlListSource:
    """Fetches image bytes from URLs with collector-style retries."""

    def __init__(self, urls: Iterable[str], max_retries: int = 3, sleep: Callable[[float], None] = time.sleep):
        self._urls = list(urls)
        self._max_retries = max_retries
        self._sleep = sleep

    def _fetch(self, url: str) -> Optional[bytes]:
        delay = 1.0
        for attempt in range(self._max_retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "scenedract-anatomy/1"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == self._max_retries:
                    return None
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
                self._sleep(wait)
                delay *= 2
            except Exception:
                if attempt == self._max_retries:
                    return None
                self._sleep(delay)
                delay *= 2
        return None

    def __iter__(self) -> Iterator[bytes]:
        for url in self._urls:
            data = self._fetch(url)
            if data is not None:
                yield data


class RunLock:
    """Exclusive run lock (pattern ported from the collector's auto_runner)."""

    def __init__(self, path):
        self._path = Path(path)
        self._fd = None

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise RuntimeError(f"another learning run holds {self._path}") from e
        os.write(self._fd, f"pid={os.getpid()}\n".encode())
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass
        return False


class LearningPipeline:
    """The stream processor.  Holds no per-image state between items."""

    def __init__(
        self,
        extractor: PoseExtractor,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        quality_check: bool = True,
    ):
        self._extractor = extractor
        self._min_samples = min_samples
        self._quality_check = quality_check

    def run(self, source: Iterable[bytes], out_dir, generated_at: str) -> dict:
        aggregator = Aggregator()
        duplicates = DuplicateFilter()
        counts = {
            "imagesSeen": 0,
            "imagesRejectedQuality": 0,
            "imagesRejectedDuplicate": 0,
            "imagesNoPose": 0,
            "imagesUsed": 0,
        }
        for image_bytes in source:
            counts["imagesSeen"] += 1
            self._process_one(image_bytes, aggregator, duplicates, counts)
            # image_bytes goes out of scope here; nothing retains it.
        model = build_model(aggregator, generated_at, self._min_samples)
        rules = derive_rules(model)
        mhash = content_hash(model)
        rules["modelHash"] = mhash
        provenance = build_provenance(
            generated_at,
            getattr(self._extractor, "name", "unknown"),
            getattr(self._extractor, "version", "unknown"),
            counts,
            model,
            mhash,
        )
        write_knowledge_model(out_dir, model, rules, provenance)
        return {"counts": counts, "modelHash": mhash, "rules": len(rules["rules"])}

    def _process_one(self, image_bytes: bytes, aggregator: Aggregator, duplicates: DuplicateFilter, counts: dict) -> None:
        if self._quality_check:
            metrics = quality_metrics(image_bytes)
            if metrics is None or not metrics["ok"]:
                counts["imagesRejectedQuality"] += 1
                return
        if duplicates.is_duplicate(perceptual_hash(image_bytes)):
            counts["imagesRejectedDuplicate"] += 1
            return
        size = image_size(image_bytes)
        if size is None:
            counts["imagesRejectedQuality"] += 1
            return
        poses = self._extractor.extract(image_bytes)
        if not poses:
            counts["imagesNoPose"] += 1
            return
        width, height = size
        contributed = False
        for pose in poses:
            measurements = measurements_from_landmarks(pose, width, height)
            if measurements:
                aggregator.add(measurements)
                contributed = True
        counts["imagesUsed" if contributed else "imagesNoPose"] += 1
