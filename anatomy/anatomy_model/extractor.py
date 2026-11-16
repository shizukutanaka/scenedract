"""Pose extraction: the pluggable front end of the pipeline.

An extractor turns image bytes (held in memory only) into pose
landmarks.  The pipeline depends only on the :class:`PoseExtractor`
protocol, so the concrete model can be replaced without touching the
aggregation or export code.

The one real implementation shipped here is
:class:`MediaPipePoseExtractor`, backed by MediaPipe's PoseLandmarker
(Tasks API).  The model file is *not* bundled with this repository --
point ``ANATOMY_POSE_MODEL`` (or the constructor argument) at a
``pose_landmarker*.task`` file you obtained yourself, so the model's
licence terms stay between you and its publisher.  Extraction runs
fully on-device; nothing is uploaded anywhere.
"""

from __future__ import annotations

import io
from typing import List, Optional, Protocol, Sequence

from .measurements import Landmark


class PoseExtractor(Protocol):
    """Anything that can read pose landmarks out of image bytes."""

    name: str
    version: str

    def extract(self, image_bytes: bytes) -> List[List[Landmark]]:
        """Return a list of poses (usually 0 or 1).

        Each pose is 33 ``(x, y, visibility)`` landmarks in normalised
        image coordinates, plus the image size is recoverable by the
        caller from the bytes themselves.
        """
        ...


class MediaPipePoseExtractor:
    """MediaPipe PoseLandmarker-backed extractor (Tasks API)."""

    name = "mediapipe-pose-landmarker"

    def __init__(self, model_path: str):
        import mediapipe as mp  # deferred: heavy import, tests may skip

        self._mp = mp
        self.version = getattr(mp, "__version__", "unknown")
        self._model_path = model_path
        self._landmarker = None

    def _ensure(self):
        if self._landmarker is None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=self._model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        return self._landmarker

    def extract(self, image_bytes: bytes) -> List[List[Landmark]]:
        import numpy as np
        from PIL import Image

        landmarker = self._ensure()
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = np.asarray(img.convert("RGB"))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        poses: List[List[Landmark]] = []
        for pose in result.pose_landmarks or []:
            poses.append(
                [(lm.x, lm.y, getattr(lm, "visibility", 1.0)) for lm in pose]
            )
        return poses

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


def image_size(image_bytes: bytes) -> Optional[tuple]:
    """(width, height) of an in-memory image, or None if unreadable."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.size
    except Exception:
        return None
