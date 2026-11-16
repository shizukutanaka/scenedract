"""Tests for the anatomy knowledge model pipeline.

Run from the ``anatomy/`` directory:

    python -m unittest discover -s tests -v

The MediaPipe smoke test only runs when ANATOMY_POSE_MODEL points at
a pose_landmarker .task file (the model is user-supplied by design).
"""

import hashlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from anatomy_model.aggregate import Aggregator, RunningStat
from anatomy_model.filters import DuplicateFilter, perceptual_hash, quality_metrics
from anatomy_model.measurements import measurements_from_landmarks
from anatomy_model.model import build_model, content_hash, derive_rules, validate_knowledge_model
from anatomy_model.pipeline import DirectorySource, LearningPipeline

W = H = 1000


def skeleton(elbow_bend=0.0):
    """A standing figure in pixel coords (image is W x H), as a dict
    of BlazePose index -> (x, y).  ``elbow_bend`` shifts the wrists
    sideways so elbow angles vary between samples."""
    pts = {
        0: (500, 150),    # nose
        2: (480, 120), 5: (520, 120),      # eyes
        7: (465, 125), 8: (535, 125),      # ears
        11: (450, 220), 12: (550, 220),    # shoulders
        13: (430, 350), 14: (570, 350),    # elbows
        15: (435 + elbow_bend, 470), 16: (565 - elbow_bend, 470),  # wrists
        23: (460, 520), 24: (540, 520),    # hips
        25: (455, 700), 26: (545, 700),    # knees
        27: (455, 880), 28: (545, 880),    # ankles
        29: (450, 940), 30: (550, 940),    # heels
        31: (445, 950), 32: (555, 950),    # foot index
    }
    landmarks = []
    for i in range(33):
        x, y = pts.get(i, (500, 150))
        landmarks.append((x / W, y / H, 1.0))
    return landmarks


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class MeasurementsTest(unittest.TestCase):
    def test_proportions_match_hand_computed_values(self):
        m = measurements_from_landmarks(skeleton(), W, H)
        eye_mid, nose = (500, 120), (500, 150)
        stature = dist(eye_mid, (500, 950)) + 1.6 * dist(eye_mid, nose)
        props = m["proportions"]
        self.assertAlmostEqual(props["torso"], 300.0 / stature, places=4)
        self.assertAlmostEqual(props["upperArm"], dist((450, 220), (430, 350)) / stature, places=4)
        self.assertAlmostEqual(props["forearm"], dist((430, 350), (435, 470)) / stature, places=4)
        self.assertAlmostEqual(props["thigh"], dist((460, 520), (455, 700)) / stature, places=4)
        self.assertAlmostEqual(props["shin"], 180.0 / stature, places=4)
        self.assertNotIn("head", props)  # no crown/chin landmark: not measured, by design

    def test_joint_angles_and_face_and_grounding(self):
        m = measurements_from_landmarks(skeleton(), W, H)
        self.assertEqual(len(m["jointAngles"]["elbow"]), 2)  # both sides pooled later
        self.assertTrue(all(0.0 <= a <= 180.0 for a in m["jointAngles"]["elbow"]))
        self.assertAlmostEqual(m["faceRelations"]["eyeSpacingRatio"], 40.0 / 70.0, places=4)
        # COM (hip midpoint x=500) sits centred over the feet span
        # [445, 555]: margin 55 px over a 430 px COM height.
        self.assertAlmostEqual(m["grounding"]["comMarginRatio"], 55.0 / 430.0, places=4)

    def test_invisible_pose_yields_nothing(self):
        lm = [(0.5, 0.5, 0.0)] * 33
        self.assertEqual(measurements_from_landmarks(lm, W, H), {})


class AggregatorTest(unittest.TestCase):
    def test_welford_and_percentiles(self):
        st = RunningStat(0.0, 200.0, 200)
        for v in range(1, 101):
            st.add(float(v))
        self.assertEqual(st.n, 100)
        self.assertAlmostEqual(st.mean, 50.5, places=6)
        self.assertAlmostEqual(st.std, 29.0115, places=3)
        self.assertTrue(49.0 < st.percentile(0.5) < 52.0)

    def test_min_sample_guard(self):
        agg = Aggregator()
        for _ in range(5):
            agg.add({"proportions": {"upperArm": 0.18}})
        model = build_model(agg, "2026-01-01T00:00:00Z", min_samples=100)
        self.assertEqual(model["proportions"]["segments"], {})  # unpublished
        self.assertEqual(model["sampleCounts"]["proportions"], 5)  # but counted


class ModelAndRulesTest(unittest.TestCase):
    def _filled_aggregator(self):
        agg = Aggregator()
        rng = np.random.RandomState(42)
        for _ in range(150):
            agg.add({
                "proportions": {
                    "upperArm": 0.18 + rng.randn() * 0.01,
                    "thigh": 0.25 + rng.randn() * 0.01,
                },
                "jointAngles": {"elbow": [170.0 + rng.randn() * 5.0]},
            })
        return agg

    def test_rules_derive_from_statistics_with_basis(self):
        model = build_model(self._filled_aggregator(), "2026-01-01T00:00:00Z", min_samples=100)
        rules = derive_rules(model)
        by_id = {r["id"]: r for r in rules["rules"]}
        self.assertIn("proportion.upperArm", by_id)
        seg = model["proportions"]["segments"]["upperArm"]
        self.assertEqual(by_id["proportion.upperArm"]["accept"], [seg["p05"], seg["p95"]])
        self.assertEqual(by_id["proportion.upperArm"]["basis"]["stat"], "proportions.segments.upperArm")
        self.assertEqual(by_id["proportion.upperArm"]["basis"]["n"], seg["n"])
        self.assertEqual(validate_knowledge_model(model, rules, {
            "schemaVersion": 1,
            "contentHash": content_hash(model),
        }), [])

    def test_deterministic_bytes(self):
        from anatomy_model.model import canonical_json, content_hash

        m1 = build_model(self._filled_aggregator(), "2026-01-01T00:00:00Z", min_samples=100)
        m2 = build_model(self._filled_aggregator(), "2026-01-01T00:00:00Z", min_samples=100)
        self.assertEqual(canonical_json(m1), canonical_json(m2))
        self.assertEqual(content_hash(m1), content_hash(m2))
        self.assertEqual(canonical_json(derive_rules(m1)), canonical_json(derive_rules(m2)))


class FakeExtractor:
    """Deterministic stand-in: varies the skeleton by image hash."""

    name = "fake-extractor"
    version = "0"

    def extract(self, image_bytes):
        digest = hashlib.sha256(image_bytes).digest()
        bend = (digest[0] / 255.0 - 0.5) * 120.0
        return [skeleton(elbow_bend=bend)]


def noise_image_bytes(seed, w=W, h=H):
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class PipelineTest(unittest.TestCase):
    def _make_source(self, root, n=12):
        src = Path(root) / "source"
        src.mkdir(parents=True)
        hashes = {}
        for i in range(n):
            data = noise_image_bytes(i, w=800, h=600)
            (src / f"img{i:02d}.png").write_bytes(data)
            hashes[f"img{i:02d}.png"] = hashlib.sha256(data).hexdigest()
        return src, hashes

    def test_stream_end_to_end_and_determinism(self):
        with tempfile.TemporaryDirectory() as td:
            src, _ = self._make_source(td)
            out1, out2 = Path(td) / "out1", Path(td) / "out2"
            pipe = LearningPipeline(FakeExtractor(), min_samples=10)
            s1 = pipe.run(DirectorySource(src), out1, "2026-01-01T00:00:00Z")
            s2 = pipe.run(DirectorySource(src), out2, "2026-01-01T00:00:00Z")
            self.assertEqual(s1["counts"]["imagesUsed"], 12)
            self.assertEqual(s1["counts"]["imagesSeen"], 12)
            for name in ("model.json", "rules.json", "provenance.json"):
                self.assertEqual((out1 / name).read_bytes(), (out2 / name).read_bytes(), name)
            model = json.loads((out1 / "model.json").read_text())
            self.assertIn("upperArm", model["proportions"]["segments"])
            self.assertGreater(s1["rules"], 0)

    def test_nothing_is_retained(self):
        """The core privacy invariant: after a run, the only files are
        the three knowledge-model JSONs; they contain no image data,
        no landmark coordinates and no per-image records."""
        with tempfile.TemporaryDirectory() as td:
            src, before = self._make_source(td)
            out = Path(td) / "out"
            pipe = LearningPipeline(FakeExtractor(), min_samples=10)
            pipe.run(DirectorySource(src), out, "2026-01-01T00:00:00Z")
            # Output directory holds exactly the three artefacts.
            self.assertEqual(
                sorted(p.name for p in out.iterdir()),
                ["model.json", "provenance.json", "rules.json"],
            )
            # Nothing else appeared anywhere outside the source dir.
            for p in Path(td).rglob("*"):
                if p.is_file() and src not in p.parents:
                    self.assertIn(p.parent, (out,), f"unexpected file: {p}")
            # Source images are untouched.
            for name, digest in before.items():
                self.assertEqual(hashlib.sha256((src / name).read_bytes()).hexdigest(), digest)
            # Outputs carry statistics only: no per-image records, no
            # landmark/coordinate/tag structures anywhere in the JSON.
            def walk_keys(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        yield k
                        yield from walk_keys(v)
                elif isinstance(obj, list):
                    for v in obj:
                        yield from walk_keys(v)

            for name in ("model.json", "rules.json", "provenance.json"):
                parsed = json.loads((out / name).read_text())
                for key in walk_keys(parsed):
                    self.assertNotIn(key, ("landmarks", "coordinates", "tags", "images", "urls"))
                self.assertNotIn("img00", (out / name).read_text())
            prov = json.loads((out / "provenance.json").read_text())
            self.assertEqual(prov["counts"]["imagesSeen"], 12)
            # Provenance records counts and the extractor identity only.
            self.assertEqual(prov["extractor"]["name"], "fake-extractor")

    def test_quality_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "source"
            src.mkdir()
            good = noise_image_bytes(7)
            (src / "a.png").write_bytes(good)
            (src / "b.png").write_bytes(good)  # exact duplicate
            tiny = io.BytesIO()
            Image.new("RGB", (64, 64), (128, 128, 128)).save(tiny, format="PNG")
            (src / "c.png").write_bytes(tiny.getvalue())  # too small / flat
            out = Path(td) / "out"
            pipe = LearningPipeline(FakeExtractor(), min_samples=1)
            summary = pipe.run(DirectorySource(src), out, "2026-01-01T00:00:00Z")
            self.assertEqual(summary["counts"]["imagesUsed"], 1)
            self.assertEqual(summary["counts"]["imagesRejectedDuplicate"], 1)
            self.assertEqual(summary["counts"]["imagesRejectedQuality"], 1)


class FiltersTest(unittest.TestCase):
    def test_quality_thresholds(self):
        self.assertTrue(quality_metrics(noise_image_bytes(3))["ok"])
        buf = io.BytesIO()
        Image.new("RGB", (1000, 1000), (128, 128, 128)).save(buf, format="PNG")
        flat = quality_metrics(buf.getvalue())
        self.assertFalse(flat["ok"])  # entropy 0
        self.assertIsNone(quality_metrics(b"not an image"))

    def test_duplicate_filter(self):
        f = DuplicateFilter()
        h1 = perceptual_hash(noise_image_bytes(1))
        h2 = perceptual_hash(noise_image_bytes(2))
        self.assertFalse(f.is_duplicate(h1))
        self.assertTrue(f.is_duplicate(h1))
        self.assertFalse(f.is_duplicate(h2))


@unittest.skipUnless(os.environ.get("ANATOMY_POSE_MODEL"), "ANATOMY_POSE_MODEL not set")
class MediaPipeSmokeTest(unittest.TestCase):
    def test_extractor_loads_and_runs(self):
        from anatomy_model.extractor import MediaPipePoseExtractor

        ex = MediaPipePoseExtractor(os.environ["ANATOMY_POSE_MODEL"])
        try:
            buf = io.BytesIO()
            Image.new("RGB", (640, 480), (0, 0, 0)).save(buf, format="PNG")
            self.assertEqual(ex.extract(buf.getvalue()), [])  # blank: no pose, no crash
        finally:
            ex.close()


if __name__ == "__main__":
    unittest.main()
