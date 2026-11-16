# Anatomy knowledge model

SceneDract's scene energy measures geometry and physics, but nothing
about the *human figures* in a scene: are their proportions and poses
within what real human bodies actually do?  This component learns
that knowledge from images and supplies it to SceneDract as data.

It is the SceneDract-side integration of the Tumblr-Image-Collector
functional redefinition (design of 2026-11-15): the collector's job
is no longer "save images with tags" -- images are a *transient
input* from which statistical knowledge about human body structure
is extracted.  The collector repository itself is untouched by this
integration; its reusable parts (fetch retry behaviour, the image
quality filter, perceptual-hash duplicate rejection, the run-lock
pattern) were ported here, as recorded in the module docstrings.

## The privacy invariant

Learning is stream processing on the user's own machine:

```
fetch -> quality/duplicate filter -> pose extraction -> aggregate -> DISCARD
```

An image exists only as bytes in memory while it is measured.  The
pipeline never writes image bytes, landmark coordinates or tags to
disk -- the tests pin this (`test_nothing_is_retained`).  The only
outputs are the knowledge model files below, which contain aggregate
statistics and counts only.  Per-statistic publication requires at
least 100 samples by default, so no published statistic can mirror
an individual.

## The knowledge model

Three JSON files, written together:

| file | contents |
|---|---|
| `model.json` | statistics: segment-proportion distributions (upperArm / forearm / thigh / shin / torso, as fractions of stature), joint-angle distributions (elbow / knee / shoulder / hip, degrees), grounding (COM margin ratio), face relations (eye spacing ratio). Each with mean / std / percentiles / n. |
| `rules.json` | validation rules derived mechanically from the statistics (accept = p05-p95 for proportions, p01-p99 for joint angles). Every rule records its basis: which statistic, which method, how many samples. Carries the model's content hash. |
| `provenance.json` | counts only (images seen / rejected / used, samples per group), the extractor name and version, the content hash of `model.json`, and the method note. No image data, no source names or URLs. |

Serialisation is canonical, so the same measurement stream with the
same `generatedAt` produces byte-identical files.

## Method, stated plainly

- Extractor: MediaPipe PoseLandmarker (Tasks API), one pose per
  image.  The model file is **not bundled** -- obtain a
  `pose_landmarker*.task` yourself (Google publishes them under
  Apache-2.0) and pass `--model` / `ANATOMY_POSE_MODEL`.  Extraction
  is fully on-device.
- Segments are joint-to-joint distances -- the same quantities the
  SceneDract consumer derives from its mannequin part table, so the
  learned ranges and the scene figures are comparable.
- Stature is estimated as eye-midpoint-to-ground distance plus a
  crown offset of 1.6 x the eye-to-nose distance (facial-thirds
  approximation; see `measurements.py`).  Proportions are only
  collected from near-upright poses.
- `head` proportions are **not** measured: pose landmarks have no
  crown or chin point, so a head length would be invented, not
  measured.  The schema reserves the slot.
- Grounding is an image-plane proxy of the physics audit's 3D
  quantity and is labelled as such.
- Joint angles have no consumer yet: the Scene IR mannequin carries
  no pose, so SceneDract currently judges proportions only (below).

## Running

```sh
pip install -r requirements.txt
python -m anatomy_model.cli learn \
    --source-dir /path/to/your/verification-images \
    --out /path/to/knowledge-model \
    --model /path/to/pose_landmarker_lite.task
# or: --url-list urls.txt   (one image URL per line)
```

Sources are pluggable (`DirectorySource`, `UrlListSource`); a Tumblr
blog source fits the same interface but is deliberately not bundled
(credentials handling and the source-terms question belong to the
collector side / the owner).

## How SceneDract consumes it

`app/SceneDract.html` gained a seventh scene-energy component,
`anatomy` (weight 2, like the other rate components):

- Provide rules by embedding them in the IR as `ir.anatomyRules`
  (the parsed contents of `rules.json`), or load a `rules.json`
  file in the app: the validation card's "Anatomy knowledge model"
  box reads the file locally, validates it before applying, shows
  the loaded file and rule counts (segment-proportion rules judged
  vs recorded-only), and on failure shows the reason and keeps the
  state in effect before the attempt. Programmatically the same
  path is `setAnatomyRules(rules)`.
- For every mannequin figure, each segment ratio derived from
  `MANNEQUIN_PARTS` is checked against the matching
  `segmentRatio.*` rule; the component is the violated share of all
  checks, with the violations listed in the energy breakdown.
- With no rules loaded, or no mannequin in the scene, the component
  is `n/a` and its weight drops out of E -- scenes without a
  knowledge model behave exactly as before.

## Tests

```sh
cd anatomy
python -m unittest discover -s tests -v
# real-extractor smoke test (needs the model file + MediaPipe):
ANATOMY_POSE_MODEL=/path/to/pose_landmarker_lite.task \
  python -m unittest discover -s tests -v
```

The suite proves: measurements match hand-computed geometry;
aggregation (Welford + histogram percentiles) is correct; the
minimum-sample guard withholds thin statistics; rules derive from
statistics with their basis recorded; the pipeline is deterministic;
and after a run, nothing but the three knowledge-model files exists
-- no images, no coordinates, no tags.
