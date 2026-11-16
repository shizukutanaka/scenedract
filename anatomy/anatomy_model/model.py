"""Knowledge model construction, rule derivation and export.

The knowledge model is three JSON files (design section 2.4):

- ``model.json`` -- the statistics themselves.
- ``rules.json`` -- validation rules derived mechanically from the
  statistics.  Every rule records its basis (which statistic, which
  percentile method, how many samples), so a consumer can explain
  *why* a threshold is what it is.
- ``provenance.json`` -- counts only: how many images were processed,
  how many samples back each group, which extractor produced the
  measurements, and a content hash of ``model.json``.  Image data,
  source names/URLs and tags are never recorded.

All serialisation is canonical (sorted keys, fixed float rounding,
fixed separators), so the same measurement stream with the same
``generated_at`` produces byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

from . import SCHEMA_VERSION
from .aggregate import Aggregator, RunningStat

# Percentile summaries published per family (design section 2.4).
_SEGMENT_PCTS = (0.05, 0.50, 0.95)
_JOINT_PCTS = (0.01, 0.50, 0.99)
_SCALAR_PCTS = (0.05, 0.50, 0.95)

# Rule derivation: (group, name) -> (rule id, target, percentiles, severity)
_RULE_SPECS = []
for _seg in ("head", "upperArm", "forearm", "thigh", "shin", "torso"):
    _RULE_SPECS.append(
        ("proportions", _seg, f"proportion.{_seg}", f"segmentRatio.{_seg}", (0.05, 0.95), "p05-p95", "warn")
    )
for _joint in ("elbow", "knee", "shoulder", "hip"):
    _RULE_SPECS.append(
        ("jointAngles", _joint, f"jointAngle.{_joint}", f"jointAngle.{_joint}", (0.01, 0.99), "p01-p99", "warn")
    )
_RULE_SPECS.append(
    ("grounding", "comMarginRatio", "grounding.comMarginRatio", "grounding.comMarginRatio", (0.05, 0.95), "p05-p95", "warn")
)
_RULE_SPECS.append(
    ("faceRelations", "eyeSpacingRatio", "face.eyeSpacingRatio", "faceRatio.eyeSpacingRatio", (0.05, 0.95), "p05-p95", "warn")
)


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def build_model(aggregator: Aggregator, generated_at: str, min_samples: int) -> dict:
    """Assemble ``model.json`` content from an aggregator.

    Statistics below ``min_samples`` are omitted (design section
    2.2); their absence is visible through ``sampleCounts`` and each
    published statistic's own ``n``.
    """
    published = aggregator.published_stats(min_samples)

    def stat_or_none(group: str, name: str, pcts) -> dict:
        st = published.get((group, name))
        return st.summary(pcts) if st is not None else None

    segments = {}
    for seg in ("head", "upperArm", "forearm", "thigh", "shin", "torso"):
        s = stat_or_none("proportions", seg, _SEGMENT_PCTS)
        if s is not None:
            segments[seg] = s
    joints = {}
    for joint in ("elbow", "knee", "shoulder", "hip"):
        s = stat_or_none("jointAngles", joint, _JOINT_PCTS)
        if s is not None:
            joints[joint] = s
    grounding = {}
    s = stat_or_none("grounding", "comMarginRatio", _SCALAR_PCTS)
    if s is not None:
        grounding["comMarginRatio"] = s
    face = {}
    s = stat_or_none("faceRelations", "eyeSpacingRatio", _SCALAR_PCTS)
    if s is not None:
        face["eyeSpacingRatio"] = s

    counts = aggregator.group_sample_counts()
    return {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": generated_at,
        "sampleCounts": {
            "proportions": counts.get("proportions", 0),
            "jointAngles": counts.get("jointAngles", 0),
            "grounding": counts.get("grounding", 0),
            "faceRelations": counts.get("faceRelations", 0),
        },
        "proportions": {"segments": segments},
        "jointAngles": joints,
        "grounding": grounding,
        "faceRelations": face,
    }


def content_hash(model: dict) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(model).encode("utf-8")).hexdigest()


def derive_rules(model: dict) -> dict:
    """Derive ``rules.json`` from a built model, mechanically.

    A rule exists exactly when its backing statistic was published;
    the accept interval is the stated percentile range of that
    statistic, and the basis points back at it.
    """
    def lookup(group: str, name: str):
        if group == "proportions":
            return model.get("proportions", {}).get("segments", {}).get(name)
        return model.get(group, {}).get(name)

    rules = []
    for group, name, rule_id, target, (plo, phi), method, severity in _RULE_SPECS:
        stat = lookup(group, name)
        if stat is None:
            continue
        lo_key = f"p{int(round(plo * 100)):02d}"
        hi_key = f"p{int(round(phi * 100)):02d}"
        if lo_key not in stat or hi_key not in stat:
            continue
        stat_path = (
            f"proportions.segments.{name}" if group == "proportions" else f"{group}.{name}"
        )
        rules.append(
            {
                "id": rule_id,
                "target": target,
                "accept": [stat[lo_key], stat[hi_key]],
                "basis": {"stat": stat_path, "method": method, "n": stat["n"]},
                "severity": severity,
            }
        )
    return {"schemaVersion": SCHEMA_VERSION, "rules": rules}


def build_provenance(
    generated_at: str,
    extractor_name: str,
    extractor_version: str,
    counts: Dict[str, int],
    model: dict,
    model_hash: str,
) -> dict:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": generated_at,
        "extractor": {"name": extractor_name, "version": extractor_version},
        "counts": dict(sorted(counts.items())),
        "sampleCounts": model["sampleCounts"],
        "contentHash": model_hash,
        "method": (
            "stream processing: images are fetched, filtered, measured and "
            "aggregated in memory only; source images, landmark coordinates "
            "and tags are discarded and never written to disk. Only counts "
            "are recorded here -- no image data, no source names or URLs."
        ),
    }


def write_knowledge_model(out_dir, model: dict, rules: dict, provenance: dict) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, obj in (("model.json", model), ("rules.json", rules), ("provenance.json", provenance)):
        path = out / name
        path.write_text(canonical_json(obj), encoding="utf-8")
        written[name] = path
    return written


def validate_knowledge_model(model: dict, rules: dict, provenance: dict) -> list:
    """Light structural validation; returns a list of problems ([] = ok)."""
    problems = []
    for label, obj in (("model", model), ("rules", rules), ("provenance", provenance)):
        if obj.get("schemaVersion") != SCHEMA_VERSION:
            problems.append(f"{label}: schemaVersion != {SCHEMA_VERSION}")
    if provenance.get("contentHash") != content_hash(model):
        problems.append("provenance: contentHash does not match model.json")
    model_stats = set()
    for seg in model.get("proportions", {}).get("segments", {}):
        model_stats.add(f"proportions.segments.{seg}")
    for group in ("jointAngles", "grounding", "faceRelations"):
        for name in model.get(group, {}):
            model_stats.add(f"{group}.{name}")
    for rule in rules.get("rules", []):
        basis = rule.get("basis", {})
        if basis.get("stat") not in model_stats:
            problems.append(f"rule {rule.get('id')}: basis stat {basis.get('stat')} not in model")
        accept = rule.get("accept", [])
        if len(accept) != 2 or accept[0] > accept[1]:
            problems.append(f"rule {rule.get('id')}: accept is not an ordered [lo, hi] pair")
    return problems
