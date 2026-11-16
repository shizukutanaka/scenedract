"""Command line entry point.

    python -m anatomy_model.cli learn \
        --source-dir /path/to/verification-images \
        --out /path/to/knowledge-model \
        --model /path/to/pose_landmarker_lite.task

or, with URLs listed one per line in a text file:

    python -m anatomy_model.cli learn --url-list urls.txt --out ...

The extractor model file is supplied by the user (see extractor.py);
``--model`` defaults to the ``ANATOMY_POSE_MODEL`` environment
variable.  ``--min-samples`` defaults to 100 (design section 2.2):
statistics backed by fewer samples are not published.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import DEFAULT_MIN_SAMPLES, DirectorySource, LearningPipeline, RunLock, UrlListSource


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="anatomy_model.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    learn = sub.add_parser("learn", help="run the learning pipeline once")
    src = learn.add_mutually_exclusive_group(required=True)
    src.add_argument("--source-dir", help="directory of local image files")
    src.add_argument("--url-list", help="text file with one image URL per line")
    learn.add_argument("--out", required=True, help="output directory for the knowledge model")
    learn.add_argument("--model", default=os.environ.get("ANATOMY_POSE_MODEL"),
                       help="path to a pose_landmarker .task model file")
    learn.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    learn.add_argument("--no-quality-check", action="store_true",
                       help="skip the resolution/entropy/sharpness input filter")
    args = parser.parse_args(argv)

    if args.command == "learn":
        if not args.model:
            parser.error("--model (or ANATOMY_POSE_MODEL) is required")
        from .extractor import MediaPipePoseExtractor

        if args.source_dir:
            source = DirectorySource(args.source_dir)
        else:
            urls = [
                line.strip()
                for line in Path(args.url_list).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            ]
            source = UrlListSource(urls)
        extractor = MediaPipePoseExtractor(args.model)
        pipeline = LearningPipeline(
            extractor, min_samples=args.min_samples, quality_check=not args.no_quality_check
        )
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out = Path(args.out)
        try:
            with RunLock(out / ".learn.lock"):
                summary = pipeline.run(source, out, generated_at)
        finally:
            extractor.close()
        print(
            "learned: {imagesUsed}/{imagesSeen} images used, "
            "{rules} rules, model {modelHash}".format(rules=summary["rules"], modelHash=summary["modelHash"], **summary["counts"])
        )
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
