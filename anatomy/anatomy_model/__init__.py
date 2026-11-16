"""Anatomy knowledge model for SceneDract.

Learns *statistical* knowledge about human body structure (segment
proportions, joint-angle ranges, grounding, face relations) from images
that are processed as a stream and never stored: an image is fetched,
filtered, measured and aggregated in memory, and then discarded.  The
only artefacts ever written are the knowledge model files
(``model.json`` / ``rules.json`` / ``provenance.json``), which contain
aggregate statistics, validation rules derived from them, and counts
only -- never images, landmark coordinates, tags or source identifiers.

Design: ``docs`` in this directory (README.md) and the functional
redefinition design of 2026-11-15 (Tumblr-Image-Collector redesign,
stages 0-2).  The consumer side lives in ``app/SceneDract.html``
(``anatomyCensus`` / the ``anatomy`` component of ``sceneEnergy``).
"""

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION"]
