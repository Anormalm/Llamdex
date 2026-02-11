"""
Perception-Grounded LLM Reasoning via Uploaded Experts (pg_vision).

This package contains:
- models: vision experts, projectors, aggregators, and LLM injection wrappers
- data: single-image and population tasks
- trainers: projector training logic
- eval: evaluation utilities
- scripts: experiment runners
"""

from . import models, data, trainers, eval, scripts  # noqa: F401

