"""Offline logic tests for batcher.py - no network, no API key needed."""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batcher import pick_prompt, WEIGHTS  # noqa: E402


def test_mixer_covers_all_classes():
    rng = random.Random(7)
    classes = [pick_prompt(rng)[0] for _ in range(2000)]
    assert set(classes) == set(WEIGHTS.keys()), f"missing classes: {set(WEIGHTS) - set(classes)}"


def test_mixer_respects_weights():
    rng = random.Random(7)
    classes = [pick_prompt(rng)[0] for _ in range(2000)]
    c = Counter(classes)
    for cls, w in WEIGHTS.items():
        observed = c[cls] / 2000
        assert abs(observed - w) < 0.06, f"{cls}: observed {observed:.3f} vs weight {w}"


def test_prompts_nonempty():
    rng = random.Random(1)
    for _ in range(100):
        _, prompt = pick_prompt(rng)
        assert isinstance(prompt, str) and len(prompt) > 10


def test_knee_detection_logic():
    """Knee: first level where scaling drops below 65% of expected linear."""
    results = [
        {"concurrency": 1, "aggregate_tok_s": 40.0, "ok": 8, "requests": 8,
         "failed": 0, "wall_p50_s": 2.0, "wall_p95_s": 3.0, "wall_max_s": 3.0,
         "per_stream_tok_s": 40.0, "prompt_tokens": 0, "completion_tokens": 0,
         "completion_tok_s": 0, "fail_samples": []},
        {"concurrency": 2, "aggregate_tok_s": 80.0, "ok": 8, "requests": 8,
         "wall_p50_s": 4.0, "wall_p95_s": 5.0, "wall_max_s": 5.0,
         "per_stream_tok_s": 40.0, "prompt_tokens": 0, "completion_tokens": 0,
         "completion_tok_s": 0, "fail_samples": []},
        {"concurrency": 4, "aggregate_tok_s": 90.0, "ok": 8, "requests": 8,
         "wall_p50_s": 8.0, "wall_p95_s": 9.0, "wall_max_s": 9.0,
         "per_stream_tok_s": 22.5, "prompt_tokens": 0, "completion_tokens": 0,
         "completion_tok_s": 0, "fail_samples": []},
    ]
    # perfect scaling 1->2 (80/40=1.0x expected 2.0x... wait: scaling metric is
    # actual/expected; here 2.0 actual / 2.0 expected = 1.0 -> no knee)
    # 2->4: 90/80 = 1.125 actual vs 2.0 expected -> ratio 0.5625 < 0.65 -> knee at 4
    knee = None
    for prev, cur in zip(results, results[1:]):
        scaling = cur["aggregate_tok_s"] / prev["aggregate_tok_s"]
        expected = cur["concurrency"] / prev["concurrency"]
        if scaling < expected * 0.65:
            knee = cur["concurrency"]
            break
    assert knee == 4, f"knee should be 4, got {knee}"


def test_no_knee_when_scaling_linear():
    results = [
        {"concurrency": 1, "aggregate_tok_s": 40.0, "ok": 8, "requests": 8,
         "wall_p50_s": 2.0, "wall_p95_s": 3.0, "wall_max_s": 3.0,
         "per_stream_tok_s": 40.0, "prompt_tokens": 0, "completion_tokens": 0,
         "completion_tok_s": 0, "fail_samples": []},
        {"concurrency": 2, "aggregate_tok_s": 80.0, "ok": 8, "requests": 8,
         "wall_p50_s": 4.0, "wall_p95_s": 5.0, "wall_max_s": 5.0,
         "per_stream_tok_s": 40.0, "prompt_tokens": 0, "completion_tokens": 0,
         "completion_tok_s": 0, "fail_samples": []},
    ]
    knee = None
    for prev, cur in zip(results, results[1:]):
        scaling = cur["aggregate_tok_s"] / prev["aggregate_tok_s"]
        expected = cur["concurrency"] / prev["concurrency"]
        if scaling < expected * 0.65:
            knee = cur["concurrency"]
            break
    assert knee is None
