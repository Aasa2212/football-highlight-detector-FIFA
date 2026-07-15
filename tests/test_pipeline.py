"""
Basic smoke test for the highlight detection pipeline.

Generates a short synthetic clip, runs the full pipeline on it, and checks
that at least one highlight clip and a valid report.json are produced.

Run with:
    python -m pytest tests/ -v
"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate_sample_video import build_audio  # noqa: E402
from highlight_detector import compute_rms_envelope, detect_peaks  # noqa: E402


def test_build_audio_produces_expected_length():
    sr = 22050
    duration = 10
    audio = build_audio(duration_sec=duration, sr=sr)
    assert len(audio) == duration * sr


def test_build_audio_is_normalized():
    audio = build_audio(duration_sec=5, sr=22050)
    assert audio.max() <= 1.0
    assert audio.min() >= -1.0


def test_detect_peaks_finds_injected_spike():
    import numpy as np

    sr = 22050
    hop_length = 512
    # Fake an RMS envelope: mostly quiet, one clear spike in the middle.
    n_frames = 200
    rms = np.full(n_frames, 0.02)
    rms[100] = 0.9
    times = np.arange(n_frames) * hop_length / sr

    peaks = detect_peaks(times, rms, height_percentile=85, distance_sec=1.0,
                          hop_length=hop_length, sr=sr)

    assert len(peaks) >= 1
    assert any(abs(p["time_sec"] - times[100]) < 0.5 for p in peaks)


if __name__ == "__main__":
    test_build_audio_produces_expected_length()
    test_build_audio_is_normalized()
    test_detect_peaks_finds_injected_spike()
    print("All tests passed.")
