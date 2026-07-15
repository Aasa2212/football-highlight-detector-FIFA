"""
generate_sample_video.py
-------------------------
Creates a synthetic "football match" video with a fluctuating crowd-noise
audio track containing a few loud spikes (simulating goals / big moments).

This is only meant to give you something to test the pipeline on if you
don't have a real match video handy. Swap it out for a real .mp4 anytime -
the rest of the pipeline doesn't care where the video came from.

Usage:
    python src/generate_sample_video.py --output sample_data/sample_match.mp4 --duration 60
"""

import argparse
import numpy as np
from scipy.io import wavfile
from moviepy import ColorClip, AudioArrayClip


def build_audio(duration_sec: int, sr: int = 22050, seed: int = 42) -> np.ndarray:
    """Builds a fake crowd-noise waveform with a handful of loud spikes."""
    rng = np.random.default_rng(seed)
    n_samples = duration_sec * sr

    # Base crowd murmur: filtered noise, moderate amplitude
    base_noise = rng.normal(0, 0.05, n_samples)
    # Smooth it a bit so it sounds like a murmur rather than static
    kernel = np.ones(200) / 200
    base_noise = np.convolve(base_noise, kernel, mode="same")

    # Slow amplitude drift so the crowd isn't perfectly flat
    t = np.linspace(0, duration_sec, n_samples)
    drift = 0.03 * np.sin(2 * np.pi * t / 20)

    audio = base_noise + drift

    # Inject a handful of "big moment" spikes: short bursts of high amplitude noise.
    # Skipped entirely for very short clips where there's no safe room to inject one.
    margin = sr * 3
    if n_samples > margin * 2:
        max_possible_spikes = max(1, (n_samples - margin * 2) // sr)
        n_spikes = min(rng.integers(4, 7), max_possible_spikes)
        spike_centers = rng.choice(
            np.arange(margin, n_samples - margin), size=n_spikes, replace=False
        )
    else:
        spike_centers = []
    for center in spike_centers:
        spike_len = rng.integers(int(sr * 1.5), int(sr * 3.5))  # 1.5-3.5s roar
        start = max(0, center - spike_len // 2)
        end = min(n_samples, start + spike_len)
        envelope = np.hanning(end - start)
        spike_noise = rng.normal(0, 1.0, end - start) * envelope
        audio[start:end] += spike_noise * 0.9

    # Normalize to [-1, 1]
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic sample match video.")
    parser.add_argument("--output", default="sample_data/sample_match.mp4")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--sr", type=int, default=22050, help="Audio sample rate")
    args = parser.parse_args()

    print(f"Generating {args.duration}s synthetic crowd audio...")
    audio = build_audio(args.duration, sr=args.sr)

    # moviepy wants stereo shape (n_samples, 2)
    stereo = np.column_stack([audio, audio])
    audio_clip = AudioArrayClip(stereo, fps=args.sr)

    # Simple color-shifting video track just so we have picture to cut.
    def make_frame(t):
        shade = int(40 + 30 * np.sin(t / 3))
        return np.full((360, 640, 3), fill_value=[shade, shade // 2, 255 - shade], dtype=np.uint8)

    video_clip = ColorClip(size=(640, 360), color=(20, 60, 20), duration=args.duration)
    video_clip.frame_function = make_frame
    final = video_clip.with_audio(audio_clip)

    print(f"Writing video to {args.output} ...")
    final.write_videofile(args.output, fps=24, codec="libx264", audio_codec="aac")
    print("Done.")


if __name__ == "__main__":
    main()
