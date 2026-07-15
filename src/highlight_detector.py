"""
highlight_detector.py
----------------------
Automatic Highlight Detector for football (soccer) match videos.

Pipeline:
    1. Extract audio track from the input video (MoviePy).
    2. Compute RMS (loudness) envelope of the audio over time (librosa).
    3. Detect peaks in the RMS envelope -> candidate highlight moments
       (crowd roars, commentator excitement, etc.) using scipy.signal.find_peaks.
    4. Cut a clip around each detected peak (MoviePy subclip).
    5. Save clips to an output directory, plus a JSON report of timestamps.

Usage:
    python src/highlight_detector.py --input sample_data/sample_match.mp4 --output output/

Tune sensitivity with --height / --distance / --clip-before / --clip-after.
See README.md for a full explanation of each parameter.
"""

import argparse
import json
import os

import librosa
import numpy as np
from moviepy import VideoFileClip
from scipy.signal import find_peaks


def extract_audio(video_path: str, temp_audio_path: str, sr: int = 22050):
    """Extracts the audio track from a video file and writes it to a WAV file."""
    print(f"[1/6] Loading video: {video_path}")
    clip = VideoFileClip(video_path)

    if clip.audio is None:
        clip.close()
        raise ValueError("The input video has no audio track to analyze.")

    print(f"[2/6] Extracting audio -> {temp_audio_path}")
    clip.audio.write_audiofile(temp_audio_path, fps=sr)
    duration = clip.duration
    clip.close()
    return duration


def compute_rms_envelope(audio_path: str, sr: int = 22050, hop_length: int = 512):
    """Loads audio and computes the RMS loudness envelope over time."""
    print("[3/6] Computing RMS loudness envelope...")
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return times, rms


def detect_peaks(times, rms, height_percentile: float = 85, distance_sec: float = 8.0,
                  hop_length: int = 512, sr: int = 22050):
    """
    Finds peaks in the RMS envelope that likely correspond to exciting moments.

    height_percentile: only peaks louder than this percentile of the whole match
                        are considered candidates (higher = fewer, stronger highlights).
    distance_sec:       minimum spacing between two detected highlights, so we don't
                         cut 5 clips out of the same 10-second roar.
    """
    print(f"[4/6] Detecting peaks (height_percentile={height_percentile}, "
          f"min_distance={distance_sec}s)...")

    height = np.percentile(rms, height_percentile)
    distance_frames = int((distance_sec * sr) / hop_length)

    peak_indices, properties = find_peaks(
        rms, height=height, distance=max(distance_frames, 1)
    )

    peaks = [
        {"time_sec": float(times[i]), "loudness": float(rms[i])}
        for i in peak_indices
    ]
    peaks.sort(key=lambda p: p["time_sec"])
    print(f"      -> {len(peaks)} candidate highlight(s) found.")
    return peaks


def cut_clips(video_path: str, peaks: list, output_dir: str,
              clip_before: float = 5.0, clip_after: float = 10.0, video_duration: float = None):
    """Cuts a clip around each detected peak timestamp and saves it to output_dir."""
    print(f"[5/6] Cutting {len(peaks)} clip(s)...")
    os.makedirs(output_dir, exist_ok=True)
    clip = VideoFileClip(video_path)
    duration = video_duration or clip.duration

    results = []
    for idx, peak in enumerate(peaks, start=1):
        center = peak["time_sec"]
        start = max(0, center - clip_before)
        end = min(duration, center + clip_after)

        out_path = os.path.join(output_dir, f"highlight_{idx:02d}_{int(center)}s.mp4")
        subclip = clip.subclipped(start, end)
        subclip.write_videofile(out_path, codec="libx264", audio_codec="aac")
        subclip.close()

        results.append({
            "clip": os.path.basename(out_path),
            "peak_time_sec": round(center, 2),
            "clip_start_sec": round(start, 2),
            "clip_end_sec": round(end, 2),
            "loudness": round(peak["loudness"], 4),
        })

    clip.close()
    return results


def run_pipeline(input_path: str, output_dir: str, height_percentile: float = 85,
                  distance_sec: float = 8.0, clip_before: float = 5.0, clip_after: float = 10.0,
                  sr: int = 22050):
    os.makedirs(output_dir, exist_ok=True)
    temp_audio_path = os.path.join(output_dir, "_temp_audio.wav")

    duration = extract_audio(input_path, temp_audio_path, sr=sr)
    times, rms = compute_rms_envelope(temp_audio_path, sr=sr)
    peaks = detect_peaks(times, rms, height_percentile=height_percentile,
                          distance_sec=distance_sec, sr=sr)

    if not peaks:
        print("No highlights detected. Try lowering --height-percentile.")
        os.remove(temp_audio_path)
        return []

    results = cut_clips(input_path, peaks, output_dir,
                         clip_before=clip_before, clip_after=clip_after,
                         video_duration=duration)

    print("[6/6] Writing report.json...")
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump({
            "input_video": input_path,
            "num_highlights": len(results),
            "parameters": {
                "height_percentile": height_percentile,
                "distance_sec": distance_sec,
                "clip_before": clip_before,
                "clip_after": clip_after,
            },
            "highlights": results,
        }, f, indent=2)

    os.remove(temp_audio_path)
    print(f"\nDone. {len(results)} highlight clip(s) saved to '{output_dir}/'.")
    print(f"Report: {report_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Automatic football highlight detector.")
    parser.add_argument("--input", required=True, help="Path to input match video (.mp4)")
    parser.add_argument("--output", default="output/", help="Directory to save highlight clips")
    parser.add_argument("--height-percentile", type=float, default=85,
                         help="Loudness percentile threshold (higher = fewer/stronger highlights)")
    parser.add_argument("--distance", type=float, default=8.0,
                         help="Minimum seconds between two detected highlights")
    parser.add_argument("--clip-before", type=float, default=5.0,
                         help="Seconds of footage to include before the peak")
    parser.add_argument("--clip-after", type=float, default=10.0,
                         help="Seconds of footage to include after the peak")
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        height_percentile=args.height_percentile,
        distance_sec=args.distance,
        clip_before=args.clip_before,
        clip_after=args.clip_after,
    )


if __name__ == "__main__":
    main()
