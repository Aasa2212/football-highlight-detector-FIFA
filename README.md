# ⚽ Automatic Football Highlight Detector

Feed in a football (soccer) match video and automatically extract the exciting
moments — goals, near-misses, big saves — by detecting spikes in crowd/commentator
loudness. No manual scrubbing through 90 minutes of footage required.

## How it works

The core idea: **exciting moments are loud moments.** When something big happens,
the crowd roars and the commentator's voice jumps. This pipeline:

1. **Extracts audio** from the match video (`moviepy`)
2. **Computes an RMS loudness envelope** over time (`librosa`) — basically, how
   loud the audio is at every point in the match
3. **Detects peaks** in that envelope using `scipy.signal.find_peaks` — these are
   your candidate highlight moments
4. **Cuts a short clip** around each detected peak (default: 5s before, 10s after)
5. **Saves the clips** plus a `report.json` describing every highlight found

```
Video → Extract Audio → RMS Loudness → Peak Detection → Cut Clips → report.json
```

## Demo

Don't have a match video handy? This repo ships with a synthetic sample
video generator that creates a fake "match" with a few loud crowd-roar spikes,
so you can see the full pipeline run without any real footage.

```bash
python src/generate_sample_video.py --output sample_data/sample_match.mp4 --duration 45
python src/highlight_detector.py --input sample_data/sample_match.mp4 --output output/
```

That's it — check the `output/` folder for the highlight clips and `report.json`.

## Setup

Requires **Python 3.10+** and `ffmpeg` installed on your system (moviepy depends on it).

```bash
git clone https://github.com/<your-username>/football-highlight-detector.git
cd football-highlight-detector
pip install -r requirements.txt
```

## Usage

Run on a real match video:

```bash
python src/highlight_detector.py --input path/to/match.mp4 --output output/
```

### Options

| Flag | Default | What it does |
|---|---|---|
| `--input` | *(required)* | Path to the input match video |
| `--output` | `output/` | Directory where highlight clips + report.json are saved |
| `--height-percentile` | `85` | Only moments louder than this percentile of the whole match count as candidates. Raise it (e.g. `92`) for fewer, stronger highlights; lower it (e.g. `75`) to catch more moments |
| `--distance` | `8.0` | Minimum seconds between two detected highlights, so one long roar doesn't produce five overlapping clips |
| `--clip-before` | `5.0` | Seconds of footage to include *before* the detected peak |
| `--clip-after` | `10.0` | Seconds of footage to include *after* the detected peak |

Example — fewer, longer highlights:

```bash
python src/highlight_detector.py --input match.mp4 --height-percentile 92 --clip-before 8 --clip-after 15
```

## Output

```
output/
├── highlight_01_128s.mp4
├── highlight_02_412s.mp4
├── highlight_03_701s.mp4
└── report.json
```

`report.json` includes the exact timestamp, loudness score, and clip boundaries
for every detected highlight — useful if you want to feed this into another
tool (e.g. auto-generate a highlight reel or subtitle overlay) instead of just
watching the individual clips.

## Project structure

```
football-highlight-detector/
├── src/
│   ├── highlight_detector.py      # Core pipeline: audio extraction, peak detection, clip cutting
│   └── generate_sample_video.py   # Generates a synthetic demo video (no real footage needed)
├── sample_data/                   # Sample/demo videos live here
├── output/                        # Generated highlight clips + report.json land here
├── tests/
│   └── test_pipeline.py           # Smoke tests for the audio generation + peak detection logic
├── requirements.txt
└── README.md
```

## Tuning false positives

Loud crowd noise isn't *always* a highlight — sometimes it's just sustained
background noise, chanting, or a false alarm. A few ways to tighten things up:

- Raise `--height-percentile` so only genuinely exceptional spikes qualify
- Increase `--distance` to avoid clustering near-duplicate detections
- **Optional upgrade:** combine this audio-based approach with scene-change
  detection (e.g. [PySceneDetect](https://www.scenedetect.com/)) to only keep
  highlights that also coincide with a camera cut — goals and replays almost
  always trigger one.

## Tech stack

- **Python 3.10+**
- [`librosa`](https://librosa.org/) — audio loudness analysis
- [`moviepy`](https://zulko.github.io/moviepy/) — video/audio extraction and clip cutting
- [`numpy`](https://numpy.org/) / [`scipy`](https://scipy.org/) — signal processing, peak detection

## License

MIT — see [LICENSE](LICENSE).
