# Lofi Preset Generation Guide

Generate high-quality lofi music from the `twitch-musicplayer` preset library using the
Stability AI stable-audio CLI. The wrapper script [`generate_from_lofi_presets.py`](generate_from_lofi_presets.py)
reads YAML presets, builds rich text prompts, and calls `stable-audio` for each track.

---

## Prerequisites

### 1. HuggingFace model access

The `medium` model (default) is gated. Request access, then log in:

```bash
# request access at https://huggingface.co/stabilityai/stable-audio-3-medium
huggingface-cli login          # paste your HF token
# or export HF_TOKEN=hf_xxxxx
```

The `small-music` model is also gated:
```
https://huggingface.co/stabilityai/stable-audio-3-small-music
```

### 2. Install dependencies

```bash
cd /path/to/stable-audio-3
uv sync                       # installs stable-audio + PyTorch (CUDA 12.6 default)
uv pip install pyyaml         # required by the wrapper script
```

For a different CUDA version:
```bash
uv pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
uv sync --no-install-package torch --no-install-package torchaudio
uv pip install pyyaml
```

### 3. Verify setup

```bash
.venv/bin/stable-audio --help
python generate_from_lofi_presets.py --list
```

---

## Quick start

```bash
# Single preset, 240-second track
python generate_from_lofi_presets.py --preset ambient_lofi

# Batch of 12 random tracks (one per run, varied prompts)
python generate_from_lofi_presets.py --random --count 12 --randomize-prompts

# See what prompts would be generated (no audio produced)
python generate_from_lofi_presets.py --all --dry-run
```

---

## All available flags

| Flag | Default | Description |
|------|---------|-------------|
| `--preset NAME` | — | Generate from a single named preset |
| `--all` | — | Generate from all 37 presets |
| `--random` | — | Pick one preset at random |
| `--random-from-bank` | — | Pick prompts directly from the 200-prompt bank |
| `--list` | — | List all preset names and exit |
| `--count N` | `1` | Number of tracks per preset |
| `--duration SECS` | `240` | Track duration (medium max: 380 s) |
| `--model MODEL` | `medium` | Model: `medium`, `small-music`, `small-sfx`, … |
| `--steps N` | `8` | Diffusion steps (more = better quality, slower) |
| `--cfg-scale F` | `1.0` | Guidance scale (1.0 for distilled; 7.0 for `-base` models) |
| `--seed N` | random | Base seed; incremented per track for reproducibility |
| `--device DEVICE` | auto | `cuda`, `mps`, or `cpu` |
| `--extra-prompt TEXT` | — | Extra text appended to every prompt |
| `--negative-prompt TEXT` | (built-in) | Override the negative prompt |
| `--no-negative-prompt` | off | Disable negative prompt entirely |
| `--randomize-prompts` | off | Inject 1–2 random atmosphere phrases per track |
| `--out-dir DIR` | `output/stable_audio` | Where to write WAV files |
| `--manifest PATH` | `OUT_DIR/manifest.jsonl` | JSONL generation log |
| `--presets-dir DIR` | `../twitch-musicplayer/presets/lofi` | Preset YAML folder |
| `--stable-audio-bin PATH` | auto | Path to `stable-audio` executable |
| `--dry-run` | off | Print commands without generating |
| `--print-prompts` | off | Print prompts before generating |

Environment variable overrides:
```bash
LOFI_PRESETS_DIR=/path/to/presets   # --presets-dir
STABLE_AUDIO_BIN=/path/to/binary    # --stable-audio-bin
HF_TOKEN=hf_xxxxx                   # HuggingFace login token
```

---

## Usage examples

### List available presets

```bash
python generate_from_lofi_presets.py --list
```

### Single preset

```bash
# Default: medium model, 240 s
python generate_from_lofi_presets.py --preset sleepy_piano

# Custom duration and output folder
python generate_from_lofi_presets.py --preset vinyl_jazz \
  --duration 180 --out-dir output/jazz_session

# Shorter track on CPU-compatible model
python generate_from_lofi_presets.py --preset ambient_lofi \
  --model small-music --duration 120 --device cpu
```

### Batch of a dozen songs

```bash
# 12 tracks from random presets, each with a different atmosphere injection
python generate_from_lofi_presets.py \
  --random --count 12 \
  --randomize-prompts \
  --seed 42 \
  --duration 240 \
  --out-dir output/dozen_session

# 12 tracks from a single favourite preset (e.g. ambient_lofi)
python generate_from_lofi_presets.py \
  --preset ambient_lofi \
  --count 12 \
  --randomize-prompts \
  --seed 100 \
  --out-dir output/ambient_12
```

### All presets

```bash
# One track per preset (37 tracks total)
python generate_from_lofi_presets.py --all --duration 240

# Two tracks per preset with prompt variation
python generate_from_lofi_presets.py --all --count 2 --randomize-prompts --seed 7
```

### Prompt randomization

Adding `--randomize-prompts` injects 1–2 random atmosphere phrases (e.g.
"rain on glass", "4 AM feeling") into each prompt so no two tracks sound
identical even when sharing the same preset.

```bash
python generate_from_lofi_presets.py --preset moonlit_keys \
  --count 5 --randomize-prompts --seed 99
```

### Generate from the 200-prompt bank

Bypass preset parameters entirely and draw prompts directly from the curated
200-prompt library built into the script:

```bash
# 10 random tracks from the bank
python generate_from_lofi_presets.py --random-from-bank --count 10

# Reproducible bank run
python generate_from_lofi_presets.py --random-from-bank --count 20 --seed 2024
```

### Higher quality (more diffusion steps)

```bash
python generate_from_lofi_presets.py --preset deep_space_ambient \
  --steps 50 --duration 240
```

### Custom prompt injection

Append your own text to every generated prompt:

```bash
python generate_from_lofi_presets.py --preset rainy_window \
  --extra-prompt "subtle bird chirps, morning light through curtains"
```

### Piano-only session

```bash
python generate_from_lofi_presets.py --all \
  --extra-prompt "solo piano, no drums, no bass" \
  --out-dir output/piano_only
```

### Negative prompt override

```bash
python generate_from_lofi_presets.py --preset chillhop_cafe \
  --negative-prompt "vocals, distortion, fast tempo"

# Or disable it entirely
python generate_from_lofi_presets.py --preset tape_warmth --no-negative-prompt
```

### Dry run — preview prompts without generating

```bash
# See all prompts for every preset
python generate_from_lofi_presets.py --all --dry-run

# Preview a single preset with randomization
python generate_from_lofi_presets.py --preset felt_piano_morning \
  --count 3 --randomize-prompts --seed 0 --dry-run
```

### Custom output directory and manifest

```bash
python generate_from_lofi_presets.py --random --count 6 \
  --out-dir ~/Music/lofi_session_$(date +%Y%m%d) \
  --manifest ~/Music/lofi_session_$(date +%Y%m%d)/log.jsonl
```

### Override the binary path

```bash
# Explicit path
python generate_from_lofi_presets.py --preset ambient_lofi \
  --stable-audio-bin /home/user/stable-audio-3/.venv/bin/stable-audio

# Or via environment variable
STABLE_AUDIO_BIN=/home/user/stable-audio-3/.venv/bin/stable-audio \
  python generate_from_lofi_presets.py --preset ambient_lofi
```

### Use a different preset directory

```bash
python generate_from_lofi_presets.py \
  --presets-dir /path/to/my/custom/presets \
  --random

# Or via environment variable
LOFI_PRESETS_DIR=/path/to/my/custom/presets \
  python generate_from_lofi_presets.py --all
```

---

## 24-hour continuous generation (bash loop)

For long Twitch streaming sessions, wrap the script in a shell loop:

```bash
#!/usr/bin/env bash
# Generates tracks continuously; adjust sleep to control generation rate.
OUT_DIR="output/stream_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

while true; do
  python generate_from_lofi_presets.py \
    --random \
    --count 1 \
    --randomize-prompts \
    --duration 240 \
    --out-dir "$OUT_DIR"
  sleep 5   # brief pause between tracks
done
```

Or use the existing `run_ambient_generation_24h.sh` script from `twitch-musicplayer/scripts/`
for the MusicGen-based pipeline.

---

## JSONL manifest

Every run writes a `manifest.jsonl` (one JSON object per line) alongside the
audio files:

```json
{
  "preset": "ambient_lofi",
  "prompt": "lofi hip hop instrumental, dreamy, atmospheric, floating mood, ...",
  "negative_prompt": "fast tempo, harsh distortion, ...",
  "output": "output/stable_audio/ambient_lofi_20260605_120000.wav",
  "duration": 240.0,
  "model": "medium",
  "steps": 8,
  "cfg_scale": 1.0,
  "seed": 1234567,
  "timestamp": "2026-06-05T12:00:00+00:00",
  "success": true
}
```

Query the manifest with `jq`:

```bash
# List only successful outputs
jq 'select(.success) | .output' output/stable_audio/manifest.jsonl

# Show all prompts
jq -r '.prompt' output/stable_audio/manifest.jsonl
```

---

## Model comparison

| Model | Hardware | Max duration | Quality | Notes |
|-------|----------|-------------|---------|-------|
| `medium` | GPU (CUDA) | 380 s | Highest | Requires Flash Attention 2 for full speed |
| `small-music` | CPU / GPU | 120 s | Good | CPU-compatible, slower on CPU |
| `small-sfx` | CPU / GPU | 120 s | Good | Optimised for sound effects |
| `medium-base` | GPU | 380 s | Configurable | Use `--cfg-scale 7.0` |
| `small-music-base` | CPU / GPU | 120 s | Configurable | Use `--cfg-scale 7.0` |

---

## Troubleshooting

**`stable-audio` not found**
```bash
cd /path/to/stable-audio-3 && uv sync
# then use the full path:
python generate_from_lofi_presets.py --preset ambient_lofi \
  --stable-audio-bin /path/to/stable-audio-3/.venv/bin/stable-audio
```

**HuggingFace 403 / access restricted**
Request access at the model page (see Prerequisites), then:
```bash
huggingface-cli login
```

**`ModuleNotFoundError: yaml`**
```bash
/path/to/stable-audio-3/.venv/bin/pip install pyyaml
# or
uv pip install pyyaml
```

**Presets directory not found**
```bash
python generate_from_lofi_presets.py \
  --presets-dir /home/user/Workspace/twitch-musicplayer/presets/lofi \
  --list
```
