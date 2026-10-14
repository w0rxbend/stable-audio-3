# Lofi Prompt Generation

Generate music from a JSONL prompt config using the upstream TFLite CPU runtime
or the Stability AI `stable-audio` PyTorch CLI. Prompt data is stored outside
Python; the script reads the config and
generates one track per prompt, in file order.

The defaults are now **Medium on CPU, five minutes per track**, using the
new [Afterhours collection of 144 prompts](prompts/AFTERHOURS.md). Set up the
CPU runtime and start the batch from the repository root:

```bash
(cd optimized/tflite && ./install.sh --download medium)
bash scripts/generate_afterhours_cpu.sh
```

For a short audition, append `--limit 1 --duration 30` to the batch command.
The CPU runtime has its own environment and needs no PyTorch/CUDA installation.
MP3 output requires `ffmpeg`; use `--format wav` to avoid conversion.

## Prompt Config

Default config:

```bash
prompts/afterhours_144.jsonl
```

Additional configs:

```bash
prompts/adhd_focus_lofi.jsonl
prompts/chillhop_rhythmic_lofi.jsonl
prompts/ambient_calm_sad_sleep.jsonl
prompts/celtic_ritual_chants.jsonl
prompts/lofi_ambient_90s_grunge.jsonl
prompts/lofi_ambient_90s_grunge_sleepy_rain.jsonl
prompts/lofi_ambient_city_night_study.jsonl
prompts/lofi_ambient_morning_study_light.jsonl
prompts/lofi_ambient_ocean_haze.jsonl
prompts/instrumental_calm_guitar.jsonl
prompts/grunge_guitar_lofi.jsonl
```

Each non-empty line is one prompt record. The simplest record is:

```json
{"name": "focus_rain_001", "prompt": "ambient ADHD music for focus and work, instrumental lofi, soft rain, no vocals"}
```

You can also use a raw JSON string:

```json
"ambient ADHD music for focus and work, instrumental lofi, soft rain, no vocals"
```

Required object fields:

| Field | Description |
| --- | --- |
| `prompt` | Stable Audio text prompt |

Optional object fields:

| Field | Description |
| --- | --- |
| `name` | Output/manifest name; defaults to `prompt_001`, `prompt_002`, etc. |
| `display_name` | Optional human-readable title used for display logging; defaults to `name`. Output filenames use `name`. |
| `duration` | Per-track duration override |
| `seed` | Per-track seed override |
| `negative_prompt` | Per-track negative prompt override |
| `model` | Per-track model override |
| `steps` | Per-track diffusion steps override |
| `cfg_scale` | Per-track CFG scale override |
| `device` | Per-track device override, such as `cuda`, `mps`, or `cpu` |
| `format` | Per-track output format: `mp3` or `wav` |
| `keep_wav` | Keep intermediate WAV when `format` is `mp3` |

Extra JSON fields are preserved in the manifest under `metadata`.

## Run

Preview every generated command without creating audio:

```bash
python generate_from_lofi_presets.py prompts/adhd_focus_lofi.jsonl --dry-run
```

Generate the default prompt config:

```bash
python generate_from_lofi_presets.py
```

Generate from a custom prompt config:

```bash
python generate_from_lofi_presets.py path/to/prompts.jsonl
```

Generate the chillhop / rhythmic / lofi-trap / lofi-rock instrumental config:

```bash
python generate_from_lofi_presets.py prompts/chillhop_rhythmic_lofi.jsonl
```

Generate the ambient / calm / sad sleep music config:

```bash
python generate_from_lofi_presets.py prompts/ambient_calm_sad_sleep.jsonl
```

Generate the Celtic ritual chants config:

```bash
python generate_from_lofi_presets.py prompts/celtic_ritual_chants.jsonl
```

Generate the lo-fi ambient calm 90s grunge configs:

```bash
python generate_from_lofi_presets.py prompts/lofi_ambient_90s_grunge.jsonl
python generate_from_lofi_presets.py prompts/lofi_ambient_90s_grunge_sleepy_rain.jsonl
```

Generate the lo-fi ambient city night study config:

```bash
python generate_from_lofi_presets.py prompts/lofi_ambient_city_night_study.jsonl
```

Generate the lo-fi ambient morning study config:

```bash
python generate_from_lofi_presets.py prompts/lofi_ambient_morning_study_light.jsonl
```

Generate the lo-fi ambient ocean haze config:

```bash
python generate_from_lofi_presets.py prompts/lofi_ambient_ocean_haze.jsonl
```

Generate the instrumental calm guitar config:

```bash
python generate_from_lofi_presets.py prompts/instrumental_calm_guitar.jsonl
```

Generate the grunge guitar lofi config:

```bash
python generate_from_lofi_presets.py prompts/grunge_guitar_lofi.jsonl
```

Use a custom output directory:

```bash
python generate_from_lofi_presets.py prompts/adhd_focus_lofi.jsonl \
  --out-dir output/adhd_focus_lofi
```

Use the PyTorch backend on CUDA (requires the root project and Flash Attention):

```bash
python generate_from_lofi_presets.py prompts/adhd_focus_lofi.jsonl \
  --backend pytorch \
  --model medium \
  --device cuda \
  --duration 240 \
  --steps 8
```

Generate WAV files instead of MP3:

```bash
python generate_from_lofi_presets.py prompts/adhd_focus_lofi.jsonl --format wav
```

## CLI Options

| Option | Default | Description |
| --- | --- | --- |
| `prompts_config` | `prompts/afterhours_144.jsonl` | JSONL prompt config path |
| `--backend pytorch\|tflite` | `tflite` | Inference runtime; TFLite supports Medium on CPU |
| `--duration SECS` | `300` | Default duration; Medium max 380, Small max 120 |
| `--model MODEL` | `medium` | Default Stable Audio model |
| `--threads N` | `8` | TFLite CPU threads |
| `--decoder same-s\|same-l` | `same-s` | TFLite decoder; SAME-L is larger |
| `--dit-precision w8a8\|fp32` | `w8a8` | TFLite DiT precision |
| `--steps N` | `8` | Default diffusion steps |
| `--cfg-scale F` | `1.0` | Default CFG scale |
| `--seed N` | `0` | Base seed, incremented once per prompt unless a record sets `seed` |
| `--device DEVICE` | auto | Torch device |
| `--format mp3\|wav` | `mp3` | Default output format |
| `--keep-wav` | off | Keep intermediate WAV files for MP3 outputs |
| `--negative-prompt TEXT` | built-in | Default negative prompt |
| `--no-negative-prompt` | off | Disable default and per-record negative prompts |
| `--out-dir DIR` | `output/stable_audio` | Output directory |
| `--manifest PATH` | `OUT_DIR/manifest.jsonl` | Manifest path |
| `--stable-audio-bin PATH` | auto | Path to runtime executable (`stable-audio` or TFLite `sa3`) |
| `--limit N` | all | Process only the first N records |
| `--dry-run` | off | Print commands without generating audio |
| `--print-prompts` | off | Print prompts before generating |

## Environment

```bash
LOFI_PROMPTS_CONFIG=/path/to/prompts.jsonl
STABLE_AUDIO_BIN=/path/to/stable-audio
HF_TOKEN=...
```

## Output

Generated files are named with their config order and prompt `name`:

```text
output/stable_audio/ADHD Focus Lofi #001 - rain-muted attention bed.mp3
```

The manifest is JSONL and records the resolved prompt, seed, model, duration,
output path, config path, metadata, timestamp, and success flag for each track.
TFLite manifests also record the backend, CPU thread count, decoder, and DiT
precision. Invalid durations and unsupported runtime settings fail before the
batch starts; a batch with failed tracks exits with status 1.
