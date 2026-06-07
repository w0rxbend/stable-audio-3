# Lofi Prompt Generation

Generate music from a JSONL prompt config using the Stability AI `stable-audio`
CLI. Prompt data is stored outside Python; the script reads the config and
generates one track per prompt, in file order.

## Prompt Config

Default config:

```bash
prompts/adhd_focus_lofi.jsonl
```

Additional configs:

```bash
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
| `display_name` | Optional human-readable title used for output filenames and display logging; defaults to `name`. |
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

Higher quality on CUDA:

```bash
python generate_from_lofi_presets.py prompts/adhd_focus_lofi.jsonl \
  --model medium \
  --device cuda \
  --duration 240 \
  --steps 30
```

Generate WAV files instead of MP3:

```bash
python generate_from_lofi_presets.py prompts/adhd_focus_lofi.jsonl --format wav
```

## CLI Options

| Option | Default | Description |
| --- | --- | --- |
| `prompts_config` | `prompts/adhd_focus_lofi.jsonl` | JSONL prompt config path |
| `--duration SECS` | `120` CPU, `240` CUDA | Default duration for records without `duration` |
| `--model MODEL` | auto | Default Stable Audio model |
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
| `--stable-audio-bin PATH` | auto | Path to `stable-audio` |
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
