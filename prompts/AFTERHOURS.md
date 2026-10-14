# Afterhours: 144 Windows into the Night

[afterhours_144.jsonl](afterhours_144.jsonl) contains 144 individually written
instrumental prompts. Each track has its own title, musical motif, groove,
instrumentation, production texture, and arrangement. Tempo and key are included
in both the prompt and metadata. These are creative directions, not guarantees
that every generated recording will follow the requested key, tempo, or form.

| Tracks | Style | Example title |
| --- | --- | --- |
| 001–012 | Dusty lo-fi hip hop | Receipt in a Winter Coat |
| 013–024 | Jazz noir chillhop | Hotel Key Number Zero |
| 025–036 | Doomer post-punk | Concrete Balcony at 4 AM |
| 037–048 | Coldwave and darkwave | The Clock Is Six Minutes Wrong |
| 049–060 | Cassette grunge rock | One Good Cable Left |
| 061–072 | Grunge lo-fi beat hybrids | Feedback Folded into Origami |
| 073–084 | Shoegaze and dream grunge | The Sky Has Tape Damage |
| 085–096 | Slowcore bedroom rock | Everything Packed except the Lamp |
| 097–108 | Trip-hop and noir downtempo | Room Service for a Ghost |
| 109–120 | Dub and lo-fi bass music | Moss between the Subwoofers |
| 121–132 | Tape ambient and vapor beats | Weather Channel for Another Planet |
| 133–144 | Experimental lo-fi and folk-electronic beats | Seven Steps to the Attic |

Negative prompts differ by style so that grunge fuzz, dark synths, and dusty
drums are allowed. The positive prompt also specifies instrumental music.
At CFG 1.0, guidance is off and negative prompts have no effect; try
`--cfg-scale 1.5` if you want to experiment with negative guidance. That adds a
second inference pass per sampling step on the CPU backend.

## Local CPU setup

Run from the repository root:

```bash
(cd optimized/tflite && ./install.sh --download medium)
```

The upstream installer uses `uv` to create `optimized/tflite/.venv`, installs
LiteRT, NumPy, SentencePiece, SoundFile, and Hugging Face Hub, and downloads the
Medium bundle. It manages Python 3.11 by default. CUDA, Flash Attention, and
PyTorch are unnecessary for this runtime. Install `ffmpeg` for MP3 output, or
pass `--format wav` to keep WAV only. No Gradio extras are needed for the batch.

The inspected workstation has a Ryzen 7 5700X (8 physical cores), about 60 GiB
RAM, and a Polaris-family AMD Radeon. Use the CPU runtime with 8 threads.
There is no compatible NVIDIA CUDA device detected. The batch has not been
benchmarked on this machine, and generating 144 five-minute tracks may take
substantial time. It produces 12 hours of audio.

## Generate

First audition one 30-second track:

```bash
bash scripts/generate_afterhours_cpu.sh --limit 1 --duration 30
```

Generate all 144 tracks at five minutes each:

```bash
bash scripts/generate_afterhours_cpu.sh
```

MP3s and `manifest.jsonl` are written to `output/afterhours_144/`. Each track
receives a distinct sequential seed, 0 through 143 by default. Repeating a
successful run uses a new filename if audio already exists, but uses the same
seeds unless you pass a different `--seed`. The manifest for the output directory
is replaced on each run; choose a new `--out-dir` to retain separate manifests.
The script launches a fresh runtime process for each track, so model loading
time is paid per track. A failure is recorded and causes a nonzero batch exit.

The equivalent explicit command is:

```bash
python3 generate_from_lofi_presets.py prompts/afterhours_144.jsonl \
  --backend tflite --model medium --duration 300 --threads 8 \
  --steps 8 --cfg-scale 1.0 --dit-precision w8a8 --decoder same-s \
  --format mp3 --out-dir output/afterhours_144
```

The general batch script now defaults to this collection, Medium, TFLite, and
300 seconds. The shell launcher additionally chooses the dedicated output
directory. Preview without installing dependencies or downloading weights:

```bash
bash scripts/generate_afterhours_cpu.sh --dry-run
```

## Duration and quality options

- Medium: up to 380 seconds (6:20). Use `--duration 300` for five minutes.
- Small-Music: up to 120 seconds. Use `--model small-music --duration 120`.
- Songs longer than 6:20: generate multiple sections and assemble them in an
  audio editor with transitions or crossfades. Longer output settings do not
  increase the model's supported musical context.
- `--decoder same-l`: use the larger codec for Medium at greater CPU cost.
  `same-s` is the upstream CPU default, including for Medium.
- `--dit-precision fp32`: compare with the default smaller `w8a8` model.
  Ryzen 5700X lacks VNNI/AMX int8 acceleration, so benchmark both precisions
  with a short audition rather than assuming int8 is faster. FP32 weights
  download separately on first use.
- `--seed 1000`: use a new series of seeds for a different batch.
- `--format wav --out-dir output/afterhours_wav`: keep WAV recordings.
- For the original PyTorch runtime, explicitly pass `--backend pytorch` and
  install the root project dependencies separately. Use Small-Music for CPU;
  the documented PyTorch Medium route requires CUDA and Flash Attention.

All records omit duration, model, and seed overrides so batch flags apply to
every track. Their titles, genres, BPM, and keys are preserved in the manifest.
The [upstream CPU runtime guide](../optimized/tflite/README.md) documents its
other options and installation details.
