#!/usr/bin/env python3
"""
Generate audio from a JSONL prompt config via the stable-audio CLI.

Prompt data lives outside this script. Each non-empty JSONL line can be either
a JSON object:

  {"name": "focus_rain_001", "prompt": "ambient ADHD music ..."}

or a JSON string:

  "ambient ADHD music for focus and work, instrumental lofi, ..."

Object records require `prompt`; `name` is optional. Per-record overrides are
supported for `duration`, `seed`, `negative_prompt`, `model`, `steps`,
`cfg_scale`, `device`, `format`, and `keep_wav`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_CONFIG = _SCRIPT_DIR / "prompts" / "adhd_focus_lofi.jsonl"


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.is_available()
    except ImportError:
        return False


_HAS_CUDA = _cuda_available()
_DEFAULT_MODEL = "medium" if _HAS_CUDA else "small-music"
_DEFAULT_DURATION = 240.0 if _HAS_CUDA else 120.0

DEFAULT_NEGATIVE_PROMPT = (
    "fast tempo, harsh distortion, heavy metal, EDM, house music, "
    "techno, screaming vocals, lead singing, aggressive percussion, noise, glitch"
)

SUPPORTED_MODELS = (
    "small-music",
    "small-sfx",
    "medium",
    "small-music-base",
    "small-sfx-base",
    "medium-base",
)
SUPPORTED_FORMATS = ("mp3", "wav")

_KNOWN_PROMPT_FIELDS = {
    "name",
    "prompt",
    "duration",
    "seed",
    "negative_prompt",
    "model",
    "steps",
    "cfg_scale",
    "device",
    "format",
    "keep_wav",
}


@dataclass(frozen=True)
class PromptRecord:
    index: int
    name: str
    prompt: str
    duration: float | None = None
    seed: int | None = None
    negative_prompt: str | None = None
    model: str | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    device: str | None = None
    audio_format: str | None = None
    keep_wav: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CliDefaults:
    duration: float
    seed_start: int
    negative_prompt: str
    disable_negative_prompt: bool
    model: str
    steps: int
    cfg_scale: float
    device: str | None
    audio_format: str
    keep_wav: bool


@dataclass(frozen=True)
class GenerationConfig:
    stable_audio_bin: str
    dry_run: bool


def _coerce_str(value: Any, field_name: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: `{field_name}` must be a non-empty string")
    return value.strip()


def _coerce_float(value: Any, field_name: str, line_number: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"line {line_number}: `{field_name}` must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number}: `{field_name}` must be a number") from exc


def _coerce_int(value: Any, field_name: str, line_number: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"line {line_number}: `{field_name}` must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number}: `{field_name}` must be an integer") from exc


def _coerce_bool(value: Any, field_name: str, line_number: int) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"line {line_number}: `{field_name}` must be true or false")
    return value


def _validate_choice(value: str, choices: tuple[str, ...], field_name: str, line_number: int) -> str:
    if value not in choices:
        joined = ", ".join(choices)
        raise ValueError(f"line {line_number}: `{field_name}` must be one of: {joined}")
    return value


def _prompt_record_from_object(record: dict[str, Any], index: int, line_number: int) -> PromptRecord:
    if "prompt" not in record:
        raise ValueError(f"line {line_number}: JSON object records require `prompt`")

    name = (
        _coerce_str(record["name"], "name", line_number)
        if "name" in record and record["name"] is not None
        else f"prompt_{index:03d}"
    )
    prompt = _coerce_str(record["prompt"], "prompt", line_number)
    audio_format = None
    if record.get("format") is not None:
        audio_format = _validate_choice(
            _coerce_str(record["format"], "format", line_number),
            SUPPORTED_FORMATS,
            "format",
            line_number,
        )
    model = None
    if record.get("model") is not None:
        model = _validate_choice(
            _coerce_str(record["model"], "model", line_number),
            SUPPORTED_MODELS,
            "model",
            line_number,
        )

    metadata = {k: v for k, v in record.items() if k not in _KNOWN_PROMPT_FIELDS}

    return PromptRecord(
        index=index,
        name=name,
        prompt=prompt,
        duration=(
            _coerce_float(record["duration"], "duration", line_number)
            if record.get("duration") is not None else None
        ),
        seed=(
            _coerce_int(record["seed"], "seed", line_number)
            if record.get("seed") is not None else None
        ),
        negative_prompt=(
            str(record["negative_prompt"])
            if record.get("negative_prompt") is not None else None
        ),
        model=model,
        steps=(
            _coerce_int(record["steps"], "steps", line_number)
            if record.get("steps") is not None else None
        ),
        cfg_scale=(
            _coerce_float(record["cfg_scale"], "cfg_scale", line_number)
            if record.get("cfg_scale") is not None else None
        ),
        device=(
            _coerce_str(record["device"], "device", line_number)
            if record.get("device") is not None else None
        ),
        audio_format=audio_format,
        keep_wav=(
            _coerce_bool(record["keep_wav"], "keep_wav", line_number)
            if record.get("keep_wav") is not None else None
        ),
        metadata=metadata,
    )


def load_prompt_records(config_path: Path) -> list[PromptRecord]:
    if not config_path.is_file():
        raise FileNotFoundError(f"prompts config not found: {config_path}")

    records: list[PromptRecord] = []
    with config_path.open(encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc

            index = len(records) + 1
            if isinstance(parsed, str):
                prompt = _coerce_str(parsed, "prompt", line_number)
                records.append(PromptRecord(index=index, name=f"prompt_{index:03d}", prompt=prompt))
            elif isinstance(parsed, dict):
                records.append(_prompt_record_from_object(parsed, index, line_number))
            else:
                raise ValueError(f"line {line_number}: expected a JSON object or JSON string")

    if not records:
        raise ValueError(f"prompts config contains no prompt records: {config_path}")
    return records


def _find_stable_audio(override: str | None) -> str | None:
    if override:
        return override
    local = _SCRIPT_DIR / ".venv" / "bin" / "stable-audio"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    return shutil.which("stable-audio")


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower())
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return stem or "prompt"


def _run_generation(
    record: PromptRecord,
    output_path: Path,
    duration: float,
    seed: int,
    negative_prompt: str,
    model: str,
    steps: int,
    cfg_scale: float,
    device: str | None,
    config: GenerationConfig,
) -> bool:
    cmd = [
        config.stable_audio_bin,
        "--model", model,
        "-p", record.prompt,
        "--duration", str(duration),
        "--steps", str(steps),
        "--cfg-scale", str(cfg_scale),
        "--seed", str(seed),
        "-o", str(output_path),
    ]
    if negative_prompt:
        cmd += ["--negative-prompt", negative_prompt]
    if device:
        cmd += ["--device", device]

    if config.dry_run:
        print(f"[dry-run] {config.stable_audio_bin} --model {model} \\")
        print(f"  -p {record.prompt!r} \\")
        if negative_prompt:
            print(f"  --negative-prompt {negative_prompt!r} \\")
        print(
            f"  --duration {duration} --steps {steps} --cfg-scale {cfg_scale} "
            f"--seed {seed} -o {output_path}"
        )
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TRANSFORMERS_VERBOSITY": "warning"}
    result = subprocess.run(cmd, text=True, check=False, env=env)
    return result.returncode == 0


def _wav_to_mp3(wav_path: Path, mp3_path: Path, keep_wav: bool = False) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-q:a", "2", str(mp3_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}", file=sys.stderr)
        return False
    if not keep_wav:
        wav_path.unlink(missing_ok=True)
    return True


def _manifest_entry(
    record: PromptRecord,
    prompts_config: Path,
    out_file: Path,
    success: bool,
    duration: float,
    seed: int,
    negative_prompt: str,
    model: str,
    steps: int,
    cfg_scale: float,
    device: str | None,
    audio_format: str,
) -> dict[str, Any]:
    return {
        "index": record.index,
        "name": record.name,
        "prompt": record.prompt,
        "negative_prompt": negative_prompt,
        "output": str(out_file),
        "duration": duration,
        "seed": seed,
        "model": model,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "device": device,
        "format": audio_format,
        "prompts_config": str(prompts_config),
        "metadata": record.metadata,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": success,
    }


def _generate_from_config(
    records: list[PromptRecord],
    prompts_config: Path,
    out_dir: Path,
    manifest_path: Path,
    defaults: CliDefaults,
    gen: GenerationConfig,
    print_prompts: bool,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(records)

    for position, record in enumerate(records, start=1):
        duration = record.duration if record.duration is not None else defaults.duration
        seed = record.seed if record.seed is not None else defaults.seed_start + position - 1
        negative_prompt = "" if defaults.disable_negative_prompt else (
            record.negative_prompt
            if record.negative_prompt is not None
            else defaults.negative_prompt
        )
        model = record.model or defaults.model
        steps = record.steps if record.steps is not None else defaults.steps
        cfg_scale = record.cfg_scale if record.cfg_scale is not None else defaults.cfg_scale
        device = record.device if record.device is not None else defaults.device
        audio_format = record.audio_format or defaults.audio_format
        keep_wav = record.keep_wav if record.keep_wav is not None else defaults.keep_wav

        stem = f"{position:03d}_{_safe_stem(record.name)}_{timestamp}"
        wav_file = out_dir / f"{stem}.wav"
        out_file = wav_file if audio_format == "wav" else out_dir / f"{stem}.{audio_format}"

        if print_prompts:
            print(f"\n[{position}/{total}] {record.name}")
            print(f"  prompt: {record.prompt}")
            if negative_prompt:
                print(f"  negative: {negative_prompt}")

        if not gen.dry_run:
            print(f"[{position}/{total}] {record.name}  seed={seed}  duration={duration}s")

        ok = _run_generation(
            record=record,
            output_path=wav_file,
            duration=duration,
            seed=seed,
            negative_prompt=negative_prompt,
            model=model,
            steps=steps,
            cfg_scale=cfg_scale,
            device=device,
            config=gen,
        )
        if ok and audio_format == "mp3" and not gen.dry_run:
            ok = _wav_to_mp3(wav_file, out_file, keep_wav=keep_wav)

        entries.append(
            _manifest_entry(
                record=record,
                prompts_config=prompts_config,
                out_file=out_file,
                success=ok,
                duration=duration,
                seed=seed,
                negative_prompt=negative_prompt,
                model=model,
                steps=steps,
                cfg_scale=cfg_scale,
                device=device,
                audio_format=audio_format,
            )
        )

    if not gen.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_from_lofi_presets.py",
        description="Generate one track per prompt from a JSONL prompts config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "JSONL examples:\n"
            '  {"name": "focus_rain_001", "prompt": "ambient ADHD music ..."}\n'
            '  "ambient ADHD music for focus and work, instrumental lofi, ..."\n\n'
            "Environment variables:\n"
            "  LOFI_PROMPTS_CONFIG  Override the default prompts config path\n"
            "  STABLE_AUDIO_BIN     Override --stable-audio-bin\n"
            "  HF_TOKEN             HuggingFace token for gated model access\n"
        ),
    )
    parser.add_argument(
        "prompts_config",
        nargs="?",
        default=os.environ.get("LOFI_PROMPTS_CONFIG", str(DEFAULT_PROMPTS_CONFIG)),
        help=f"JSONL prompts config (default: {DEFAULT_PROMPTS_CONFIG})",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=_DEFAULT_DURATION,
        metavar="SECS",
        help=(
            f"Default track duration in seconds (default: {_DEFAULT_DURATION:.0f}; "
            "medium max: 380, small-music max: 120)"
        ),
    )
    parser.add_argument(
        "--model",
        "-m",
        default=_DEFAULT_MODEL,
        choices=SUPPORTED_MODELS,
        help=f"Default stable-audio model (default: {_DEFAULT_MODEL}; auto-selected by CUDA)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        metavar="N",
        help="Default diffusion steps (default: 8; try 20-50 for higher quality)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        metavar="F",
        help="Default CFG scale (default: 1.0; use 7.0 for *-base models)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="Base seed, incremented once per prompt unless a record sets seed (default: 0)",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Torch device: cuda / mps / cpu (auto-detected by stable-audio by default)",
    )
    parser.add_argument(
        "--format",
        default="mp3",
        choices=SUPPORTED_FORMATS,
        help="Default output format (default: mp3; converted via ffmpeg after generation)",
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="Keep intermediate WAV files when output format is mp3",
    )
    parser.add_argument(
        "--negative-prompt",
        metavar="TEXT",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Default negative prompt",
    )
    parser.add_argument(
        "--no-negative-prompt",
        action="store_true",
        help="Disable default negative prompts and ignore per-record negative_prompt values",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default="output/stable_audio",
        metavar="DIR",
        help="Output directory (default: output/stable_audio)",
    )
    parser.add_argument(
        "--manifest",
        metavar="PATH",
        help="JSONL manifest path (default: OUT_DIR/manifest.jsonl)",
    )
    parser.add_argument(
        "--stable-audio-bin",
        metavar="PATH",
        default=os.environ.get("STABLE_AUDIO_BIN"),
        help="Path to stable-audio executable",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts and stable-audio commands without generating",
    )
    parser.add_argument(
        "--print-prompts",
        action="store_true",
        help="Print each resolved prompt before generation",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    prompts_config = Path(args.prompts_config)
    try:
        records = load_prompt_records(prompts_config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    stable_audio = _find_stable_audio(args.stable_audio_bin)
    if stable_audio is None and not args.dry_run:
        print(
            "Error: stable-audio executable not found. Run 'uv sync' in stable-audio-3/.\n"
            "  Also ensure HuggingFace access and login: huggingface-cli login",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.jsonl"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    defaults = CliDefaults(
        duration=args.duration,
        seed_start=args.seed,
        negative_prompt=args.negative_prompt,
        disable_negative_prompt=args.no_negative_prompt,
        model=args.model,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        device=args.device,
        audio_format=args.format,
        keep_wav=args.keep_wav,
    )
    gen = GenerationConfig(
        stable_audio_bin=stable_audio or "stable-audio",
        dry_run=args.dry_run,
    )

    entries = _generate_from_config(
        records=records,
        prompts_config=prompts_config,
        out_dir=out_dir,
        manifest_path=manifest_path,
        defaults=defaults,
        gen=gen,
        print_prompts=args.print_prompts or args.dry_run,
    )

    success_count = sum(1 for entry in entries if entry["success"])
    if args.dry_run:
        print(f"\n[dry-run] Would generate {len(entries)} track(s) to {out_dir}/")
        return

    print(f"\nDone: {success_count}/{len(entries)} tracks generated.")
    print(f"Output: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
