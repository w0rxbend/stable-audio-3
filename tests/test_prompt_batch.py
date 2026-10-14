"""Offline checks for batch CLI translation, output paths, and preset diversity."""

import contextlib
import io
import json
import shlex
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import generate_from_lofi_presets as batch


class PromptBatchTests(unittest.TestCase):
    def invoke(self, *args):
        output = io.StringIO()
        with patch.object(sys, "argv", ["generate_from_lofi_presets.py", *args]):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                batch.main()
        return output.getvalue()

    def test_cpu_defaults_and_limit(self):
        output = self.invoke("--dry-run", "--limit", "1")
        command = shlex.split(
            next(
                line for line in output.splitlines() if line.startswith("[dry-run] /")
            )[10:]
        )
        self.assertEqual(command[0], str(batch.TFLITE_BIN))
        self.assertEqual(command[command.index("--dit") + 1], "medium")
        self.assertEqual(command[command.index("--seconds") + 1], "300.0")
        self.assertEqual(command[command.index("--threads") + 1], "8")
        self.assertTrue(Path(command[command.index("--out") + 1]).is_absolute())
        self.assertNotIn("--device", command)
        self.assertIn("Would generate 1 track(s)", output)

    def test_all_prompts_unique_and_balanced(self):
        records = batch.load_prompt_records(batch.DEFAULT_PROMPTS_CONFIG)
        self.assertEqual(len(records), 144)
        for field in ("name", "display_name", "prompt"):
            self.assertEqual(len({getattr(record, field) for record in records}), 144)
        self.assertEqual(
            sorted(Counter(record.metadata["genre"] for record in records).values()),
            [12] * 12,
        )
        existing = {
            record.prompt
            for path in batch.DEFAULT_PROMPTS_CONFIG.parent.glob("*.jsonl")
            if path != batch.DEFAULT_PROMPTS_CONFIG
            for record in batch.load_prompt_records(path)
        }
        self.assertFalse(existing & {record.prompt for record in records})
        for record in records:
            self.assertIn("VocalType: Instrumental", record.prompt)
            self.assertIn(f"{record.metadata['bpm']} BPM", record.prompt)
            self.assertIsNotNone(record.negative_prompt)
            self.assertNotIn("harsh distortion", record.negative_prompt)

    def test_full_collection_dry_run(self):
        output = self.invoke("--dry-run")
        self.assertEqual(output.count("[dry-run] /"), 144)
        self.assertIn("--seed 143", output)
        self.assertIn("Would generate 144 track(s)", output)

    def test_small_model_translation(self):
        for model, dit in (("small-music", "sm-music"), ("small-sfx", "sm-sfx")):
            with self.subTest(model=model):
                output = self.invoke(
                    "--dry-run", "--limit", "1", "--model", model, "--duration", "120"
                )
                self.assertIn(f"--dit {dit}", output)
                self.assertIn("--seconds 120.0", output)

    def test_pytorch_translation(self):
        output = self.invoke(
            "--dry-run",
            "--limit",
            "1",
            "--backend",
            "pytorch",
            "--model",
            "small-music",
            "--duration",
            "120",
            "--device",
            "cpu",
        )
        self.assertIn("--model small-music", output)
        self.assertIn("--duration 120.0", output)
        self.assertIn("--cfg-scale 1.0", output)
        self.assertIn("--device cpu", output)
        self.assertNotIn("--dit ", output)

    def test_duration_limits_and_invalid_runtime_options(self):
        cases = (
            ("--duration", "381"),
            ("--duration", "0"),
            ("--duration", "nan"),
            ("--duration", "inf"),
            ("--model", "small-music", "--duration", "121"),
            ("--model", "medium-base"),
            ("--device", "cuda"),
            ("--threads", "0"),
            ("--limit", "0"),
        )
        for args in cases:
            with self.subTest(args=args), self.assertRaises(SystemExit) as exc:
                self.invoke("--dry-run", "--limit", "1", *args)
            self.assertEqual(exc.exception.code, 2)
        self.assertIn(
            "--seconds 380.0",
            self.invoke("--dry-run", "--limit", "1", "--duration", "380"),
        )

    def test_record_overrides_and_manifest_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="sa3 batch ") as temp:
            directory = Path(temp)
            config = directory / "prompts.jsonl"
            config.write_text(
                json.dumps(
                    {
                        "name": "Test 'title'",
                        "prompt": "guitar 'motif'",
                        "duration": 45,
                        "seed": 1234,
                        "steps": 6,
                        "cfg_scale": 1.5,
                        "negative_prompt": "vocals",
                        "format": "wav",
                        "genre": "grunge",
                    }
                )
                + "\n"
            )
            with patch.object(
                batch.subprocess, "run", return_value=SimpleNamespace(returncode=0)
            ) as run:
                self.invoke(
                    str(config),
                    "--stable-audio-bin",
                    "/custom/sa3",
                    "--out-dir",
                    str(directory / "output"),
                    "--decoder",
                    "same-l",
                    "--dit-precision",
                    "fp32",
                )
            command = run.call_args.args[0]
            for flag, value in (
                ("--seconds", "45.0"),
                ("--seed", "1234"),
                ("--steps", "6"),
                ("--cfg", "1.5"),
                ("--decoder", "same-l"),
                ("--dit-precision", "fp32"),
                ("--negative-prompt", "vocals"),
            ):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertEqual(command[command.index("--prompt") + 1], "guitar 'motif'")
            self.assertTrue(Path(command[command.index("--out") + 1]).is_absolute())
            entry = json.loads((directory / "output" / "manifest.jsonl").read_text())
            self.assertEqual(entry["device"], "cpu")
            self.assertEqual(entry["backend"], "tflite")
            self.assertEqual(entry["decoder"], "same-l")
            self.assertEqual(entry["dit_precision"], "fp32")
            self.assertEqual(entry["metadata"], {"genre": "grunge"})
            self.assertTrue(entry["success"])

    def test_failed_generation_exits_nonzero_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(
                batch.subprocess, "run", return_value=SimpleNamespace(returncode=1)
            ):
                with self.assertRaises(SystemExit) as exc:
                    self.invoke(
                        "--limit",
                        "1",
                        "--format",
                        "wav",
                        "--stable-audio-bin",
                        "/custom/sa3",
                        "--out-dir",
                        temp,
                    )
            self.assertEqual(exc.exception.code, 1)
            self.assertFalse(
                json.loads((Path(temp) / "manifest.jsonl").read_text())["success"]
            )

    def test_disable_negative_prompts(self):
        output = self.invoke("--dry-run", "--limit", "1", "--no-negative-prompt")
        self.assertNotIn("--negative-prompt", output)


if __name__ == "__main__":
    unittest.main()
