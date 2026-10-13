import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slurm_llm.config import ConfigError, Profile
from slurm_llm.weights import download_entry, model_path, prepare_weights, state_path, validate_gguf


class WeightTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def create(self, name, data=b"GGUF"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_all_shards_are_required(self):
        first = self.create("model-00001-of-00003.gguf")
        self.create("model-00002-of-00003.gguf")
        with self.assertRaisesRegex(ConfigError, "00003-of-00003"):
            validate_gguf(first)
        self.create("model-00003-of-00003.gguf")
        self.assertEqual(validate_gguf(first), first)

    def test_empty_shard_is_rejected(self):
        first = self.create("model-00001-of-00002.gguf")
        self.create("model-00002-of-00002.gguf", b"")
        with self.assertRaisesRegex(ConfigError, "empty"):
            validate_gguf(first)

    def test_entry_point_must_be_first_shard(self):
        for name in ("model-00002-of-00003.gguf", "model-00001-of-00000.gguf"):
            with self.subTest(name=name), self.assertRaises(ConfigError):
                validate_gguf(self.create(name))

    def test_legacy_download_is_reused_without_metadata(self):
        first = self.create("models/Q4/model-00001-of-00001.gguf")
        selected = Profile(
            "custom",
            {
                "download": {
                    "repo": "org/model",
                    "directory": "models",
                    "include": ["Q4/*.gguf"],
                    "entrypoint": "Q4/*-00001-of-*.gguf",
                }
            },
        )
        self.assertEqual(model_path(self.root, selected), first)

    def test_ambiguous_entrypoint_fails(self):
        self.create("models/one.gguf")
        self.create("models/two.gguf")
        selected = Profile(
            "custom",
            {
                "download": {
                    "repo": "org/model",
                    "directory": "models",
                    "include": ["*.gguf"],
                    "entrypoint": "*.gguf",
                }
            },
        )
        with self.assertRaisesRegex(ConfigError, "exactly one"):
            download_entry(self.root, selected)

    def test_quant_change_is_recorded_and_used_for_next_job(self):
        first = self.create("models/Q5/model-00001-of-00001.gguf")
        selected = Profile(
            "custom",
            {
                "model": {"quant": "Q4"},
                "download": {
                    "repo": "org/model",
                    "directory": "models",
                    "include": ["{quant}/*.gguf"],
                    "entrypoint": "{quant}/*-00001-of-*.gguf",
                },
            },
        )
        with (
            patch("slurm_llm.weights.hf_command", return_value=["hf"]),
            patch("slurm_llm.weights.run") as download,
        ):
            prepare_weights(self.root, selected, "Q5")
        self.assertIn("Q5/*.gguf", download.call_args.args[0])
        self.assertEqual(model_path(self.root, selected), first)
        self.assertEqual(state_path(self.root, selected).read_text().strip(), str(first))

    def test_each_include_pattern_gets_its_own_flag(self):
        self.create("models/model.gguf")
        selected = Profile(
            "custom",
            {
                "download": {
                    "repo": "org/model",
                    "directory": "models",
                    "include": ["*.json", "*.gguf"],
                    "entrypoint": "*.gguf",
                },
            },
        )
        with (
            patch("slurm_llm.weights.hf_command", return_value=["hf"]),
            patch("slurm_llm.weights.run") as download,
        ):
            prepare_weights(self.root, selected)
        command = download.call_args.args[0]
        self.assertEqual(command[-4:], ["--include", "*.json", "--include", "*.gguf"])

    def test_failed_download_does_not_replace_prepared_model(self):
        first = self.create("models/Q4/model.gguf")
        selected = Profile(
            "custom",
            {
                "model": {"quant": "Q4"},
                "download": {
                    "repo": "org/model",
                    "directory": "models",
                    "include": ["{quant}/*.gguf"],
                    "entrypoint": "{quant}/*.gguf",
                },
            },
        )
        metadata = state_path(self.root, selected)
        metadata.parent.mkdir(parents=True)
        metadata.write_text(str(first) + "\n")
        with (
            patch("slurm_llm.weights.hf_command", return_value=["hf"]),
            patch("slurm_llm.weights.run"),
            self.assertRaises(ConfigError),
        ):
            prepare_weights(self.root, selected, "Q5")
        self.assertEqual(model_path(self.root, selected), first)
