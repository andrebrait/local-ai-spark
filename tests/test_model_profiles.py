import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ModelProfileTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name)
        self.log = directory / "docker.jsonl"
        docker = directory / "docker"
        docker.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['DOCKER_LOG'], 'a') as stream:\n"
            "    stream.write(json.dumps(args) + '\\n')\n"
            "if args[:1] == ['inspect']:\n"
            "    name = args[-1]\n"
            "    suffix = name.upper().replace('-', '_')\n"
            "    if os.environ.get('MISSING_' + suffix) == 'true':\n"
            "        raise SystemExit(1)\n"
            "    print(os.environ.get('RUNNING_' + suffix, 'false'))\n"
            "elif args[:1] == ['create']:\n"
            "    print('fake-container-id')\n"
        )
        docker.chmod(0o755)
        self.env = {
            "PATH": f"{directory}:{os.environ['PATH']}",
            "DOCKER_LOG": str(self.log),
        }

    def run_script(self, name, *args, **env):
        return subprocess.run(
            ["bash", str(ROOT / "scripts" / name), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
        )

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_nvidia_profile_creates_the_verified_single_gpu_server(self):
        result = self.run_script("create-27b.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.calls()[0]
        self.assertEqual(args[:3], ["create", "--name", "qwen38-27b"])
        self.assertIn("unless-stopped", args)
        self.assertIn("/home/andre/local-ai/models/Qwen3.8-27B-NVFP4:/models/qwen38-27b:ro", args)
        self.assertIn("sha256:05eb4719754d1390b2b577a761eb9e53cc8413c17f7863511633e9fba45102c8", args)
        self.assertEqual(args[args.index("--served-model-name") + 1], "qwen3.8-27b")
        self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "1")
        self.assertEqual(args[args.index("--max-model-len") + 1], "262144")
        self.assertEqual(args[args.index("--max-num-seqs") + 1], "32")
        self.assertEqual(args[args.index("--kv-cache-dtype") + 1], "fp8_e4m3")
        self.assertEqual(
            json.loads(args[args.index("--speculative-config") + 1]),
            {"method": "mtp", "num_speculative_tokens": 3},
        )
        self.assertIn("--no-enable-flashinfer-autotune", args)

    def test_swift_profile_uses_its_checkpoint_and_model_id(self):
        result = self.run_script("create-swift-27b.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.calls()[0]
        self.assertEqual(args[:3], ["create", "--name", "swift-qwen38-27b"])
        self.assertIn("/home/andre/local-ai/models/Swift-Qwen3.8-27B-NVFP4:/models/swift-qwen38-27b:ro", args)
        self.assertIn("sha256:05eb4719754d1390b2b577a761eb9e53cc8413c17f7863511633e9fba45102c8", args)
        self.assertEqual(args[args.index("--served-model-name") + 1], "swift-qwen3.8-27b")
        self.assertEqual(args[args.index("--kv-cache-dtype") + 1], "fp8_e4m3")
        self.assertEqual(
            json.loads(args[args.index("--speculative-config") + 1]),
            {"method": "mtp", "num_speculative_tokens": 3},
        )

    def test_selector_stops_the_other_model_before_starting_target(self):
        result = self.run_script(
            "local-ai-model",
            "nvidia",
            RUNNING_SWIFT_QWEN38_27B="true",
            RUNNING_QWEN38_27B="false",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        stop = ["stop", "swift-qwen38-27b"]
        start = ["start", "qwen38-27b"]
        self.assertIn(stop, calls)
        self.assertIn(start, calls)
        self.assertLess(calls.index(stop), calls.index(start))

    def test_selector_reports_a_missing_target_container(self):
        result = self.run_script(
            "local-ai-model",
            "nvidia",
            RUNNING_SWIFT_QWEN38_27B="false",
            MISSING_QWEN38_27B="true",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("container qwen38-27b not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
