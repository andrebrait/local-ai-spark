"""Validate launch arguments without a GPU, real weights, or Docker mutations."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServeConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        cache = Path(self.temp.name)
        model = cache / 'official-nvidia-checkpoint'
        model.mkdir()
        (model / 'config.json').write_text('{"text_config":{"vocab_size":248320}}')
        snapshot = cache / 'hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/test'
        snapshot.mkdir(parents=True)
        hybrid = snapshot.with_name('test-fp8hybrid')
        hybrid.mkdir()
        (hybrid / '.prepared').touch()
        api_env = cache / 'api.env'
        api_env.write_text('VLLM_API_KEY=' + 'test-key-' * 5 + '\n')
        api_env.chmod(0o600)
        self.env = {'PATH': os.environ['PATH'], 'HOME': str(cache),
                    'HF_CACHE': str(cache), 'CACHE_HOST': str(cache / 'runtime'),
                    'MODEL_HOST': str(model), 'DRY_RUN': '1',
                    'IMAGE': 'local-ai:test', 'API_ENV_FILE': str(api_env), 'GMU': '0.75'}

    def test_auth_configuration_fails_closed(self):
        command = ['bash', str(ROOT / 'scripts/serve.sh')]
        valid = subprocess.run(command, env=self.env, capture_output=True, text=True)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertNotIn('test-key-' * 5, valid.stdout + valid.stderr)
        for content in ('WRONG_KEY_NAME=abcdef\n', 'VLLM_API_KEY=\n',
                        'VLLM_API_KEY=' + 'x' * 32 + '\nVLLM_API_KEY=\n'):
            with self.subTest(content=content):
                Path(self.env['API_ENV_FILE']).write_text(content)
                rejected = subprocess.run(command, env=self.env, capture_output=True, text=True)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertNotIn('starting on', rejected.stdout)


    def test_invalid_settings(self):
        for overrides in [{'SEQS': '0'}, {'MTP': '0'}, {'MTP': '03'},
                          {'CHUNK': 'bad'}, {'CAPTURE_SIZES': '4,0'},
                          {'CAPTURE_SIZES': '4,broken'}, {'PREFIX_CACHE': 'yes'},
                          {'BIND_HOST': '0.0.0.0'}, {'BIND_HOST': '192.168.1.2'}]:
            with self.subTest(overrides=overrides):
                result = subprocess.run(['bash', str(ROOT / 'scripts/serve.sh')],
                                        env={**self.env, **overrides},
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('run -d', result.stdout)


if __name__ == '__main__':
    unittest.main()
