"""Exercise production argument/auth refusal without Docker or GPU execution."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:213e470219da6e21119f0a13a4df35f7e0d1f18ed2fb26e43f68c58965b30dfe'


class ServeConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name)
        model = directory / 'checkpoint'
        model.mkdir()
        (model / 'config.json').write_text('{"text_config":{"vocab_size":248320}}')
        self.api_env = directory / 'api.env'
        self.secret = 'test-key-' * 5
        self.api_env.write_text('VLLM_API_KEY=' + self.secret + '\n')
        self.api_env.chmod(0o600)
        self.env = {'PATH': os.environ['PATH'], 'HOME': str(directory),
                    'CACHE_HOST': str(directory / 'cache'), 'MODEL_HOST': str(model),
                    'DRY_RUN': '1', 'API_ENV_FILE': str(self.api_env)}

    def launch(self, **overrides):
        return subprocess.run(['bash', str(ROOT / 'scripts/serve.sh')],
                              env={**self.env, **overrides}, capture_output=True, text=True)

    def test_bounded_authenticated_no_restart_recipe(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(self.secret, result.stdout + result.stderr)
        args = shlex.split(result.stdout)
        self.assertEqual(args[:2], ['docker', 'create'])
        self.assertIn(IMAGE, args)
        for key, value in {'--restart': 'no', '--max-num-seqs': '2',
                           '--max-model-len': '262144', '--kv-cache-memory-bytes': '5368709120',
                           '--env-file': str(self.api_env), '--kv-cache-dtype': 'fp8_e4m3',
                           '--mamba-cache-mode': 'align'}.items():
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertIn('--enable-prompt-tokens-details', args)
        self.assertEqual(json.loads(args[args.index('--compilation-config') + 1])['cudagraph_capture_sizes'], [4, 8])

    def test_auth_configuration_fails_closed(self):
        for content in ('WRONG_KEY_NAME=abcdef\n', 'VLLM_API_KEY=\n',
                        'VLLM_API_KEY=' + self.secret + '\nVLLM_API_KEY=\n',
                        'VLLM_API_KEY=too-short\n'):
            with self.subTest(content=content):
                self.api_env.write_text(content)
                result = self.launch()
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('docker create', result.stdout)
                self.assertNotIn(self.secret, result.stdout + result.stderr)

    def test_public_or_symlinked_secret_is_refused(self):
        self.api_env.chmod(0o644)
        self.assertNotEqual(self.launch().returncode, 0)
        self.api_env.chmod(0o600)
        link = self.api_env.with_name('link.env')
        link.symlink_to(self.api_env)
        self.assertNotEqual(self.launch(API_ENV_FILE=str(link)).returncode, 0)

    def test_unsafe_production_overrides_are_refused(self):
        for overrides in [{'SEQS': '6'}, {'MAXLEN': '524288'}, {'MTP': '03'},
                          {'CHUNK': ''}, {'CAPTURE_SIZES': 'auto'}, {'CAPTURE_SIZES': '4,0'},
                          {'PREFIX_CACHE': '0'}, {'KV_CACHE_MEMORY_BYTES': ''},
                          {'KV_CACHE_MEMORY_BYTES': '10737418240'}, {'KV_DTYPE': 'auto'},
                          {'PLE_MODE': 'none'}, {'PLE_WORKERS': '128'}, {'GMU': '0.8'},
                          {'HOST_RESERVE_GIB': '30'}, {'DRAFT_VOCAB': '0'}, {'PATCH_DIR': '/tmp'},
                          {'MAMBA_SSM_CACHE_DTYPE': 'float32'}, {'IMAGE': 'local-ai:latest'},
                          {'BIND_HOST': '0.0.0.0'}, {'BIND_HOST': '192.168.1.2'},
                          {'BIND_HOST': '::1'}, {'PORT': '65536'}]:
            with self.subTest(overrides=overrides):
                result = self.launch(**overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('docker create', result.stdout)


if __name__ == '__main__':
    unittest.main()
