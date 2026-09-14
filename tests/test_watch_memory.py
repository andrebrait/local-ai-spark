"""CPU-only lifecycle proofs: fake Docker metadata, real sacrificial pidfds.

No model, CUDA, Docker daemon, systemd, network listener, or root required.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('watch_memory', ROOT / 'scripts/watch-memory.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
CID = 'a' * 64
GATE = 'import os,signal,sys; signal.signal(signal.SIGUSR1,lambda *_:os.execvp("vllm",["vllm","serve",*sys.argv[1:]])); signal.pause()'


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        owner = mock.patch.object(guard, 'ROOT_UID', os.geteuid())
        owner.start()
        self.addCleanup(owner.stop)
        self.state = guard.State(Path(temporary.name))
        self.addCleanup(self.state.close)
        self.record = {'phase': 'dirty', 'run': 'b' * 32, 'image': guard.IMAGE, 'container_id': CID}
        self.container = {
            'Id': CID, 'Name': '/' + guard.NAME, 'Image': guard.IMAGE,
            'Config': {'Image': guard.IMAGE, 'Entrypoint': ['python3'], 'Cmd': ['-c', GATE],
                       'Env': ['VLLM_API_KEY=' + 'x' * 32],
                       'Labels': {'local-ai.lifecycle': guard.OWNER, 'local-ai.run': self.record['run']}},
            'HostConfig': {'PidMode': '', 'RestartPolicy': {'Name': 'no'}, 'Privileged': False},
            'State': {'Status': 'created', 'Running': False, 'Pid': 0},
        }
        self.child = None

    def spawn(self):
        child = subprocess.Popen([sys.executable, '-c',
            'import signal,time; signal.signal(signal.SIGUSR1,lambda *_:None); print("ready",flush=True); time.sleep(60)'],
            stdout=subprocess.PIPE, text=True)
        self.addCleanup(self.reap, child)
        self.assertEqual(child.stdout.readline().strip(), 'ready')
        child.stdout.close()
        self.child = child
        return child

    @staticmethod
    def reap(child):
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)

    def test_dirty_record_survives_new_supervisor_and_refuses_reload(self):
        self.state.save(self.record)
        reopened = guard.State(self.state.directory)
        with mock.patch.object(guard, 'config', return_value=('127.0.0.1', 8000, 'x' * 32)), \
                mock.patch.object(guard, 'notify'), mock.patch.object(guard, 'docker') as docker:
            with self.assertRaises(guard.Refusal):
                guard.supervise(reopened, ['unused'])
            docker.assert_not_called()
        self.assertEqual(reopened.load()['phase'], 'dirty')
        self.assertEqual(self.state.path.stat().st_mode & 0o777, 0o600)

    def test_reset_refuses_live_target_then_allows_confirmed_stop(self):
        self.state.save(self.record)
        self.container['State'] = {'Status': 'running', 'Running': True, 'Pid': 1234}
        with mock.patch.object(guard, 'target', return_value=self.container):
            with self.assertRaises(guard.Refusal):
                guard.reset(self.state)
            self.assertEqual(self.state.load()['phase'], 'dirty')
            self.container['State'] = {'Status': 'exited', 'Running': False, 'Pid': 0}
            guard.reset(self.state)
        self.assertEqual(self.state.load()['phase'], 'clean')

    def test_lifecycle_lock_excludes_operator_reset(self):
        self.state.lock()
        other = guard.State(self.state.directory)
        with self.assertRaises(guard.Refusal):
            other.lock()
        self.state.close()
        other.lock()
        other.close()

    def test_identity_mismatches_never_signal_unrelated_process(self):
        child = self.spawn()
        self.container['State'] = {'Status': 'running', 'Running': True, 'Pid': child.pid}
        variants = []
        for section, key, value in [('Config', 'Image', 'local-ai:latest'),
                                    ('HostConfig', 'PidMode', 'host'),
                                    ('HostConfig', 'PidMode', 'container:other'),
                                    ('HostConfig', 'RestartPolicy', {'Name': 'always'})]:
            bad = copy.deepcopy(self.container)
            bad[section][key] = value
            variants.append(bad)
        for field, value in [('Id', 'c' * 64), ('Image', 'sha256:' + 'd' * 64),
                             ('Name', '/qwen38-flash')]:
            bad = copy.deepcopy(self.container)
            bad[field] = value
            variants.append(bad)
        bad = copy.deepcopy(self.container)
        bad['Config']['Labels']['local-ai.run'] = 'c' * 32
        variants.append(bad)
        for bad in variants:
            with self.subTest(container=bad), mock.patch.object(guard, 'target', return_value=bad):
                with self.assertRaises(guard.Refusal):
                    guard.stop_owned(self.state, self.record)
                self.assertIsNone(child.poll())

    def test_pid_attachment_refuses_process_outside_container_cgroup(self):
        child = self.spawn()
        self.container['State'] = {'Status': 'running', 'Running': True, 'Pid': child.pid}
        with self.assertRaises(guard.Refusal):
            guard.attach(self.container, self.record)
        self.assertIsNone(child.poll())

    def test_corrupt_or_public_state_fails_closed(self):
        self.state.save(self.record)
        self.state.path.chmod(0o644)
        with self.assertRaises(guard.Refusal):
            self.state.load()
        self.state.path.chmod(0o600)
        self.state.path.write_text('{broken')
        with self.assertRaises(json.JSONDecodeError):
            self.state.load()

    def run_supervisor(self, outcome):
        child = self.spawn()
        exists = False
        self.samples = 0
        handlers = {}

        def docker(*args):
            nonlocal exists
            if args[:2] == ('container', 'ls'):
                return CID if exists else ''
            if args[:2] == ('container', 'create'):
                exists = True
                saved = self.state.load()
                self.assertEqual(saved['phase'], 'dirty')
                self.container['Config']['Labels']['local-ai.run'] = saved['run']
                return CID
            if args[:2] == ('container', 'start'):
                saved = self.state.load()
                self.assertEqual(saved['container_id'], CID)
                self.assertEqual(saved['phase'], 'dirty')
                self.container['State'] = {'Status': 'running', 'Running': True, 'Pid': child.pid}
                return CID
            if args[:2] == ('container', 'inspect'):
                if child.poll() is not None:
                    code = child.returncode if child.returncode >= 0 else 128 - child.returncode
                    self.container['State'] = {'Status': 'exited', 'Running': False, 'Pid': 0,
                                               'ExitCode': code, 'OOMKilled': outcome == 'oom'}
                return json.dumps([self.container])
            raise AssertionError('Unexpected Docker operation')

        def available():
            self.samples += 1
            if self.samples <= 3:
                return guard.PREFLIGHT_KIB
            if outcome == 'memory_error':
                raise OSError('unreadable meminfo')
            if outcome == 'pressure':
                return guard.FLOOR_KIB - 1
            return guard.PREFLIGHT_KIB

        health = mock.Mock()
        health.result = (time.monotonic(), False)
        health.thread.is_alive.return_value = True
        if outcome in ('clean', 'oom'):
            health.thread.start.side_effect = lambda: handlers[signal.SIGTERM](signal.SIGTERM, None)
        original_read = Path.read_text

        def read_path(path, *args, **kwargs):
            if str(path) == f'/proc/{child.pid}/status':
                return 'SigCgt:\t0000000000000200\n'
            return original_read(path, *args, **kwargs)

        with mock.patch.object(guard, 'config', return_value=('127.0.0.1', 8000, 'x' * 32)), \
                mock.patch.object(guard, 'notify'), mock.patch.object(guard, 'docker', side_effect=docker), \
                mock.patch.object(guard, 'available_kib', side_effect=available), \
                mock.patch.object(guard, 'attach', side_effect=lambda *_: (os.pidfd_open(child.pid), child.pid)), \
                mock.patch.object(guard, 'Health', return_value=health), \
                mock.patch.object(guard, 'BOOT_TIME', time.monotonic() - 32), \
                mock.patch.object(guard.signal, 'signal', side_effect=lambda sig, handler: handlers.update({sig: handler})), \
                mock.patch.object(guard.socket, 'socket'), \
                mock.patch.object(Path, 'read_text', read_path):
            if outcome == 'memory_error':
                with self.assertRaises(OSError):
                    guard.supervise(self.state, ['unused'])
            else:
                result = guard.supervise(self.state, ['unused'])
                self.assertEqual(result, 0 if outcome == 'clean' else 1)
        self.assertIsNotNone(child.poll())
        self.assertEqual(self.state.load()['phase'], 'clean' if outcome == 'clean' else 'latched')

    def test_pressure_kills_owned_init_and_persists_latch(self):
        self.run_supervisor('pressure')

    def test_monitor_failure_kills_owned_init_and_persists_latch(self):
        self.run_supervisor('memory_error')

    def test_operator_stop_clears_dirty_state_only_after_confirmed_exit(self):
        self.run_supervisor('clean')

    def test_clean_stop_waits_for_delayed_docker_exit_metadata(self):
        inspect = guard.inspect_id
        delayed = 0

        def stale_inspect(cid):
            nonlocal delayed
            container = inspect(cid)
            if container['State']['Status'] == 'exited' and delayed < 2:
                delayed += 1
                container['State'] = {'Status': 'running', 'Running': True, 'Pid': self.child.pid}
            return container

        with mock.patch.object(guard, 'inspect_id', side_effect=stale_inspect):
            self.run_supervisor('clean')

    def test_cuda_release_waits_until_preload_lease_expires(self):
        lease_until = time.monotonic() + 0.4
        send = guard.send
        released = []

        def checked_send(fd, sig):
            if sig == signal.SIGUSR1:
                self.assertGreaterEqual(time.monotonic(), lease_until)
                self.assertIsNone(guard.STARTUP_LEASE_UNTIL)
                released.append(True)
            send(fd, sig)

        with mock.patch.object(guard, 'STARTUP_LEASE_UNTIL', lease_until), \
                mock.patch.object(guard, 'send', side_effect=checked_send):
            self.run_supervisor('clean')
        self.assertEqual(released, [True])

    def test_oom_during_operator_stop_keeps_latch(self):
        self.run_supervisor('oom')

    def test_low_preflight_memory_never_creates_container(self):
        with mock.patch.object(guard, 'config', return_value=('127.0.0.1', 8000, 'x' * 32)), \
                mock.patch.object(guard, 'notify'), mock.patch.object(guard, 'named_container', return_value=None), \
                mock.patch.object(guard, 'available_kib', return_value=115 * 1024 * 1024), \
                mock.patch.object(guard.socket, 'socket'), \
                mock.patch.object(guard, 'docker', side_effect=AssertionError('Cold-load reserve must precede Docker')) as docker:
            with self.assertRaises(guard.Refusal):
                guard.supervise(self.state, ['unused'])
            docker.assert_not_called()
        self.assertIsNone(self.state.load())

    def test_stop_post_kills_survivor_and_does_not_clear_interlock(self):
        child = self.spawn()
        self.record['phase'] = 'ready'
        self.state.save(self.record)
        self.container['State'] = {'Status': 'running', 'Running': True, 'Pid': child.pid}

        def inspect(_):
            if child.poll() is not None:
                self.container['State'] = {'Status': 'exited', 'Running': False, 'Pid': 0}
            return self.container

        with mock.patch.object(guard, 'target', return_value=self.container), \
                mock.patch.object(guard, 'attach', side_effect=lambda *_: (os.pidfd_open(child.pid), child.pid)), \
                mock.patch.object(guard, 'inspect_id', side_effect=inspect):
            guard.stop_owned(self.state, self.record)
        self.assertIsNotNone(child.poll())
        self.assertEqual(self.state.load()['phase'], 'latched')

    def test_named_replacement_cannot_hide_recorded_live_identity(self):
        replacement = copy.deepcopy(self.container)
        replacement['Id'] = 'c' * 64
        recorded = copy.deepcopy(self.container)
        recorded['Name'] = '/renamed-owned-container'
        recorded['State'] = {'Status': 'running', 'Running': True, 'Pid': 1234}
        self.state.save(self.record)
        with mock.patch.object(guard, 'named_container', return_value=replacement), \
                mock.patch.object(guard, 'docker', return_value=CID), \
                mock.patch.object(guard, 'inspect_id', return_value=recorded):
            with self.assertRaises(guard.Refusal):
                guard.reset(self.state)
        self.assertEqual(self.state.load()['phase'], 'dirty')


if __name__ == '__main__':
    unittest.main()
