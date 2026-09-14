"""CPU-only lifecycle proofs: fake Docker metadata, real sacrificial pidfds.

No model, CUDA, Docker daemon, systemd, network listener, or root required.
"""
import copy
import importlib.util
import json
import http.server
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import threading
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
            'HostConfig': {'PidMode': '', 'RestartPolicy': {'Name': 'no'}, 'Privileged': False, 'Init': True},
            'State': {'Status': 'created', 'Running': False, 'Pid': 0},
        }
        self.notifications = []
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
                                    ('HostConfig', 'RestartPolicy', {'Name': 'always'}),
                                    ('HostConfig', 'Init', False)]:
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

    def test_gate_handler_belongs_to_verified_init_and_cgroup(self):
        init_pid, child_pid = 123, 456
        proc = self.state.directory / 'proc'
        children = proc / str(init_pid) / 'task' / str(init_pid) / 'children'
        children.parent.mkdir(parents=True)
        children.write_text(f'{child_pid}\n')
        child = proc / str(child_pid)
        child.mkdir()
        (child / 'status').write_text(
            f'Name:\tpython3\nPPid:\t{init_pid}\nSigCgt:\t0000000000000200\n')
        (child / 'cgroup').write_text(f'0::/system.slice/docker-{CID}.scope\n')
        self.assertTrue(guard.gate_handler_ready(init_pid, CID, proc))
        (child / 'cgroup').write_text('0::/system.slice/docker-unrelated.scope\n')
        self.assertFalse(guard.gate_handler_ready(init_pid, CID, proc))

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
        runtime = False
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
            if args[:2] == ('info', '--format'):
                return 'false'
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
            if not runtime:
                return guard.PREFLIGHT_KIB
            if outcome == 'memory_error':
                raise OSError('unreadable meminfo')
            if outcome in ('pressure', 'notify_error'):
                return guard.FLOOR_KIB - 1
            return guard.PREFLIGHT_KIB

        def notify(message):
            self.notifications.append(message)
            if outcome == 'notify_error' and message.startswith('STOPPING=1'):
                raise OSError('notify socket unavailable')

        health = mock.Mock()
        health.result = (time.monotonic(), outcome == 'ready_clean')
        health.thread.is_alive.return_value = True
        def start_health():
            nonlocal runtime
            runtime = True
            if outcome in ('clean', 'oom', 'ready_clean'):
                handlers[signal.SIGTERM](signal.SIGTERM, None)

        health.thread.start.side_effect = start_health

        with mock.patch.object(guard, 'config', return_value=('127.0.0.1', 8000, 'x' * 32)), \
                mock.patch.object(guard, 'notify', side_effect=notify), mock.patch.object(guard, 'docker', side_effect=docker), \
                mock.patch.object(guard, 'available_kib', side_effect=available), \
                mock.patch.object(guard, 'compact_host_memory'), \
                mock.patch.object(guard, 'attach', side_effect=lambda *_: (os.pidfd_open(child.pid), child.pid)), \
                mock.patch.object(guard, 'Health', return_value=health), \
                mock.patch.object(guard, 'BOOT_TIME', time.monotonic() - 32), \
                mock.patch.object(guard.signal, 'signal', side_effect=lambda sig, handler: handlers.update({sig: handler})), \
                mock.patch.object(guard.socket, 'socket'), \
                mock.patch.object(guard, 'gate_handler_ready', return_value=True):
            if outcome == 'memory_error':
                with self.assertRaises(OSError):
                    guard.supervise(self.state, ['unused'])
            else:
                result = guard.supervise(self.state, ['unused'])
                self.assertEqual(result, 0 if outcome in ('clean', 'ready_clean') else 1)
        self.assertIsNotNone(child.poll())
        self.assertEqual(self.state.load()['phase'], 'clean' if outcome in ('clean', 'ready_clean') else 'latched')

    def test_pressure_kills_owned_init_and_persists_latch(self):
        self.run_supervisor('pressure')

    def test_monitor_failure_kills_owned_init_and_persists_latch(self):
        self.run_supervisor('memory_error')

    def test_authenticated_health_marks_ready_before_clean_stop(self):
        self.run_supervisor('ready_clean')
        self.assertTrue(any('EXTEND_TIMEOUT_USEC=120000000' in message
                            for message in self.notifications if message.startswith('STOPPING=1')))
        self.assertTrue(any(message.startswith('READY=1') for message in self.notifications))
        ready_index = next(index for index, message in enumerate(self.notifications)
                           if message.startswith('READY=1'))
        self.assertTrue(any(message == 'EXTEND_TIMEOUT_USEC=120000000'
                            for message in self.notifications[:ready_index]))

    def test_operator_stop_clears_dirty_state_only_after_confirmed_exit(self):
        self.run_supervisor('clean')
        self.assertTrue(any('EXTEND_TIMEOUT_USEC=120000000' in message
                            for message in self.notifications if message.startswith('STOPPING=1')))

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
                mock.patch.object(guard, 'renew_startup_lease'), \
                mock.patch.object(guard, 'send', side_effect=checked_send):
            self.run_supervisor('clean')
        self.assertEqual(released, [True])

    def test_oom_during_operator_stop_keeps_latch(self):
        self.run_supervisor('oom')

    def test_shutdown_notification_failure_still_kills_and_latches(self):
        self.run_supervisor('notify_error')

    def test_waits_for_private_address_but_does_not_retry_busy_port(self):
        with mock.patch.object(guard.socket, 'socket') as factory, \
                mock.patch.object(guard, 'renew_startup_lease'), \
                mock.patch.object(guard.time, 'sleep') as sleep:
            probe = factory.return_value.__enter__.return_value
            bind = probe.bind
            bind.side_effect = [OSError(guard.errno.EADDRNOTAVAIL, 'not assigned'), None]
            guard.wait_for_bind('100.64.255.60', 8000)
            self.assertEqual(bind.call_count, 2)
            probe.setsockopt.assert_called_with(guard.socket.SOL_SOCKET, guard.socket.SO_REUSEADDR, 1)
            sleep.reset_mock()
            bind.side_effect = OSError(guard.errno.EADDRINUSE, 'occupied')
            with self.assertRaises(guard.Refusal):
                guard.wait_for_bind('100.64.255.60', 8000)
            sleep.assert_not_called()

    def test_private_file_refuses_fifo_without_blocking(self):
        fifo = self.state.directory / 'credential.fifo'
        os.mkfifo(fifo, 0o600)
        started = time.monotonic()
        with self.assertRaises(guard.Refusal):
            guard.private_file(fifo, os.geteuid())
        self.assertLess(time.monotonic() - started, 1)

    def test_health_requires_bearer_key_and_http_200(self):
        requests = []
        received = threading.Event()

        class Handler(http.server.BaseHTTPRequestHandler):
            status = 200

            def do_GET(self):
                requests.append((self.path, self.headers.get('Authorization')))
                self.send_response(self.status)
                self.end_headers()
                received.set()

            def log_message(self, *_):
                pass

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            healthy = guard.Health('127.0.0.1', server.server_port, 'correct-key')
            healthy.thread.start()
            self.assertTrue(received.wait(2))
            deadline = time.monotonic() + 2
            while not healthy.result[1] and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(healthy.result[1])
            self.assertEqual(requests[-1], ('/health', 'Bearer correct-key'))
            healthy.done.set()
            healthy.thread.join(2)

            received.clear()
            Handler.status = 401
            rejected = guard.Health('127.0.0.1', server.server_port, 'wrong-key')
            rejected.thread.start()
            self.assertTrue(received.wait(2))
            self.assertFalse(rejected.result[1])
            rejected.done.set()
            rejected.thread.join(2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_missing_private_address_has_a_bounded_wait(self):
        now = 0

        def advance(_):
            nonlocal now
            now = 60

        with mock.patch.object(guard.socket, 'socket') as factory, \
                mock.patch.object(guard, 'renew_startup_lease'), \
                mock.patch.object(guard.time, 'monotonic', side_effect=lambda: now), \
                mock.patch.object(guard.time, 'sleep', side_effect=advance):
            bind = factory.return_value.__enter__.return_value.bind
            bind.side_effect = [OSError(guard.errno.EADDRNOTAVAIL, 'not assigned'),
                                OSError(guard.errno.EADDRNOTAVAIL, 'not assigned'),
                                AssertionError('Address wait exceeded its deadline')]
            with self.assertRaises(guard.Refusal):
                guard.wait_for_bind('100.64.255.60', 8000)

    def test_host_compaction_writes_kernel_trigger(self):
        trigger = self.state.directory / 'compact_memory'
        trigger.write_text('0\n')
        with mock.patch.object(guard, 'COMPACT_MEMORY', trigger), \
                mock.patch.object(guard, 'renew_startup_lease'):
            guard.compact_host_memory()
        self.assertEqual(trigger.read_text(), '1\n')

    def test_host_compaction_precedes_durable_state_and_docker(self):
        def stop_after_compaction():
            self.assertIsNone(self.state.load())
            raise guard.Refusal('test boundary')

        with mock.patch.object(guard, 'config', return_value=('127.0.0.1', 8000, 'x' * 32)), \
                mock.patch.object(guard, 'notify'), \
                mock.patch.object(guard, 'named_container', return_value=None), \
                mock.patch.object(guard, 'available_kib', return_value=guard.PREFLIGHT_KIB), \
                mock.patch.object(guard, 'wait_for_bind'), \
                mock.patch.object(guard, 'compact_host_memory', side_effect=stop_after_compaction), \
                mock.patch.object(guard, 'docker') as docker:
            with self.assertRaises(guard.Refusal):
                guard.supervise(self.state, ['unused'])
            docker.assert_not_called()
        self.assertIsNone(self.state.load())

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

    def test_live_restore_is_refused_before_state_or_container_creation(self):
        with mock.patch.object(guard, 'config', return_value=('127.0.0.1', 8000, 'x' * 32)), \
                mock.patch.object(guard, 'notify'), \
                mock.patch.object(guard, 'available_kib', return_value=guard.PREFLIGHT_KIB), \
                mock.patch.object(guard, 'wait_for_bind'), \
                mock.patch.object(guard, 'compact_host_memory'), \
                mock.patch.object(guard, 'docker', return_value='true') as docker:
            with self.assertRaises(guard.Refusal):
                guard.supervise(self.state, ['unused'])
        docker.assert_called_once_with('info', '--format', '{{.LiveRestoreEnabled}}')
        self.assertIsNone(self.state.load())


    def test_stop_post_preserves_existing_latched_reason(self):
        self.record.update(phase='latched', reason='memory reserve crossed')
        self.state.save(self.record)
        with mock.patch.object(guard, 'target', return_value=self.container):
            guard.stop_owned(self.state, self.record)
        saved = self.state.load()
        self.assertEqual(saved['phase'], 'latched')
        self.assertEqual(saved['reason'], 'memory reserve crossed')

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
