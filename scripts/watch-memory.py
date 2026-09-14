#!/usr/bin/env python3
"""Foreground model lifecycle: durable interlock, gated load, pidfd RAM guard.

Adapted from the tested guarded_start.py. Linux/systemd, local rootful Docker
and Python 3.9+ with pidfd support are required. No GPU imports in this process.
"""
import argparse
import fcntl
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
import uuid

IMAGE = 'sha256:213e470219da6e21119f0a13a4df35f7e0d1f18ed2fb26e43f68c58965b30dfe'
NAME = 'qwen38-flash-two-slot'
OWNER = 'qwen38-two-slot-v1'
STATE_DIR = Path('/var/lib/local-ai-model')
FLOOR_KIB = 20 * 1024 * 1024
# Cold loading consumed approximately 96 GiB in the two-slot trial. Preserve the
# runtime floor before allocating; steady-state free RAM is not a load budget.
PREFLIGHT_KIB = 116 * 1024 * 1024
STARTUP_SECONDS = 1200
FULL_ID = re.compile(r'[0-9a-f]{64}')
ROOT_UID = 0
BOOT_TIME = time.monotonic()
# One lifecycle per process. None disables long leases after CUDA is released.
STARTUP_LEASE_UNTIL = None


class Refusal(RuntimeError):
    pass


def private_file(path, owner=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != (ROOT_UID if owner is None else owner) or info.st_mode & 0o077:
            raise Refusal('Credential/state file must be root-owned and private')
        return stream.read(16384)


def config():
    if os.environ.get('IMAGE', IMAGE) != IMAGE:
        raise Refusal('Production requires the tested immutable image ID')
    host = os.environ.get('BIND_HOST', '100.64.255.60')
    address = ipaddress.IPv4Address(host)
    if not (address.is_loopback or address in ipaddress.ip_network('100.64.0.0/10')):
        raise Refusal('Bind address must be loopback or a Tailscale IPv4 address')
    port = os.environ.get('PORT', '8000')
    if not re.fullmatch(r'[1-9][0-9]{0,4}', port) or int(port) > 65535:
        raise Refusal('Invalid port')
    # Rootless dry-run fixtures are allowed; no container is created in that mode.
    owner = os.geteuid() if os.environ.get('DRY_RUN') == '1' else ROOT_UID
    text = private_file(os.environ.get('API_ENV_FILE', '/home/andre/local-ai/secrets/api.env'), owner)
    if not re.fullmatch(r'VLLM_API_KEY=[A-Za-z0-9_-]{32,}\n?', text):
        raise Refusal('Credential must contain only one URL-safe VLLM_API_KEY (32+ characters)')
    return host, int(port), text.rstrip('\n').split('=', 1)[1]


class State:
    def __init__(self, directory=STATE_DIR):
        self.directory = directory
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != ROOT_UID or info.st_mode & 0o077:
            raise Refusal('State directory must be root-owned, private, and not a symlink')
        self.path = directory / 'state.json'
        self.lock_fd = None

    def lock(self):
        fd = os.open(self.directory / 'lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != ROOT_UID or info.st_mode & 0o077:
                raise Refusal('Unsafe lifecycle lock')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            os.close(fd)
            raise Refusal('Supervisor or another lifecycle action is active') from None
        self.lock_fd = fd

    def close(self):
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None

    def load(self):
        try:
            record = json.loads(private_file(self.path))
        except FileNotFoundError:
            return None
        if (not isinstance(record, dict) or record.get('phase') not in ('dirty', 'ready', 'latched', 'clean')
                or not re.fullmatch(r'[0-9a-f]{32}', record.get('run', ''))
                or record.get('image') != IMAGE
                or (record.get('container_id') is not None and not FULL_ID.fullmatch(record['container_id']))):
            raise Refusal('Invalid lifecycle state; preserve it for manual investigation')
        return record

    def save(self, record):
        fd, temporary = tempfile.mkstemp(dir=self.directory, prefix='.state-')
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(record, stream)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def docker(*args):
    global STARTUP_LEASE_UNTIL
    if STARTUP_LEASE_UNTIL is not None:
        notify('EXTEND_TIMEOUT_USEC=20000000')
        STARTUP_LEASE_UNTIL = time.monotonic() + 20
    # Pin the local daemon: DOCKER_HOST/context/TLS environment cannot retarget stops.
    result = subprocess.run(['/usr/bin/docker', '--host', 'unix:///var/run/docker.sock', *args],
                            capture_output=True, text=True, timeout=15,
                            env={'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'HOME': '/root'})
    if result.returncode:
        # Docker errors/inspect output may contain env-file secrets. Never relay them.
        raise Refusal('Docker lifecycle operation failed')
    return result.stdout.strip()


def inspect_id(cid):
    if not FULL_ID.fullmatch(cid):
        raise Refusal('A full container ID is required')
    objects = json.loads(docker('container', 'inspect', cid))
    if len(objects) != 1 or objects[0]['Id'] != cid:
        raise Refusal('Container ID mismatch')
    return objects[0]


def named_container():
    ids = docker('container', 'ls', '--all', '--no-trunc', '--filter', 'name=^/' + NAME + '$', '--format', '{{.ID}}').splitlines()
    if not ids:
        return None
    if len(ids) != 1:
        raise Refusal('Ambiguous production container name')
    return inspect_id(ids[0])


def verify_identity(container, record):
    cid = container['Id']
    labels = container['Config'].get('Labels') or {}
    host = container['HostConfig']
    if (not FULL_ID.fullmatch(cid) or container['Name'] != '/' + NAME
            or (record.get('container_id') is not None and cid != record['container_id'])
            or labels.get('local-ai.lifecycle') != OWNER
            or labels.get('local-ai.run') != record['run']
            or container['Image'] != IMAGE or container['Config']['Image'] != IMAGE
            or host.get('PidMode', '') != '' or host['RestartPolicy']['Name'] != 'no'
            or host.get('Privileged', False) or host.get('AutoRemove', False)):
        raise Refusal('Refusing container with unverified ownership, image, namespace or restart policy')
    return cid


def target(record):
    container = named_container()
    if record.get('container_id'):
        # An absent name is not proof that the recorded object has stopped: it may
        # have been renamed. Resolve the recorded ID independently and fail closed.
        ids = docker('container', 'ls', '--all', '--no-trunc', '--filter',
                     'id=' + record['container_id'], '--format', '{{.ID}}').splitlines()
        if ids:
            container = inspect_id(record['container_id'])
    if container is not None:
        verify_identity(container, record)
    return container


def stopped(container):
    state = container['State']
    return (state['Status'] in ('created', 'exited', 'dead') and not state['Running']
            and not state.get('Restarting', False) and not state.get('Paused', False)
            and state['Pid'] == 0)


def attach(container, record):
    cid = verify_identity(container, record)
    pid = container['State']['Pid']
    if not container['State']['Running'] or pid <= 1:
        raise Refusal('Container exited before pidfd attachment')
    fd = os.pidfd_open(pid)
    try:
        cgroup = Path(f'/proc/{pid}/cgroup').read_text()
        if (not re.search(r'(?:/|docker-)' + cid + r'(?:\.scope)?(?:/|\n|$)', cgroup)
                or os.stat(f'/proc/{pid}/ns/pid').st_ino == os.stat('/proc/self/ns/pid').st_ino):
            raise Refusal('Unverified container init PID/cgroup/private namespace')
        # Pin before reading /proc, then re-inspect to close the exit/PID-reuse race.
        fresh = inspect_id(cid)
        verify_identity(fresh, record)
        if fresh['State']['Pid'] != pid or not fresh['State']['Running'] or exited(fd):
            raise Refusal('Container init changed during pidfd attachment')
        return fd, pid
    except BaseException:
        os.close(fd)
        raise


def exited(fd, milliseconds=0):
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    events = poll.poll(milliseconds)
    if any(event & (select.POLLERR | select.POLLNVAL) for _, event in events):
        raise Refusal('pidfd monitor failed')
    return bool(events)


def send(fd, sig):
    try:
        signal.pidfd_send_signal(fd, sig)
    except ProcessLookupError:
        pass


def available_kib():
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    value = int(memory['MemAvailable'].split()[0])
    if value < 0 or value > int(memory['MemTotal'].split()[0]):
        raise Refusal('Invalid available-memory sample')
    return value


def notify(message):
    address = os.environ.get('NOTIFY_SOCKET')
    if not address:
        raise Refusal('Run through local-ai-memory.service (systemd notify socket required)')
    if address.startswith('@'):
        address = '\0' + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.settimeout(0.05)
        sock.connect(address)
        sock.sendall(message.encode())


class Health:
    def __init__(self, host, port, key):
        self.host, self.port, self.key = host, port, key
        self.result = (time.monotonic(), False)
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.done.is_set():
            connection = http.client.HTTPConnection(self.host, self.port, timeout=2)
            healthy = False
            try:
                connection.request('GET', '/health', headers={'Authorization': 'Bearer ' + self.key})
                healthy = connection.getresponse().status == 200
            except (OSError, http.client.HTTPException):
                pass
            finally:
                connection.close()
            self.result = (time.monotonic(), healthy)
            self.done.wait(2)


def finish(state, record, clean_requested, confirmed_stopped, reason):
    record.update(phase='clean' if clean_requested and confirmed_stopped else 'latched',
                  reason=reason, updated_at=time.time())
    state.save(record)


def stop_owned(state, record):
    container = target(record)
    forced = container is not None and not stopped(container)
    if container is not None and not stopped(container):
        fd, _ = attach(container, record)
        try:
            send(fd, signal.SIGKILL)
            if not exited(fd, 10000):
                raise Refusal('Owned container init did not exit')
        finally:
            os.close(fd)
        confirm_stopped(container['Id'], record)
    if forced or record['phase'] != 'clean':
        finish(state, record, False, True, 'supervisor_teardown; operator reset required')


def confirm_stopped(cid, record):
    deadline = time.monotonic() + 15
    while True:
        container = inspect_id(cid)
        verify_identity(container, record)
        if stopped(container):
            return container
        if time.monotonic() >= deadline:
            raise Refusal('Docker did not confirm a stopped owned container')
        time.sleep(0.25)


def reset(state):
    record = state.load()
    container = named_container() if record is None else target(record)
    if container is not None:
        if record is None:
            raise Refusal('No saved ownership record for existing container')
        if not stopped(container):
            raise Refusal('Refusing latch reset while owned container is live')
    if record is not None:
        finish(state, record, True, True, 'explicit operator reset after confirmed stop')
    print('Latch clear; model remains stopped', flush=True)


def supervise(state, args):
    global STARTUP_LEASE_UNTIL
    host, port, key = config()
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise Refusal('Linux pidfd support is required')
    notify('STATUS=Preflight; not ready')
    previous = state.load()
    if previous is not None and previous['phase'] != 'clean':
        raise Refusal('Dirty-run latch is set; investigate and explicitly reset before starting')
    old = named_container() if previous is None else target(previous)
    if old is not None:
        if previous is None or not stopped(old):
            raise Refusal('Existing production container is live or lacks saved ownership')
        docker('container', 'rm', verify_identity(old, previous))
    if available_kib() < PREFLIGHT_KIB:
        raise Refusal('Cold-load preflight requires at least 116 GiB MemAvailable')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, port))  # Existing listeners must not satisfy our readiness.
    record = {'phase': 'dirty', 'run': uuid.uuid4().hex, 'image': IMAGE,
              'container_id': None, 'updated_at': time.time(), 'reason': 'load armed'}
    # Durable before create/start: SIGKILL, power loss, and reboot all leave an interlock.
    state.save(record)
    fd = None
    health = None
    stop_requested = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())
    signal.signal(signal.SIGINT, lambda *_: stop_requested.set())
    clean = False
    reason = 'startup or monitor failed'
    try:
        cid = docker('container', 'create', '--name', NAME,
                     '--label', 'local-ai.lifecycle=' + OWNER,
                     '--label', 'local-ai.run=' + record['run'], *args)
        if not FULL_ID.fullmatch(cid):
            raise Refusal('Docker did not return a full container ID')
        record['container_id'] = cid
        state.save(record)
        created = inspect_id(cid)
        verify_identity(created, record)
        if (created['Config'].get('Entrypoint') != ['python3']
                or created['Config'].get('Cmd', [])[:2] != ['-c',
                    'import os,signal,sys; signal.signal(signal.SIGUSR1,lambda *_:os.execvp("vllm",["vllm","serve",*sys.argv[1:]])); signal.pause()']
                or [item for item in created['Config'].get('Env', []) if item.startswith('VLLM_API_KEY=')]
                    != ['VLLM_API_KEY=' + key]):
            raise Refusal('Missing pre-CUDA gate or authenticated environment')
        if created['State']['Status'] != 'created':
            raise Refusal('New container was started outside its supervisor')
        if stop_requested.is_set() or available_kib() < PREFLIGHT_KIB:
            raise Refusal('Start cancelled or preflight reserve lost')
        docker('container', 'start', cid)
        fd, pid = attach(inspect_id(cid), record)
        gate_deadline = time.monotonic() + 10
        while True:
            status = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines())
            if int(status['SigCgt'].strip(), 16) & (1 << (signal.SIGUSR1 - 1)):
                break
            if exited(fd, 50) or stop_requested.is_set() or time.monotonic() >= gate_deadline:
                raise Refusal('Init did not register its pre-CUDA release handler')
        # systemd's regular watchdog starts only at READY=1. Let its initial
        # 30-second startup grace expire while CUDA is gated; renewable five-
        # second startup deadlines then protect loading without false readiness.
        while time.monotonic() < max(BOOT_TIME + 31, (STARTUP_LEASE_UNTIL or 0) + 1):
            notify('EXTEND_TIMEOUT_USEC=5000000')
            if exited(fd, 250) or available_kib() < PREFLIGHT_KIB or stop_requested.is_set():
                raise Refusal('Load refused while arming startup monitoring')
        notify('EXTEND_TIMEOUT_USEC=5000000')
        if available_kib() < PREFLIGHT_KIB or stop_requested.is_set():
            raise Refusal('Load refused: preflight reserve lost or stop requested')
        STARTUP_LEASE_UNTIL = None
        health = Health(host, port, key)
        health.thread.start()
        started = time.monotonic()
        last_healthy = started
        ready = False
        send(fd, signal.SIGUSR1)
        print('Watching verified model init: 250ms poll, 20 GiB RAM floor; not ready', flush=True)
        while True:
            # No Docker, network requests or disk writes in this pressure path.
            if available_kib() < FLOOR_KIB:
                reason = 'memory reserve crossed'
                break
            if exited(fd):
                reason = 'container exited without an operator stop'
                break
            now = time.monotonic()
            checked, healthy = health.result
            if not health.thread.is_alive() or now - checked > 10:
                reason = 'health monitor failed'
                break
            if healthy:
                last_healthy = checked
                if not ready:
                    record.update(phase='ready', reason='authenticated health 200', updated_at=time.time())
                    state.save(record)
                    notify('READY=1\nSTATUS=Authenticated health 200; RAM guard active')
                    ready = True
                    print('Ready: authenticated /health returned 200', flush=True)
            if (not ready and now - started >= STARTUP_SECONDS) or (ready and now - last_healthy > 60):
                reason = 'startup timeout' if not ready else 'health unavailable for 60 seconds'
                break
            if stop_requested.is_set():
                reason = 'operator stop'
                notify('STOPPING=1\nSTATUS=Stopping verified model init')
                send(fd, signal.SIGTERM)
                deadline = now + 30
                # Continue RAM and pidfd monitoring during graceful teardown.
                while not exited(fd, 250):
                    notify('WATCHDOG=1')
                    if available_kib() < FLOOR_KIB or time.monotonic() >= deadline:
                        reason = 'forced stop: reserve crossed or graceful deadline expired'
                        break
                else:
                    clean = True
                break
            notify('WATCHDOG=1\nEXTEND_TIMEOUT_USEC=5000000')
            exited(fd, 250)
    finally:
        if health is not None:
            health.done.set()
        # Stop before any slow persistence/Docker operation, including exceptions.
        if fd is not None:
            try:
                if not clean:
                    send(fd, signal.SIGKILL)
                if not exited(fd, 10000):
                    clean = False
                    raise Refusal('Model init did not exit after stop')
            finally:
                os.close(fd)
        if fd is not None:
            # The pidfd already proved death. Wait for Docker's asynchronous
            # exit metadata instead of reattaching to an already-dead PID.
            container = confirm_stopped(record['container_id'], record)
        else:
            container = target(record)
            if container is not None and not stopped(container):
                stop_owned(state, record)
                clean = False
            if record['container_id'] and container is not None:
                container = confirm_stopped(record['container_id'], record)
        clean = (clean and container is not None and stopped(container)
                 and container['State'].get('ExitCode') in (0, 143)
                 and not container['State'].get('OOMKilled', False)
                 and not container['State'].get('Error'))
        finish(state, record, clean, container is not None and stopped(container), reason)
    print('Clean stop' if clean else 'Stopped and latched: ' + reason, flush=True)
    return 0 if clean else 1


def main():
    global STARTUP_LEASE_UNTIL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'check-config', 'status', 'reset', 'stop-owned'))
    parser.add_argument('create_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == 'check-config':
        config()
        return 0
    if os.geteuid() != 0:
        raise Refusal('Lifecycle operations require root')
    state = State()
    try:
        if args.action == 'status':
            print(json.dumps(state.load() or {'phase': 'clean', 'reason': 'never started'}, indent=2))
            return 0
        state.lock()
        if args.action == 'run':
            command = args.create_args
            if command[:1] == ['--']:
                command = command[1:]
            if not command:
                raise Refusal('Use serve.sh to construct the production profile')
            STARTUP_LEASE_UNTIL = time.monotonic()
            return supervise(state, command)
        if args.action == 'reset':
            reset(state)
        elif args.action == 'stop-owned':
            record = state.load()
            if record is not None:
                stop_owned(state, record)
            elif named_container() is not None:
                raise Refusal('Existing container has no ownership record; refusing teardown')
        return 0
    finally:
        STARTUP_LEASE_UNTIL = None
        state.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # Never print command lines, inspect objects, HTTP headers, or credentials.
        print('Local AI refused/failed: ' + (str(error) if isinstance(error, Refusal) else type(error).__name__), flush=True)
        raise SystemExit(1)
