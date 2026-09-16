#!/usr/bin/env python3
"""One bounded disk-undo acceptance run on zns4; check/start/status/stop/rejudge.

Reuses run_live_paired_bench.py and /data2/bench-runs/restore_vanilla_node.sh.
The host handoff/profile follow run_zns4_vanilla_arm.sh; prefix inventory and
undo/reapply checks follow deep_reorg_step6.sh. No timing comparison is made.
"""

import argparse
from dataclasses import asdict
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from analyze_validation_bench import ANSI_ESCAPE, load_jsonl, select_samples

BENCH = Path('/data2/bench-runs')
DATADIR = Path('/data2/reth_data')
JWT = Path('/data2/secrets/jwt.hex')
RESTORE = BENCH / 'restore_vanilla_node.sh'
REQUIRED_COMMIT = 'b89bd9d38'
REPLAY_BLOCKS = 200
BAD_UNDO = (
    'Undo coverage lost', 'Undo coverage interrupted', 'Disk undo refused',
    'Could not remove expired undo file', "Could not remove failed undo write",
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def clean_env(depth):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('PS_') and k not in ('MALLOC_CONF', '_RJEM_MALLOC_CONF', 'RUST_LOG')}
    env.update(PS_RETAIN_DEPTH=str(depth), PS_TRIE_REPR='exact', PS_UNDO_LAYOUT='frames',
               PS_UNDO_RECORD='on', PS_RETAIN_GENERATION='1',
               RUST_LOG='warn,partial_stateless=debug')
    return env


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def process_identity(pid):
    """PID plus start time prevents a stale stop command from signalling a reused PID."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': stat[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(identity):
    return process_identity(identity['pid']) == identity


def node_processes():
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().decode().split('\0')[:-1]
            if len(argv) < 2 or argv[1] != 'node':
                continue
            exe = Path(argv[0]).name
            if exe not in ('reth', 'reth-partial-stateless'):
                continue
            datadir = next((a.split('=', 1)[1] for a in argv if a.startswith('--datadir=')), None)
            if '--datadir' in argv:
                datadir = argv[argv.index('--datadir') + 1]
            if datadir and Path(datadir).resolve() == DATADIR.resolve():
                identity = process_identity(int(entry.name))
                if identity:
                    found.append((identity, exe, argv))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def synced():
    body = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'eth_syncing', 'params': []}).encode()
    req = urllib.request.Request('http://127.0.0.1:8545', body,
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response).get('result') is False


def preflight(args):
    require(git(args.repo, 'branch', '--show-current') == 'main', 'run from main')
    require(not git(args.repo, 'status', '--porcelain'), 'commit repository changes before start')
    subprocess.run(['git', '-C', str(args.repo), 'merge-base', '--is-ancestor', REQUIRED_COMMIT,
                    'HEAD'], check=True)
    for tool in ('cargo', 'nice', 'pgrep'):
        require(shutil.which(tool), f'{tool} is not available')
    for path in (JWT, RESTORE, args.repo / 'Cargo.lock', args.spool):
        require(path.exists(), f'missing {path}')
    require(os.access(RESTORE, os.X_OK), f'not executable: {RESTORE}')
    require(DATADIR.is_dir(), f'missing {DATADIR}')
    require(shutil.disk_usage(BENCH).free >= 20 * 1024**3, 'need 20 GiB free under bench-runs')
    mem = re.search(r'^MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text(), re.M)
    require(mem and int(mem[1]) >= 8 * 1024**2, 'need 8 GiB MemAvailable')
    require(subprocess.run(['pgrep', '-x', 'lighthouse'], stdout=subprocess.DEVNULL).returncode == 0,
            'lighthouse must be running')
    nodes = node_processes()
    require(len(nodes) == 1 and nodes[0][1] == 'reth',
            f'need exactly one vanilla reth on {DATADIR}; found {len(nodes)} owners')
    require(synced(), 'vanilla reth must be synced on 8545')
    return nodes[0]


def prefix_paths(spool):
    """Select by filename, then let ps-replay decode/validate the bounded prefix."""
    chosen, commits = [], 0
    for path in sorted(spool.glob('*.frame')):
        require(path.name.split('_', 1)[0] == f'{len(chosen):012d}', 'non-contiguous spool prefix')
        chosen.append(path)
        commits += path.name.endswith('_commit.frame')
        if commits == REPLAY_BLOCKS:
            return chosen
    raise RuntimeError(f'spool needs {REPLAY_BLOCKS} commits')


def place_reorgs(rows, depth):
    commits = [r for r in rows if r['kind'] == 'commit']
    checkpoints = [r for r in rows if r['kind'] == 'checkpoint']
    require(len(commits) == REPLAY_BLOCKS and len(checkpoints) == 1,
            'prefix must contain one checkpoint and 200 commits')
    require([r['sequence'] for r in rows] == list(range(len(rows))), 'inventory sequence gap')
    require(checkpoints[0]['sequence'] < commits[0]['sequence'], 'checkpoint must precede commits')
    require(all(r['kind'] == 'commit' for r in rows[commits[0]['sequence']:]),
            'prefix has a lifecycle event; choose another spool')
    previous = checkpoints[0]['block']
    for row in commits:
        require(row['block']['number'] == previous['number'] + 1 and
                row['parent_hash'] == previous['hash'], 'prefix is not one parent chain')
        previous = row['block']
    return [(1, commits[depth + 7]['block']['number']),
            (depth, commits[2 * depth + 15]['block']['number'])]


def judge_replay(rows, schedule, depth, commit):
    manifests = [r for r in rows if r.get('kind') == 'run_manifest']
    reports = [r for r in rows if 'blocks' in r]
    require(len(manifests) == len(reports) == 1, 'need exactly one replay manifest and report')
    manifest, report = manifests[0], reports[0]
    require(manifest['retain_depth'] == depth and manifest['undo_layout'] == 'frames'
            and manifest['undo_record'] is True and manifest['undo_dir'], 'wrong undo profile')
    require(manifest['provenance']['build_commit'] == commit
            and manifest['provenance']['build_dirty'] is False
            and manifest['allocator'] == 'jemalloc', 'wrong replay build')
    require(manifest['forced_reorgs'] == [f'{d}@{at}' for d, at in schedule], 'wrong schedule')
    require(all(report[k] is True for k in ('agreed', 'continuous', 'complete')), 'incomplete replay')
    require(all(report[k] == 0 for k in ('failures', 'disagreements', 'mutation_failures',
            'winning_branch_incomplete', 'skipped_after_fault', 'skipped_awaiting_resync')),
            'replay reported failures or skipped blocks')
    require(report['commits'] == REPLAY_BLOCKS and not report['resyncs'], 'wrong replay coverage')
    require(len(report['forced_reorgs']) == len(schedule), 'missing forced-reorg outcome')
    for outcome, (d, at) in zip(report['forced_reorgs'], schedule):
        require(outcome['depth'] == d and outcome['at'] == at and outcome['outcome'] == 'applied'
                and outcome['frames_applied'] == outcome['reapplied'] == d
                and outcome['resumed_identical'] is True
                and outcome['ancestor']['number'] == at - d, f'undo/reapply failed: {d}@{at}')
        require(len(set(outcome['reapplied_sequences'])) == d, 'incomplete reapplied sequences')
    blocks = report['blocks']
    require(len(blocks) == REPLAY_BLOCKS + sum(d for d, _ in schedule), 'wrong verdict count')
    require(all(b['verdict'] == 'accepted' and b.get('undo_resident_blocks') == 0 for b in blocks),
            'rejected block or resident undo payload')
    return {'commits': report['commits'], 'forced_reorgs': report['forced_reorgs']}


def undo_sample(root):
    sessions = {}
    for session in root.glob('undo-*'):
        files, size, temporary = [], 0, 0
        for path in session.glob('*'):
            try:
                if path.suffix == '.undo':
                    size += path.stat().st_size
                    files.append(int(path.stem))
                elif path.suffix == '.tmp':
                    temporary += 1
            except FileNotFoundError:
                pass  # A completed write or expiry can race the directory observation.
        if files or temporary:
            sessions[session.name] = {'files': sorted(files), 'bytes': size, 'temporary': temporary}
    return {'timestamp_unix': time.time(), 'sessions': sessions}


def judge_rotation(samples, depth):
    previous, full, rotated, peak = {}, False, False, 0
    for sample in samples:
        for name, observed in sample['sessions'].items():
            files = set(observed['files'])
            peak = max(peak, len(files))
            # A finishing write can coexist briefly with the K retained files.
            require(len(files) <= depth + 1 and observed['temporary'] <= 1,
                    'undo files exceeded K plus one finishing write')
            full |= len(files) >= depth
            if name in previous and files:
                old = previous[name]
                rotated |= bool(old - files) and bool(files - old) and min(files) > min(old)
            if files:
                previous[name] = files
    require(full and rotated, 'did not observe undo history filling and rotating')
    return {'max_observed_files_per_pair': peak, 'rotation_observed': rotated}


def log_fields(line):
    return {name: quoted or bare for name, quoted, bare in
            re.findall(r'\b(\w+)=(?:"([^"]*)"|(\S+))', line)}


def legacy_cold_start(lines):
    """Recognize only the old first-commit warning, corroborated by lifecycle observations.

    This compatibility rule is used only by rejudge. New runs must emit the initialization
    event instead. No exemption applies after a commit, on restart, or to another cause.
    """
    def events(marker):
        return [(i, log_fields(line)) for i, line in enumerate(lines) if marker in line]

    starts = events('Partial Stateless ExEx started')
    parents = events('Cache is not synced to the parent block.')
    commits = events('Observed cache undo retention after commit')
    readiness = events('Cache readiness changed')
    warnings = events('Undo coverage lost; discarding history behind an unspillable block')
    if (len(starts) != 1 or not parents or len(commits) < 2 or not readiness or not warnings
            or any('Cold-resetting both caches' in line for line in lines)):
        return None
    start, parent, warning, first, ready, second = (
        starts[0], parents[0], warnings[0], commits[0], readiness[0], commits[1])
    number = warning[1].get('block', '')
    if not number.isdigit():
        return None
    def has(event, **fields):
        return all(event[1].get(k) == str(v) for k, v in fields.items())
    if (start[0] < parent[0] < warning[0] < first[0] < ready[0] < second[0]
            and has(parent, block=number, cache_block=0, expected_parent_block=int(number) - 1)
            and has(warning, cause='full_fallback', dropped_generations=1, retained_depth=0)
            and has(first, block=number, retained_depth=0, resident_blocks=0)
            and has(ready, block=number, replay_depth=1, **{'from': 'cold', 'to': 'warming'})
            and has(second, block=int(number) + 1, retained_depth=1, resident_blocks=0)):
        return warning[0]
    return None


def check_undo_log(path, *, allow_legacy_cold_start=False):
    text = ANSI_ESCAPE.sub('', path.read_text(errors='replace'))
    lines = text.splitlines()
    initial = legacy_cold_start(lines) if allow_legacy_cold_start else None
    failures = [line for i, line in enumerate(lines)
                if i != initial and any(mark in line for mark in BAD_UNDO)]
    require(not failures, 'undo warnings: ' + '\n'.join(failures[:5]))
    return text, ([int(log_fields(lines[initial])['block'])] if initial is not None else [])


def judge_live(base, samples, depth, *, allow_legacy_cold_start=False):
    paired = base / 'paired'
    log = paired / 'reth-partial-stateless.log'
    raw = load_jsonl(paired / 'paired.jsonl')
    accepted, stats = select_samples(raw, load_jsonl(paired / 'engine.jsonl'), log, 0, samples)
    # The old runner can exit 0 at its wall-clock deadline with fewer than its requested samples.
    require(len(accepted) == samples, f'only {len(accepted)}/{samples} accepted samples')
    require(all(row.get('valid') is True for row in raw), 'invalid paired sample was filtered out')
    require(stats.invalid == 0 and stats.missing_log_position == 0, 'unaccounted paired samples')
    text, initial = check_undo_log(log, allow_legacy_cold_start=allow_legacy_cold_start)
    require(re.search(rf'depth={depth}\s+recording=true\s+layout="?frames"?', text),
            'disk-undo startup configuration was not logged')
    observed = [line for line in text.splitlines() if 'Observed cache undo retention after commit' in line]
    require(len(observed) >= samples, 'missing per-commit residency observations')
    require(all(re.search(r'\bresident_blocks=0\b', line) for line in observed),
            'resident undo payload observed')
    rotation = judge_rotation(load_jsonl(base / 'undo-files.jsonl'), depth)
    return {'accepted': len(accepted), 'selection': asdict(stats), **rotation,
            'legacy_cold_initialization_blocks': initial,
            'writer_wait_warnings': text.count('Undo spill waiting for disk writer')}


def rejudge(base, repo):
    """Re-evaluate saved artifacts without starting a process or replacing the original verdict."""
    result = {'judge_commit': git(repo, 'rev-parse', 'HEAD'),
              'judge_dirty': bool(git(repo, 'status', '--porcelain')),
              'judge_sha256': sha256(Path(__file__)),
              'rejudged_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'scope': 'saved artifacts only; no new binary or live run'}
    try:
        result['original_result'] = (base / 'RESULT').read_text().strip()
        build = json.loads((base / 'build.json').read_text())
        result['run_commit'] = build['commit']
        schedule = place_reorgs(load_jsonl(base / 'frames.jsonl'), build['depth'])
        require(json.loads((base / 'schedule.json').read_text()) == [list(item) for item in schedule],
                'saved schedule differs from prefix inventory')
        result['replay'] = judge_replay(load_jsonl(base / 'replay.jsonl'), schedule,
                                        build['depth'], build['commit'])
        check_undo_log(base / 'replay.log')
        result['live'] = judge_live(base, build['samples'], build['depth'],
                                    allow_legacy_cold_start=True)
        restore = (base / 'restore.log').read_text()
        relaunched = re.search(r'restore: vanilla reth relaunched, pid (\d+)', restore)
        require(relaunched and f'restore: pid {relaunched[1]} alive after 15s' in restore,
                'missing successful vanilla restoration record')
        require('RESTORE FAILED' not in result['original_result'], 'original restoration failed')
        result['restoration'] = 'confirmed in saved restore.log; current node not queried'
        result['verdict'] = 'PASS'
    except Exception as error:
        result.update(verdict='FAIL', error=f'{type(error).__name__}: {error}')
    write_json(base / 'rejudged-result.json', result)
    (base / 'REJUDGED_RESULT').write_text(result['verdict'] + '\n')
    print(f'{result["verdict"]}: {base / "rejudged-result.json"}')
    if 'error' in result:
        print(result['error'])
    return int(result['verdict'] != 'PASS')


class Runner:
    def __init__(self, args):
        self.args, self.base = args, args.output
        self.env = clean_env(args.depth)
        self.child = None
        self.handoff = False
        self.original = None
        self.restore_env = {k: v for k, v in self.env.items()
                            if not k.startswith('PS_') and k != 'RUST_LOG'}

    def stage(self, name):
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())} {name}', flush=True)
        (self.base / 'STATUS').write_text(name + '\n')

    def run(self, command, log, *, env=None, timeout=7200, monitor=None, stdout=None):
        with (self.base / log).open('wb') as err:
            self.child = subprocess.Popen([str(v) for v in command], cwd=self.args.repo,
                                          env=env or self.env, stdin=subprocess.DEVNULL,
                                          stdout=stdout or err, stderr=err,
                                          start_new_session=True, close_fds=True)
            started = time.monotonic()
            while self.child.poll() is None:
                if monitor:
                    monitor()
                require(time.monotonic() - started < timeout, f'timed out: {log}')
                time.sleep(2)
            rc = self.child.wait()
            self.child = None
            require(rc == 0, f'command exited {rc}; see {log}')

    def stop_child(self):
        if self.child is not None and self.child.poll() is None:
            # SIGINT reaches the paired driver's finally; SIGTERM would bypass it.
            self.child.send_signal(signal.SIGINT)
            self.child.wait(timeout=360)
        self.child = None

    def restore(self):
        if not self.handoff:
            return
        require(not any(exe == 'reth-partial-stateless' for _, exe, _ in node_processes()),
                'producer is still running; do not start a second datadir owner')
        self.stage('restore-vanilla')
        # An interruption may arrive while the original node is still shutting down.
        deadline = time.monotonic() + 600
        while self.original and same_process(self.original):
            require(time.monotonic() < deadline, 'original node is still shutting down')
            time.sleep(2)
        self.run(['bash', RESTORE], 'restore.log', env=self.restore_env, timeout=420)
        owners = node_processes()
        require(len(owners) == 1 and owners[0][1] == 'reth', 'vanilla restore did not establish an owner')
        self.handoff = False

    def execute(self):
        args, base = self.args, self.base
        self.stage('preflight')
        preflight(args)
        selected = prefix_paths(args.spool)
        commit = git(args.repo, 'rev-parse', 'HEAD')
        self.env.update(PS_BUILD_COMMIT=commit, PS_BUILD_DIRTY='0',
                        PS_CARGO_LOCK_SHA256=sha256(args.repo / 'Cargo.lock'))
        (base / 'bin').mkdir()
        apparatus = base / 'apparatus'
        apparatus.mkdir()
        for name in ('run_live_paired_bench.py', 'analyze_builder_bench.py', 'analyze_validation_bench.py'):
            shutil.copy2(args.repo / 'crates/partial-stateless-exex/scripts' / name, apparatus / name)
        for package, binary, features in (
            ('partial-stateless-replay', 'ps-replay', ['--features', 'jemalloc']),
            ('partial-stateless-exex', 'reth-partial-stateless', []),
        ):
            self.stage(f'build-{binary}')
            self.run(['nice', '-n', '15', 'cargo', 'build', '--locked', '--offline', '--release',
                      '-j', args.jobs, '-p', package, '--bin', binary, *features], f'build-{binary}.log')
            require(git(args.repo, 'rev-parse', 'HEAD') == commit
                    and not git(args.repo, 'status', '--porcelain'), 'source changed during build')
            shutil.copy2(args.repo / 'target/release' / binary, base / 'bin' / binary)
        write_json(base / 'build.json', {'commit': commit, 'depth': args.depth, 'samples': args.samples,
                   'cargo_lock_sha256': self.env['PS_CARGO_LOCK_SHA256'],
                   'binaries': {p.name: sha256(p) for p in (base / 'bin').iterdir()},
                   'scripts': {p.name: sha256(p) for p in apparatus.iterdir()}})

        self.stage('replay-prefix')
        prefix = base / 'prefix'
        prefix.mkdir()
        for source in selected:
            try:
                os.link(source, prefix / source.name)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copy2(source, prefix / source.name)
        replay = base / 'bin/ps-replay'
        with (base / 'frames.jsonl').open('wb') as output:
            self.run([replay, '--list-frames', prefix], 'inventory.log', stdout=output)
        schedule = place_reorgs(load_jsonl(base / 'frames.jsonl'), args.depth)
        write_json(base / 'schedule.json', schedule)
        forced = [item for d, at in schedule for item in ('--forced-reorg', f'{d}@{at}')]
        self.stage('replay-depth-1-and-cap')
        self.run([replay, prefix, '--limit', REPLAY_BLOCKS, '--no-mutations',
                  '--undo-dir', base / 'replay-undo', *forced, '--json', base / 'replay.jsonl'],
                 'replay.log', timeout=3600)
        replay_result = judge_replay(load_jsonl(base / 'replay.jsonl'), schedule, args.depth, commit)
        check_undo_log(base / 'replay.log')
        write_json(base / 'replay-result.json', replay_result)

        self.stage('live-handoff')
        identity, _, argv = preflight(args)  # Recheck after the build/replay; the node stayed up.
        binary = Path(f'/proc/{identity["pid"]}/exe').resolve(strict=True)
        self.restore_env['RETH_BIN'] = str(binary)
        write_json(base / 'node-before.json', {'identity': identity, 'argv': argv, 'binary': str(binary)})
        self.original = identity
        self.handoff = True
        os.kill(identity['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 600
        while same_process(identity):
            require(time.monotonic() < deadline, 'vanilla did not stop within 600 seconds')
            time.sleep(2)
        require(not node_processes(), 'another node acquired the datadir during handoff')
        # Preserve the stale ExEx WAL instead of deleting it as the historical launcher did.
        if (DATADIR / 'exex').exists():
            (DATADIR / 'exex').rename(base / 'exex-wal-before')
        live_env = self.env | dict(PS_ACCOUNT_WINDOW='90', PS_STORAGE_WINDOW='60',
                                  PS_WITNESS_V3='1', PS_ENGINE_PAYLOAD='on',
                                  PS_UNDO_DIR=str(base / 'live-undo'))
        self.stage('live-paired')
        def observe():
            with (base / 'undo-files.jsonl').open('a') as output:
                output.write(json.dumps(undo_sample(base / 'live-undo')) + '\n')
        self.run([sys.executable, apparatus / 'run_live_paired_bench.py',
                  '--reth-bin', base / 'bin/reth-partial-stateless', '--datadir', DATADIR,
                  '--jwtsecret', JWT, '--output', base / 'paired', '--samples', args.samples,
                  '--warmup', '0', '--canonical-rebuild', 'off', '--parallel-initial-proof', 'off',
                  '--retain-generation', 'on', '--engine-access', 'on', '--shadow-sample', '50',
                  '--max-seconds', args.max_seconds, '--', '--minimal', '--http',
                  '--db.read-transaction-timeout', '0', '--http.api', 'eth,net,web3,debug,trace',
                  '--authrpc.addr', '127.0.0.1', '--authrpc.port', '8551',
                  '--ws', '--ws.addr', '127.0.0.1', '--ws.port', '8546',
                  '--ws.api', 'eth,trace,debug,net', '--ws.origins', 'localhost'],
                 'driver.log', env=live_env, timeout=args.max_seconds + 360, monitor=observe)
        self.restore()  # Restore before file-only analysis, as in the existing launcher.
        self.stage('judge')
        live_result = judge_live(base, args.samples, args.depth)
        write_json(base / 'result.json', {'live': live_result, 'replay': replay_result})


def worker(args):
    # Same host lock as phase3/60h. Popen(close_fds=True) prevents restored reth inheriting it.
    with (BENCH / '.zns4-phase3.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            (args.output / 'RESULT').write_text('FAIL another runner holds .zns4-phase3.lock\n')
            return 1
        runner = Runner(args)
        write_json(args.output / 'worker.json', process_identity(os.getpid()))
        def interrupt(_sig, _frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGTERM, interrupt)
        failure = None
        try:
            runner.execute()
        except (Exception, KeyboardInterrupt) as error:
            failure = f'{type(error).__name__}: {error}'
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                runner.stop_child()
            except Exception as error:
                failure = f'{failure or ""}; child shutdown failed: {error}'
            try:
                runner.restore()
            except Exception as error:
                failure = f'{failure or ""}; RESTORE FAILED: {error}; run {RESTORE} after producer exits'
        verdict = f'FAIL {failure}' if failure else 'PASS'
        (args.output / 'RESULT').write_text(verdict + '\n')
        runner.stage(verdict)
        return int(failure is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'start', 'status', 'stop', 'rejudge', '_worker'))
    parser.add_argument('output', nargs='?', type=Path)
    parser.add_argument('--repo', type=Path, default=Path('/data2/reth'))
    parser.add_argument('--spool', type=Path, default=BENCH / 'zns4-60h-10000-20260902-070355/spool')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--depth', type=int, choices=(32, 64), default=32)
    parser.add_argument('--max-seconds', type=int, default=21600)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    require(args.samples > 2 * args.depth and args.max_seconds > 0 and args.jobs > 0,
            'need samples > 2K, positive max-seconds and build jobs')
    args.repo, args.spool = args.repo.resolve(), args.spool.resolve()
    if args.output:
        args.output = args.output.resolve()
    if args.action == 'check':
        print(f'main -> stamped release builds -> {REPLAY_BLOCKS} replay commits (depth 1/{args.depth})'
              f' -> {args.samples} live paired samples -> vanilla restore', flush=True)
        print(f'profile: 90/60, v3, Exact, K={args.depth}, frames, all undo on disk', flush=True)
        preflight(args)
        print(f'PASS: {len(prefix_paths(args.spool))} prefix files; check made no changes')
        return 0
    if args.action == 'start':
        preflight(args)
        prefix_paths(args.spool)
        args.output = args.output or BENCH / time.strftime('disk-undo-%Y%m%d-%H%M%S', time.gmtime())
        args.output.mkdir()  # Refuse reuse; every result belongs to this invocation.
        command = [sys.executable, Path(__file__).resolve(), '_worker', args.output, '--repo', args.repo,
                   '--spool', args.spool, '--samples', args.samples, '--depth', args.depth,
                   '--max-seconds', args.max_seconds, '--jobs', args.jobs]
        with (args.output / 'run.log').open('wb') as output:
            process = subprocess.Popen([str(v) for v in command], stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        for _ in range(30):
            require(process.poll() is None, f'worker exited; see {args.output}/run.log and RESULT')
            if (args.output / 'worker.json').exists():
                break
            time.sleep(0.1)
        require((args.output / 'worker.json').exists(), f'worker startup pending; inspect {args.output}')
        print(f'Started: {args.output}\nWatch: tail -f {args.output}/run.log')
        return 0
    require(args.output is not None, 'this action requires RUN_DIR')
    if args.action == 'rejudge':
        return rejudge(args.output, args.repo)
    if args.action == '_worker':
        return worker(args)
    identity_path = args.output / 'worker.json'
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else None
    if args.action == 'stop':
        require(identity is not None and same_process(identity), 'worker is no longer running; no signal sent')
        os.kill(identity['pid'], signal.SIGINT)
        print('Graceful stop requested; worker will stop its child and restore vanilla.')
    else:
        print(f'running={identity is not None and same_process(identity)}')
        for name in ('STATUS', 'RESULT', 'REJUDGED_RESULT'):
            if (args.output / name).exists():
                print(f'{name}: {(args.output / name).read_text().strip()}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (Exception, KeyboardInterrupt) as error:
        sys.exit(f'{type(error).__name__}: {error}')
