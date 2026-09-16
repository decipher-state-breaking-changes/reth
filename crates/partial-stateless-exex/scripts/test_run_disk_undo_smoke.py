#!/usr/bin/env python3
"""Acceptance gates and node cleanup, using fixtures/mocks without launching a node."""
import copy
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_disk_undo_smoke as run
from analyze_validation_bench import SelectionStats


def inventory():
    rows = [{'sequence': 0, 'kind': 'manifest'},
            {'sequence': 1, 'kind': 'checkpoint', 'block': {'number': 100, 'hash': 'h100'}}]
    for n in range(101, 101 + run.REPLAY_BLOCKS):
        rows.append({'sequence': len(rows), 'kind': 'commit',
                     'block': {'number': n, 'hash': f'h{n}'}, 'parent_hash': f'h{n - 1}'})
    return rows


def replay_fixture(depth=32):
    schedule = run.place_reorgs(inventory(), depth)
    manifest = {'kind': 'run_manifest', 'retain_depth': depth, 'undo_layout': 'frames',
                'undo_record': True, 'undo_dir': '/tmp/undo', 'allocator': 'jemalloc',
                'provenance': {'build_commit': 'commit', 'build_dirty': False},
                'forced_reorgs': [f'{d}@{at}' for d, at in schedule]}
    report = dict.fromkeys(('failures', 'disagreements', 'mutation_failures',
                           'winning_branch_incomplete', 'skipped_after_fault', 'skipped_awaiting_resync'), 0)
    report.update(agreed=True, continuous=True, complete=True, commits=run.REPLAY_BLOCKS, resyncs=[])
    report['blocks'] = [{'verdict': 'accepted', 'undo_resident_blocks': 0}
                        for _ in range(run.REPLAY_BLOCKS + depth + 1)]
    report['forced_reorgs'] = [dict(depth=d, at=at, outcome='applied', frames_applied=d,
                                  reapplied=d, resumed_identical=True, ancestor={'number': at - d},
                                  reapplied_sequences=list(range(d))) for d, at in schedule]
    return [manifest, report], schedule


def cold_start_log():
    # Keep the production event ordering: commit observation precedes readiness publication.
    return '\n'.join([
        'Partial Stateless ExEx started',
        'Cache is not synced to the parent block. block=100 cache_block=0 expected_parent_block=99',
        'Undo coverage lost; discarding history behind an unspillable block '
        'cause="full_fallback" block=100 dropped_generations=1 retained_depth=0 configured_depth=32',
        'Observed cache undo retention after commit block=100 retained_depth=0 resident_blocks=0',
        'Cache readiness changed block=100 from="cold" to="warming" replay_depth=1',
        'Observed cache undo retention after commit block=101 retained_depth=1 resident_blocks=0',
    ])


class AcceptanceTests(unittest.TestCase):
    def test_old_cold_start_requires_explicit_compatibility_and_correlated_first_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'log'
            path.write_text(cold_start_log())
            with self.assertRaisesRegex(RuntimeError, 'undo warnings'):
                run.check_undo_log(path)
            _, initial = run.check_undo_log(path, allow_legacy_cold_start=True)
            self.assertEqual(initial, [100])
            variants = [
                cold_start_log().replace('cache_block=0', 'cache_block=99'),
                cold_start_log().replace('from="cold"', 'from="warming"'),
                cold_start_log().replace('dropped_generations=1', 'dropped_generations=2'),
                cold_start_log().replace('cause="full_fallback"', 'cause="missing_flat_record"'),
                cold_start_log().replace('block=101', 'block=102'),
                cold_start_log().replace('replay_depth=1', 'replay_depth=2'),
                cold_start_log().replace('Cache readiness changed', 'missing event'),
                cold_start_log() + '\nPartial Stateless ExEx started',
                'Observed cache undo retention after commit block=99 retained_depth=1 '
                'resident_blocks=0\n' + cold_start_log(),
            ]
            for text in variants:
                with self.subTest(text=text), self.assertRaisesRegex(RuntimeError, 'undo warnings'):
                    path.write_text(text)
                    run.check_undo_log(path, allow_legacy_cold_start=True)

    def test_initialization_never_hides_later_coverage_loss_or_disk_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'log'
            for warning in run.BAD_UNDO:
                path.write_text(cold_start_log() + '\n' + warning + ' block=102')
                with self.subTest(warning=warning), self.assertRaisesRegex(RuntimeError, 'undo warnings'):
                    run.check_undo_log(path, allow_legacy_cold_start=True)
            path.write_text('Initialized disk undo history; no recoverable parent '
                            'cause="cold_start" block=100 retained_depth=0 configured_depth=32')
            self.assertEqual(run.check_undo_log(path)[1], [])

    def test_rejudge_preserves_original_result_and_requires_restoration_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            rows, schedule = replay_fixture()
            run.write_json(base / 'build.json', {'commit': 'commit', 'depth': 32, 'samples': 1000})
            run.write_json(base / 'schedule.json', schedule)
            for name, records in [('frames.jsonl', inventory()), ('replay.jsonl', rows)]:
                (base / name).write_text(''.join(json.dumps(row) + '\n' for row in records))
            (base / 'replay.log').write_text('')
            original = 'FAIL RuntimeError: undo warnings\n'
            (base / 'RESULT').write_text(original)
            (base / 'restore.log').write_text('restore: vanilla reth relaunched, pid 123\n'
                                             'restore: pid 123 alive after 15s\n')
            with patch.object(run, 'git', return_value='judge-commit'), \
                 patch.object(run, 'judge_live', return_value={'accepted': 1000}):
                self.assertEqual(run.rejudge(base, base), 0)
                result = json.loads((base / 'rejudged-result.json').read_text())
                self.assertEqual(result['run_commit'], 'commit')
                self.assertEqual(result['judge_commit'], 'judge-commit')
                self.assertEqual((base / 'RESULT').read_text(), original)
                (base / 'restore.log').write_text('restore: vanilla reth relaunched, pid 123\n')
                self.assertEqual(run.rejudge(base, base), 1)
                self.assertEqual((base / 'REJUDGED_RESULT').read_text(), 'FAIL\n')
                self.assertEqual((base / 'RESULT').read_text(), original)

    def test_replay_accepts_both_caps_and_refuses_missed_or_corrupt_recovery(self):
        for depth in (32, 64):
            rows, schedule = replay_fixture(depth)
            run.judge_replay(rows, schedule, depth, 'commit')
            for field, value in [('outcome', 'not_reached'), ('frames_applied', depth - 1),
                                 ('resumed_identical', False), ('reapplied', 0)]:
                bad = copy.deepcopy(rows)
                bad[1]['forced_reorgs'][1][field] = value
                with self.subTest(depth=depth, field=field), self.assertRaises(RuntimeError):
                    run.judge_replay(bad, schedule, depth, 'commit')
            rows[1]['blocks'][0]['undo_resident_blocks'] = None
            with self.assertRaisesRegex(RuntimeError, 'resident'):
                run.judge_replay(rows, schedule, depth, 'commit')

    def test_inventory_refuses_a_changed_parent_or_intervening_lifecycle(self):
        rows = inventory()
        rows[80]['parent_hash'] = 'other-branch'
        with self.assertRaisesRegex(RuntimeError, 'parent chain'):
            run.place_reorgs(rows, 32)
        rows = inventory()
        rows.insert(80, {'kind': 'reorg'})
        for seq, row in enumerate(rows):
            row['sequence'] = seq
        with self.assertRaisesRegex(RuntimeError, 'lifecycle'):
            run.place_reorgs(rows, 32)

    def test_live_deadline_and_invalid_filtered_samples_are_not_success(self):
        with patch.object(run, 'load_jsonl', return_value=[{'valid': True}]), \
             patch.object(run, 'select_samples', return_value=([{}] * 999, SelectionStats())):
            with self.assertRaisesRegex(RuntimeError, '999/1000'):
                run.judge_live(Path('/unused'), 1000, 32)
        with patch.object(run, 'load_jsonl', return_value=[{'valid': False}]), \
             patch.object(run, 'select_samples', return_value=([{}] * 1000, SelectionStats())):
            with self.assertRaisesRegex(RuntimeError, 'invalid paired'):
                run.judge_live(Path('/unused'), 1000, 32)

    def test_file_rotation_allows_finishing_write_but_rejects_growth_or_missing_evidence(self):
        def row(start, count, name='undo-session'):
            return {'sessions': {name: {'files': list(range(start, start + count)), 'temporary': 0}}}
        run.judge_rotation([row(1, 32), row(2, 33), row(3, 32)], 32)
        for samples in ([row(1, 32), row(1, 34)], [row(1, 32)],
                        [row(1, 32), row(2, 32, 'new-session')]):
            with self.assertRaises(RuntimeError):
                run.judge_rotation(samples, 32)

    def test_environment_cannot_inherit_old_profiles_or_a_capture(self):
        with patch.dict(run.os.environ, {'PS_UNDO_LAYOUT': 'hybrid', 'PS_TRIE_REPR': 'parallel',
                                       'PS_UNDO_RECORD': 'off', 'PS_STREAM_DIR': '/unwanted',
                                       'PS_FORCE_PREVIOUS_CACHE_SNAPSHOT': '1', 'MALLOC_CONF': 'old'}):
            env = run.clean_env(32)
        self.assertEqual(env['PS_UNDO_LAYOUT'], 'frames')
        self.assertEqual(env['PS_TRIE_REPR'], 'exact')
        self.assertEqual(env['PS_UNDO_RECORD'], 'on')
        self.assertNotIn('PS_STREAM_DIR', env)
        self.assertNotIn('PS_FORCE_PREVIOUS_CACHE_SNAPSHOT', env)
        self.assertNotIn('MALLOC_CONF', env)

    def test_worker_restores_after_failure_and_interruption(self):
        for error in (RuntimeError('live failed'), KeyboardInterrupt()):
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                order = []
                runner = Mock()
                runner.execute.side_effect = error
                runner.stop_child.side_effect = lambda: order.append('stop child')
                runner.restore.side_effect = lambda: order.append('restore vanilla')
                with patch.object(run, 'BENCH', base), patch.object(run, 'Runner', return_value=runner), \
                     patch.object(run.signal, 'signal'):
                    self.assertEqual(run.worker(SimpleNamespace(output=base)), 1)
                self.assertEqual(order, ['stop child', 'restore vanilla'])
                self.assertTrue((base / 'RESULT').read_text().startswith('FAIL'))

    def test_restore_failure_cannot_be_reported_as_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            runner = Mock()
            runner.restore.side_effect = RuntimeError('restore failed')
            with patch.object(run, 'BENCH', base), patch.object(run, 'Runner', return_value=runner), \
                 patch.object(run.signal, 'signal'):
                self.assertEqual(run.worker(SimpleNamespace(output=base)), 1)
            self.assertIn('RESTORE FAILED', (base / 'RESULT').read_text())

    def test_stop_uses_sigint_for_driver_cleanup(self):
        runner = run.Runner(SimpleNamespace(output=Path('/unused'), depth=32))
        child = Mock()
        child.poll.return_value = None
        runner.child = child
        runner.stop_child()
        child.send_signal.assert_called_once_with(signal.SIGINT)
        child.wait.assert_called_once_with(timeout=360)

    def test_restore_waits_for_original_shutdown_before_calling_existing_hook(self):
        runner = run.Runner(SimpleNamespace(output=Path('/unused'), depth=32))
        runner.handoff = True
        runner.original = {'pid': 123, 'start_ticks': '456'}
        order = []
        runner.stage = Mock()
        runner.run = Mock(side_effect=lambda *a, **k: order.append('restore hook'))
        with patch.object(run, 'same_process', side_effect=[True, False]), \
             patch.object(run.time, 'sleep', side_effect=lambda _: order.append('wait')), \
             patch.object(run, 'node_processes', side_effect=[[], [(None, 'reth', [])]]):
            runner.restore()
        self.assertEqual(order, ['wait', 'restore hook'])
        self.assertFalse(runner.handoff)

    def test_stale_pid_is_not_a_matching_worker(self):
        old = {'pid': 123, 'start_ticks': '456'}
        with patch.object(run, 'process_identity', return_value={'pid': 123, 'start_ticks': '789'}):
            self.assertFalse(run.same_process(old))


if __name__ == '__main__':
    unittest.main()
