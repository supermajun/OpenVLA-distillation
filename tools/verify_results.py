"""Check the portable publication data; no models, downloads or GPU required."""
import csv
import json
import math
from pathlib import Path


def verify(root):
    results = json.loads((root / 'results/main_results.json').read_text())['results']
    with (root / 'results/main_results.csv').open(newline='') as f:
        table = list(csv.DictReader(f))
    expected = {(m, n) for m in ('500m', '1b', '2b') for n in (500, 1000, 2000)}
    keys = [(r['model'], r['updates']) for r in results]
    assert len(keys) == len(set(keys)) == 9 and set(keys) == expected
    assert len(table) == 9
    by_key = {(r['model'], int(r['updates'])): r for r in table}
    assert len(by_key) == 9 and set(by_key) == expected
    for r in results:
        episodes = r['rollouts']
        assert r['episodes'] == len(episodes) == 8
        assert [e['initial_state_index'] for e in episodes] == list(range(8))
        assert all(e['seed'] == 17 and type(e['success']) is bool for e in episodes)
        assert r['successes'] == sum(e['success'] for e in episodes)
        assert r['val_samples'] == 80 and r['adapters_removed'] is True
        assert all(0 < e['steps'] <= 220 and e['replans'] > 0 for e in episodes)
        row = by_key[r['model'], r['updates']]
        for key, value in row.items():
            assert value == str(r[key]), (key, value, r[key])
        for key in ('validation_l1', 'model_forward_latency_ms', 'policy_file_bytes'):
            assert math.isfinite(r[key]) and r[key] > 0
    for model in ('500m', '1b', '2b'):
        p = root / f'results/training/{model}_metrics.jsonl'
        records = [json.loads(line) for line in p.read_text().splitlines()]
        assert [r['step'] for r in records] == list(range(1, 2001))
        for r in records:
            for key in ('demo', 'action', 'feature', 'total', 'grad_norm', 'seconds'):
                assert math.isfinite(r[key]) and r[key] >= 0
    return 'PASS: 9 checkpoints, 72 paired-state outcomes, CSV agreement, 6,000 training updates.'


if __name__ == '__main__':
    print(verify(Path(__file__).resolve().parents[1]))
