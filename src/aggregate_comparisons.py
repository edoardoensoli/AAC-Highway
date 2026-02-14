#!/usr/bin/env python3
"""
Aggregate comparison_summary.json from test_200k, test_500k, test_1M and produce summary tables.

Outputs (directory `comparison_reports/`):
- overall_summary.csv / .md : per scenario+model shows survival/avg_reward for 200k/500k/1M
- test_200k_summary.md/csv : per-scenario tables comparing models inside test_200k
- test_500k_summary.md/csv
- test_1M_summary.md/csv

Usage: python aggregate_comparisons.py
"""
from pathlib import Path
import json
import csv
import sys

ROOT = Path(__file__).resolve().parent
RUN_DIRS = [
    ("200k", ROOT / 'test_200k' / 'comparison_summary.json'),
    ("500k", ROOT / 'test_500k' / 'comparison_summary.json'),
    ("1M",   ROOT / 'test_1M' / 'comparison_summary.json'),
]
OUT = ROOT / 'comparison_reports'
OUT.mkdir(exist_ok=True)

# Load available jsons
data = {}
for suffix, path in RUN_DIRS:
    if path.exists():
        try:
            data[suffix] = json.loads(path.read_text())
        except Exception as e:
            print(f"Could not parse {path}: {e}")
            data[suffix] = None
    else:
        print(f"Warning: {path} not found")
        data[suffix] = None

# Collect scenarios and models
scenarios = set()
models = set()
for suffix, payload in data.items():
    if not payload:
        continue
    res = payload.get('results', {})
    scenarios.update(res.keys())
    for sc, scdict in res.items():
        for m in scdict.keys():
            models.add(m)

scenarios = sorted(scenarios)
models = sorted(models)

# Build overall summary rows
overall_rows = []
for sc in scenarios:
    for m in models:
        row = {
            'scenario': sc,
            'model': m,
        }
        for suffix, _path in RUN_DIRS:
            surv = None
            avg = None
            payload = data.get(suffix)
            if payload and 'results' in payload:
                sc_dict = payload['results'].get(sc, {})
                m_dict = sc_dict.get(m)
                if m_dict:
                    surv = m_dict.get('survival_rate')
                    avg = m_dict.get('avg_reward')
            row[f'survival_{suffix}'] = surv
            row[f'avg_reward_{suffix}'] = avg
        overall_rows.append(row)

# Write overall CSV
overall_csv = OUT / 'overall_summary.csv'
fields = ['scenario', 'model'] + [f for suffix in [s for s,_ in RUN_DIRS] for f in (f'survival_{suffix}', f'avg_reward_{suffix}')]
with overall_csv.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in overall_rows:
        writer.writerow(r)

# Write overall MD
overall_md = OUT / 'overall_summary.md'
with overall_md.open('w', encoding='utf-8') as f:
    f.write('# Overall Comparison: survival and avg_reward (200k / 500k / 1M)\n\n')
    for sc in scenarios:
        f.write(f'## Scenario: {sc}\n\n')
        f.write('| Model | Survival (200k) | AvgReward (200k) | Survival (500k) | AvgReward (500k) | Survival (1M) | AvgReward (1M) |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|\n')
        for m in models:
            vals = []
            for suffix, _ in RUN_DIRS:
                payload = data.get(suffix)
                surv = ''
                avg = ''
                if payload and 'results' in payload:
                    sc_dict = payload['results'].get(sc, {})
                    m_dict = sc_dict.get(m)
                    if m_dict:
                        surv = m_dict.get('survival_rate', '')
                        avg = m_dict.get('avg_reward', '')
                vals.extend([str(surv), str(avg)])
            f.write(f'| {m} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {vals[4]} | {vals[5]} |\n')
        f.write('\n')

# Per-folder summaries
for suffix, path in RUN_DIRS:
    payload = data.get(suffix)
    if not payload:
        continue
    out_md = OUT / f'test_{suffix}_summary.md'
    out_csv = OUT / f'test_{suffix}_summary.csv'
    results = payload.get('results', {})

    # CSV: rows = scenario, columns: model|survival|avg_reward (flattened)
    csv_fields = ['scenario'] + [f'{m}_survival' for m in models] + [f'{m}_avg_reward' for m in models]
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for sc in scenarios:
            row = {'scenario': sc}
            sc_dict = results.get(sc, {})
            for m in models:
                m_dict = sc_dict.get(m, {})
                row[f'{m}_survival'] = m_dict.get('survival_rate', '')
                row[f'{m}_avg_reward'] = m_dict.get('avg_reward', '')
            writer.writerow(row)

    # MD: nice tables per scenario
    with out_md.open('w', encoding='utf-8') as f:
        f.write(f'# Summary for {suffix}\n\n')
        f.write(f"Models: {', '.join(models)}\n\n")
        for sc in scenarios:
            f.write(f'## Scenario: {sc}\n\n')
            f.write('| Model | Survival % | Avg Reward |\n')
            f.write('|---|---:|---:|\n')
            sc_dict = results.get(sc, {})
            for m in models:
                m_dict = sc_dict.get(m, {})
                surv = '' if m_dict is None else m_dict.get('survival_rate', '')
                avg = '' if m_dict is None else m_dict.get('avg_reward', '')
                f.write(f'| {m} | {surv} | {avg} |\n')
            f.write('\n')

print(f'Reports written to: {OUT}')
print('Overall CSV:', overall_csv)
print('Overall MD:', overall_md)
print('Per-folder summaries created.')
