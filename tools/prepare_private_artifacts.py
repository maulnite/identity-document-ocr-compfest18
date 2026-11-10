#!/usr/bin/env python3
"""Restore frozen inputs from the complete original project archived in this repo.

No OCR, package installation, network operation, or data upload is performed.
"""
from __future__ import annotations

import argparse
import ast
import base64
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def copy_identical_or_new(source: Path, target: Path) -> None:
    if target.exists():
        if target.read_bytes() != source.read_bytes():
            raise FileExistsError(f"Target berbeda; periksa secara manual: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'original_project', help='Default: original_project/ dalam repository; dapat diarahkan ke ekstraksi lain')
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    bundle = source / 'v27-28/compfest/compfest_academy_project/V27_REPRODUCIBLE_BUNDLE'
    if not (bundle / 'run_v27.py').is_file():
        raise SystemExit('Ekstrak v27-28.zip ke subfolder v27-28/ terlebih dahulu (lihat docs/REPRODUCIBILITY.md).')
    # Restore working inputs and reference outputs; never replace source code.
    for folder in ['results', 'results_v3_stable', 'results_v17_per_layout_generic_mrz',
                   'results_v22_final_permit_ukraine_consensus', 'reference_outputs']:
        for path in sorted((bundle / folder).rglob('*')):
            if path.is_file():
                copy_identical_or_new(path, ROOT / 'pipeline' / path.relative_to(bundle))
    (ROOT / 'pipeline/results_v28_generic_unknown_peer_consensus').mkdir(exist_ok=True)
    (ROOT / 'pipeline/reports').mkdir(exist_ok=True)
    for path in (bundle / 'reports').glob('*.csv'):
        copy_identical_or_new(path, ROOT / 'pipeline/reports' / path.name)
    nb_path = source / 'output/Notebook_ketolak pacil_COMPFEST18.ipynb'
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    for cell in nb['cells']:
        text = ''.join(cell.get('source', []))
        if '_BUNDLE_B64 = ' not in text:
            continue
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_BUNDLE_B64' for t in node.targets):
                payload = base64.b64decode(ast.literal_eval(node.value), validate=True)
                dest = ROOT / 'data/private/embedded_evidence.zip'
                if dest.exists() and dest.read_bytes() != payload:
                    raise FileExistsError(f'Target berbeda: {dest}')
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(payload)
                print('Frozen inputs dan notebook evidence siap dari original archive. Tidak ada upload tambahan.')
                return
    raise SystemExit('Embedded evidence tidak ditemukan di notebook asal.')


if __name__ == '__main__':
    main()
