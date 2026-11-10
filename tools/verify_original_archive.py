#!/usr/bin/env python3
"""Verify every original project file against the extraction inventory."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    original = ROOT / 'original_project'
    with (ROOT / 'docs/archive_inventory.csv').open(newline='', encoding='utf-8') as handle:
        expected = list(csv.DictReader(handle))
    expected_paths = {row['path'] for row in expected}
    actual_paths = {str(path.relative_to(original)) for path in original.rglob('*') if path.is_file()}
    if expected_paths != actual_paths:
        raise RuntimeError(f'Original file set berbeda: missing={len(expected_paths - actual_paths)}, extra={len(actual_paths - expected_paths)}')
    total_bytes = 0
    for row in expected:
        path = original / row['path']
        if path.stat().st_size != int(row['bytes']) or sha256_file(path) != row['sha256']:
            raise RuntimeError(f'Original file checksum berbeda: {row["path"]}')
        total_bytes += path.stat().st_size
    print(json.dumps({'verification': 'passed', 'original_files': len(expected), 'original_bytes': total_bytes}, indent=2))


if __name__ == '__main__':
    main()
