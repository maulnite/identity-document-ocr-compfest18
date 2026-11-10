#!/usr/bin/env python3
"""Recompute aggregate metrics locally without printing identity values."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline'))
import identity_ocr_compfest18 as core

OFFICIAL_TEST_SHA256 = 'd30030f546124d12c6c7e514b8dc702b3631dcf209f8b00a91532400aac60624'


def main() -> None:
    summary = {}
    for version, folder in [
        ('v27', 'results_v27_fuzzy_peer_medoid_names'),
        ('v28', 'results_v28_generic_unknown_peer_consensus'),
    ]:
        directory = ROOT / 'pipeline' / folder
        frame = pd.read_parquet(directory / 'predictions_released.parquet')
        assert len(frame) == 732 and frame.filename.nunique() == 732
        labeled = frame.loc[frame.has_ground_truth.astype(bool)].copy()
        assert len(labeled) == 632
        scored = core.score_prediction_rows(labeled)
        expected = json.loads((ROOT / f'results/aggregate/{version}_metrics.json').read_text())
        metrics = {
            'Name EM': float(scored.name_exact.mean()),
            'DOB EM': float(scored.dob_exact.mean()),
            'Address EM': float(scored.address_exact.mean()),
            'Macro Field EM': float(scored.macro_field_exact.mean()),
            'Full Record Accuracy': float(scored.full_record_correct.mean()),
            'CER': float(scored.record_cer.mean()),
            'Normalized Edit Similarity': float(scored.normalized_edit_similarity.mean()),
        }
        for key, value in metrics.items():
            assert abs(value - expected[key]) < 1e-10, (version, key, value, expected[key])
        present = scored.loc[scored.address_present_gt]
        assert len(present) == 112
        assert abs(float(present.address_exact.mean()) - 0.25) < 1e-10
        assert hashlib.sha256((directory / 'test_predictions.csv').read_bytes()).hexdigest() == OFFICIAL_TEST_SHA256
        summary[version] = {'labeled_rows': len(scored), 'metrics_match': True, 'address_present_rows': len(present), 'official_test_hash_match': True}
    test = pd.read_csv(ROOT / 'pipeline/results/test_predictions.csv', keep_default_na=False)
    assert list(test.columns) == ['filename', 'name', 'birth_date', 'address']
    assert len(test) == 100 and test.filename.nunique() == 100
    assert hashlib.sha256((ROOT / 'pipeline/results/test_predictions.csv').read_bytes()).hexdigest() == OFFICIAL_TEST_SHA256
    print(json.dumps({'verification': 'passed', 'versions': summary}, indent=2))


if __name__ == '__main__':
    main()
