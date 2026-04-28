"""Task #56 validation tests that do NOT require torch.

Validates:
  - CLI parser defaults for score_lambda, wf_train_months, recency_half_life, short_min_fraction
  - V5_FOLD_HEALTH and V5_THRESHOLD_GUIDE log blocks are present in v5_train.py
    with correct VERDICT states and --v5-min-threshold CLI flag
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_task56_cli_parser_defaults():
    """Task #56 A1/B3/C1/C2: quick_start.py CLI defaults match the task spec.

    - score_lambda=0.30 (A1: break-even p_side 0.333→0.231)
    - wf_train_months=9  (C2: 12→9 for more folds over history)
    - recency_half_life=60 (C1: 90→60 for faster regime adaptation)
    - short_min_fraction=0.40 (B3: 0.35→0.40 to fix SHORT under-representation)
    """
    qs_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'quick_start.py')
    )
    assert os.path.exists(qs_path), f"quick_start.py not found at {qs_path}"

    with open(qs_path, 'r') as f:
        src = f.read()

    assert re.search(r'v5-score-lambda.*?default\s*=\s*0\.30', src, re.DOTALL), \
        "CLI --v5-score-lambda must default to 0.30 (Task #56 A1)"

    assert re.search(r'v5-wf-train-months.*?default\s*=\s*9[^0-9]', src, re.DOTALL), \
        "CLI --v5-wf-train-months must default to 9 (Task #56 C2)"

    assert re.search(r'v5-recency-half-life.*?default\s*=\s*60[^0-9]', src, re.DOTALL), \
        "CLI --v5-recency-half-life must default to 60 (Task #56 C1)"

    assert re.search(r'v5-short-min-fraction.*?default\s*=\s*0\.40', src, re.DOTALL), \
        "CLI --v5-short-min-fraction must default to 0.40 (Task #56 B3)"


def test_task56_fold_health_and_threshold_guide_in_source():
    """Task #56 D1/D2: v5_train.py contains [V5_FOLD_HEALTH] and [V5_THRESHOLD_GUIDE] blocks.

    D1 must include LIVE_READY/COLLAPSED verdicts and expected_daily_R.
    D2 must recommend --v5-min-threshold (NOT --v5-score-threshold).
    """
    v5_train_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'train', 'v5_train.py')
    )
    assert os.path.exists(v5_train_path), f"v5_train.py not found at {v5_train_path}"

    with open(v5_train_path, 'r') as f:
        src = f.read()

    # D1: fold health block
    assert '[V5_FOLD_HEALTH]' in src, \
        "[V5_FOLD_HEALTH] not in v5_train.py — D1 missing"
    assert 'LIVE_READY' in src, \
        "LIVE_READY verdict missing — D1 incomplete"
    assert 'COLLAPSED' in src, \
        "COLLAPSED verdict missing — D1 incomplete"
    assert 'expected_daily_R' in src, \
        "expected_daily_R missing from D1 fold health block"

    # D2: threshold guide block
    assert '[V5_THRESHOLD_GUIDE]' in src, \
        "[V5_THRESHOLD_GUIDE] not in v5_train.py — D2 missing"

    # Extract the THRESHOLD_GUIDE section from all occurrences and check the log.info f-string
    # There are multiple occurrences: comment, f-string header, f-string body.
    # Join them all and search for --v5-min-threshold (found in the log.info body).
    tg_all = ''.join(src.split('[V5_THRESHOLD_GUIDE]')[1:]).split('agg_report')[0]
    assert '--v5-min-threshold' in tg_all, \
        "D2 must recommend --v5-min-threshold (not --v5-score-threshold)"
    assert '--v5-score-threshold' not in tg_all, \
        "D2 must NOT recommend --v5-score-threshold (wrong flag); use --v5-min-threshold"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
