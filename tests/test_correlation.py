"""Tests for v5.0.8 cross-asset correlation tracking & smart blocking."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np


class TestRollingDailyCorr(unittest.TestCase):

    def test_perfect_positive_correlation(self):
        from train.v5_correlation import RollingDailyCorr
        tracker = RollingDailyCorr(['BTCUSDT', 'ETHUSDT'], window_days=30)
        for i in range(20):
            d = f"2024-01-{i+1:02d}"
            tracker.record_trade('BTCUSDT', d, float(i))
            tracker.record_trade('ETHUSDT', d, float(i) * 2)

        corr = tracker.pairwise_corr('BTCUSDT', 'ETHUSDT')
        self.assertIsNotNone(corr)
        self.assertAlmostEqual(corr, 1.0, places=4)

    def test_perfect_negative_correlation(self):
        from train.v5_correlation import RollingDailyCorr
        tracker = RollingDailyCorr(['BTCUSDT', 'ETHUSDT'], window_days=30)
        for i in range(20):
            d = f"2024-01-{i+1:02d}"
            tracker.record_trade('BTCUSDT', d, float(i))
            tracker.record_trade('ETHUSDT', d, -float(i))

        corr = tracker.pairwise_corr('BTCUSDT', 'ETHUSDT')
        self.assertIsNotNone(corr)
        self.assertAlmostEqual(corr, -1.0, places=4)

    def test_insufficient_data_returns_none(self):
        from train.v5_correlation import RollingDailyCorr
        tracker = RollingDailyCorr(['BTCUSDT', 'ETHUSDT'],
                                   window_days=30, min_aligned_days=10)
        for i in range(5):
            d = f"2024-01-{i+1:02d}"
            tracker.record_trade('BTCUSDT', d, float(i))
            tracker.record_trade('ETHUSDT', d, float(i))

        corr = tracker.pairwise_corr('BTCUSDT', 'ETHUSDT')
        self.assertIsNone(corr)

    def test_correlation_matrix_three_symbols(self):
        from train.v5_correlation import RollingDailyCorr
        syms = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        tracker = RollingDailyCorr(syms, window_days=30)
        np.random.seed(42)
        for i in range(20):
            d = f"2024-01-{i+1:02d}"
            base = np.random.randn()
            tracker.record_trade('BTCUSDT', d, base)
            tracker.record_trade('ETHUSDT', d, base + 0.1 * np.random.randn())
            tracker.record_trade('SOLUSDT', d, -base + 0.1 * np.random.randn())

        mat, labels = tracker.correlation_matrix()
        self.assertEqual(labels, syms)
        self.assertEqual(mat.shape, (3, 3))
        for i in range(3):
            self.assertAlmostEqual(mat[i, i], 1.0, places=4)
        self.assertAlmostEqual(mat[0, 1], mat[1, 0], places=4)

    def test_daily_r_aggregation(self):
        from train.v5_correlation import RollingDailyCorr
        tracker = RollingDailyCorr(['BTCUSDT'], window_days=30)
        tracker.record_trade('BTCUSDT', '2024-01-01', 1.0)
        tracker.record_trade('BTCUSDT', '2024-01-01', -0.5)
        tracker.record_trade('BTCUSDT', '2024-01-01', 0.3)

        self.assertAlmostEqual(tracker.daily_r['BTCUSDT']['2024-01-01'], 0.8, places=4)

    def test_nan_trade_ignored(self):
        from train.v5_correlation import RollingDailyCorr
        tracker = RollingDailyCorr(['BTCUSDT'], window_days=30)
        tracker.record_trade('BTCUSDT', '2024-01-01', 1.0)
        tracker.record_trade('BTCUSDT', '2024-01-01', float('nan'))
        self.assertAlmostEqual(tracker.daily_r['BTCUSDT']['2024-01-01'], 1.0)

    def test_window_limits_data(self):
        from train.v5_correlation import RollingDailyCorr
        tracker = RollingDailyCorr(['BTCUSDT', 'ETHUSDT'],
                                   window_days=5, min_aligned_days=3)
        for i in range(20):
            d = f"2024-01-{i+1:02d}"
            tracker.record_trade('BTCUSDT', d, float(i))
            tracker.record_trade('ETHUSDT', d, float(i) * 2)

        sa, sb = tracker.get_aligned_daily_series('BTCUSDT', 'ETHUSDT', max_days=5)
        self.assertEqual(len(sa), 5)


class TestCorrBlocker(unittest.TestCase):

    def test_blocks_high_corr_same_side(self):
        from train.v5_correlation import RollingDailyCorr, CorrBlocker, CorrConfig
        tracker = RollingDailyCorr(['BTCUSDT', 'ETHUSDT'], window_days=30)
        for i in range(15):
            d = f"2024-01-{i+1:02d}"
            tracker.record_trade('BTCUSDT', d, float(i))
            tracker.record_trade('ETHUSDT', d, float(i) * 1.5)

        cfg = CorrConfig(enabled=True, threshold=0.60, same_side_only=True)
        blocker = CorrBlocker(tracker, cfg)

        open_pos = {'BTCUSDT': 1}
        result = blocker.should_block('ETHUSDT', 1, open_pos)
        self.assertTrue(result)
        self.assertEqual(blocker.blocked_count, 1)

    def test_allows_different_side_when_same_side_only(self):
        from train.v5_correlation import RollingDailyCorr, CorrBlocker, CorrConfig
        tracker = RollingDailyCorr(['BTCUSDT', 'ETHUSDT'], window_days=30)
        for i in range(15):
            d = f"2024-01-{i+1:02d}"
            tracker.record_trade('BTCUSDT', d, float(i))
            tracker.record_trade('ETHUSDT', d, float(i) * 1.5)

        cfg = CorrConfig(enabled=True, threshold=0.60, same_side_only=True)
        blocker = CorrBlocker(tracker, cfg)

        open_pos = {'BTCUSDT': 1}
        result = blocker.should_block('ETHUSDT', -1, open_pos)
        self.assertFalse(result)

    def test_blocks_any_side_when_same_side_off(self):
        from train.v5_correlation import RollingDailyCorr, CorrBlocker, CorrConfig
        tracker = RollingDailyCorr(['BTCUSDT', 'ETHUSDT'], window_days=30)
        for i in range(15):
            d = f"2024-01-{i+1:02d}"
            tracker.record_trade('BTCUSDT', d, float(i))
            tracker.record_trade('ETHUSDT', d, float(i) * 1.5)

        cfg = CorrConfig(enabled=True, threshold=0.60, same_side_only=False)
        blocker = CorrBlocker(tracker, cfg)

        open_pos = {'BTCUSDT': 1}
        result = blocker.should_block('ETHUSDT', -1, open_pos)
        self.assertTrue(result)

    def test_allows_when_insufficient_data(self):
        from train.v5_correlation import RollingDailyCorr, CorrBlocker, CorrConfig
        tracker = RollingDailyCorr(['BTCUSDT', 'ETHUSDT'],
                                   window_days=30, min_aligned_days=10)
        for i in range(3):
            d = f"2024-01-{i+1:02d}"
            tracker.record_trade('BTCUSDT', d, float(i))
            tracker.record_trade('ETHUSDT', d, float(i))

        cfg = CorrConfig(enabled=True, threshold=0.50)
        blocker = CorrBlocker(tracker, cfg)

        open_pos = {'BTCUSDT': 1}
        result = blocker.should_block('ETHUSDT', 1, open_pos)
        self.assertFalse(result)

    def test_allows_when_corr_below_threshold(self):
        from train.v5_correlation import RollingDailyCorr, CorrBlocker, CorrConfig
        tracker = RollingDailyCorr(['BTCUSDT', 'ETHUSDT'], window_days=30)
        np.random.seed(99)
        for i in range(20):
            d = f"2024-01-{i+1:02d}"
            tracker.record_trade('BTCUSDT', d, np.random.randn())
            tracker.record_trade('ETHUSDT', d, np.random.randn())

        cfg = CorrConfig(enabled=True, threshold=0.90)
        blocker = CorrBlocker(tracker, cfg)

        corr = tracker.pairwise_corr('BTCUSDT', 'ETHUSDT')
        self.assertIsNotNone(corr)
        self.assertLess(abs(corr), 0.90)

        open_pos = {'BTCUSDT': 1}
        result = blocker.should_block('ETHUSDT', 1, open_pos)
        self.assertFalse(result)

    def test_disabled_blocker_allows_all(self):
        from train.v5_correlation import RollingDailyCorr, CorrBlocker, CorrConfig
        tracker = RollingDailyCorr(['BTCUSDT', 'ETHUSDT'], window_days=30)
        for i in range(15):
            d = f"2024-01-{i+1:02d}"
            tracker.record_trade('BTCUSDT', d, float(i))
            tracker.record_trade('ETHUSDT', d, float(i))

        cfg = CorrConfig(enabled=False, threshold=0.50)
        blocker = CorrBlocker(tracker, cfg)

        open_pos = {'BTCUSDT': 1}
        result = blocker.should_block('ETHUSDT', 1, open_pos)
        self.assertFalse(result)

    def test_single_symbol_allows_all(self):
        from train.v5_correlation import RollingDailyCorr, CorrBlocker, CorrConfig
        tracker = RollingDailyCorr(['BTCUSDT'], window_days=30)
        cfg = CorrConfig(enabled=True, threshold=0.50)
        blocker = CorrBlocker(tracker, cfg)

        result = blocker.should_block('BTCUSDT', 1, {})
        self.assertFalse(result)

    def test_no_open_positions_allows(self):
        from train.v5_correlation import RollingDailyCorr, CorrBlocker, CorrConfig
        tracker = RollingDailyCorr(['BTCUSDT', 'ETHUSDT'], window_days=30)
        cfg = CorrConfig(enabled=True, threshold=0.50)
        blocker = CorrBlocker(tracker, cfg)

        result = blocker.should_block('ETHUSDT', 1, {})
        self.assertFalse(result)


class TestOverlapRatio(unittest.TestCase):

    def test_no_overlap(self):
        from train.v5_correlation import compute_overlap_ratio
        spans = {
            'BTCUSDT': [(0, 10)],
            'ETHUSDT': [(20, 30)],
        }
        ratio = compute_overlap_ratio(spans, 100)
        self.assertAlmostEqual(ratio, 0.0)

    def test_full_overlap(self):
        from train.v5_correlation import compute_overlap_ratio
        spans = {
            'BTCUSDT': [(0, 100)],
            'ETHUSDT': [(0, 100)],
        }
        ratio = compute_overlap_ratio(spans, 100)
        self.assertAlmostEqual(ratio, 1.0)

    def test_partial_overlap(self):
        from train.v5_correlation import compute_overlap_ratio
        spans = {
            'BTCUSDT': [(0, 50)],
            'ETHUSDT': [(25, 75)],
        }
        ratio = compute_overlap_ratio(spans, 100)
        self.assertAlmostEqual(ratio, 0.25)

    def test_single_symbol_no_overlap(self):
        from train.v5_correlation import compute_overlap_ratio
        spans = {'BTCUSDT': [(0, 50)]}
        ratio = compute_overlap_ratio(spans, 100)
        self.assertAlmostEqual(ratio, 0.0)


class TestCorrReport(unittest.TestCase):

    def test_build_report_structure(self):
        from train.v5_correlation import (
            RollingDailyCorr, CorrBlocker, CorrConfig,
            build_fold_corr_report
        )
        tracker = RollingDailyCorr(['BTCUSDT', 'ETHUSDT'], window_days=30)
        for i in range(15):
            d = f"2024-01-{i+1:02d}"
            tracker.record_trade('BTCUSDT', d, float(i))
            tracker.record_trade('ETHUSDT', d, float(i) * 0.8)

        cfg = CorrConfig(enabled=True)
        blocker = CorrBlocker(tracker, cfg)

        report = build_fold_corr_report(
            fold_id=1, window_train="2023-01→2024-01",
            window_test="2024-01→2024-02",
            corr_tracker=tracker, overlap_ratio=0.15, blocker=blocker,
        )

        self.assertEqual(report['fold_id'], 1)
        self.assertIn('corr_matrix', report)
        self.assertIn('mean_abs_corr', report)
        self.assertIn('max_abs_corr', report)
        self.assertIn('max_abs_corr_pair', report)
        self.assertIn('overlap_ratio', report)
        self.assertIn('daily_r_series', report)
        self.assertIn('corr_blocked_trades', report)
        self.assertEqual(len(report['corr_matrix']), 2)
        self.assertEqual(len(report['corr_matrix'][0]), 2)


class TestCLIFlags(unittest.TestCase):

    def _make_parser(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--train-v5", action="store_true")
        parser.add_argument("--symbols", type=str, default="BTCUSDT")
        parser.add_argument("--v5-corr-block", action="store_true", default=True)
        parser.add_argument("--v5-no-corr-block", dest="v5_corr_block", action="store_false")
        parser.add_argument("--v5-corr-window-days", type=int, default=30)
        parser.add_argument("--v5-corr-thresh", type=float, default=0.70)
        parser.add_argument("--v5-corr-same-side-only", action="store_true", default=True)
        parser.add_argument("--v5-corr-any-side", dest="v5_corr_same_side_only", action="store_false")
        parser.add_argument("--v5-log-corr-matrix", action="store_true", default=True)
        parser.add_argument("--no-correlation-block", action="store_true", default=False)
        return parser

    def test_corr_flags_parse(self):
        parser = self._make_parser()
        args = parser.parse_args([
            '--v5-corr-thresh', '0.80',
            '--v5-corr-window-days', '14',
        ])
        self.assertAlmostEqual(args.v5_corr_thresh, 0.80)
        self.assertEqual(args.v5_corr_window_days, 14)
        self.assertTrue(args.v5_corr_block)
        self.assertTrue(args.v5_corr_same_side_only)

    def test_corr_block_disabled(self):
        parser = self._make_parser()
        args = parser.parse_args(['--v5-no-corr-block'])
        self.assertFalse(args.v5_corr_block)

    def test_corr_any_side(self):
        parser = self._make_parser()
        args = parser.parse_args(['--v5-corr-any-side'])
        self.assertFalse(args.v5_corr_same_side_only)

    def test_deprecated_no_correlation_block_overrides(self):
        parser = self._make_parser()
        args = parser.parse_args(['--no-correlation-block'])
        self.assertTrue(args.no_correlation_block)
        effective = args.v5_corr_block and not args.no_correlation_block
        self.assertFalse(effective)


if __name__ == '__main__':
    unittest.main()
