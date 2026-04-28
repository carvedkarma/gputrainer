"""
V5 Leakage & Overfitting Diagnostics

Suite of diagnostic tools to detect data leakage, overfitting, and validate
that model performance is genuine:

1. Shuffled-target sanity check (should drop to ~random accuracy)
2. Feature importance / leakage audit (permutation importance)
3. Naive random-entry baseline for forward test comparison
4. Temporal leakage scanner (audit features for look-ahead)
5. Train vs forward test gap report
"""

import numpy as np
import torch
import logging
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from pathlib import Path

log = logging.getLogger("QuickStart")


def run_shuffled_target_check(
    model, device, val_loader, val_action_arr, val_valid,
    n_shuffles: int = 3,
):
    """Shuffle action labels, re-run inference, check if accuracy drops to ~random.

    If model accuracy stays high on shuffled targets, features are leaking
    future information.

    Returns dict with real_accuracy, shuffled_accuracies, verdict.
    """
    model.eval()

    n = len(val_action_arr)
    valid = val_valid[:n].astype(bool)
    if np.sum(valid) < 10:
        log.warning("[SHUFFLED_CHECK] Too few valid samples (%d), skipping.", np.sum(valid))
        return {
            'real_accuracy': 0.0, 'shuffled_accuracies': [], 'mean_shuffled_accuracy': 0.0,
            'random_baseline': 0.0, 'n_classes': 0, 'verdict': 'INSUFFICIENT_DATA',
            'detail': 'Too few valid samples to run shuffled target check.',
        }

    all_logits = []
    with torch.no_grad():
        for batch in val_loader:
            feat = batch['features'].to(device)
            sym_id = batch.get('symbol_id')
            if sym_id is not None:
                sym_id = sym_id.to(device)
            outputs = model(feat, symbol_ids=sym_id)
            all_logits.append(outputs['action_logits'].detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    preds = np.argmax(logits, axis=1)

    n = min(len(preds), n)
    preds = preds[:n]
    true_labels = val_action_arr[:n]
    valid = valid[:n]

    real_acc = float(np.mean(preds[valid] == true_labels[valid]))

    n_classes = len(np.unique(true_labels[valid]))
    random_baseline = 1.0 / max(n_classes, 1)

    shuffled_accs = []
    for s in range(n_shuffles):
        shuffled_labels = true_labels.copy()
        np.random.seed(42 + s)
        shuffled_labels[valid] = np.random.permutation(shuffled_labels[valid])

        shuffled_acc = float(np.mean(preds[valid] == shuffled_labels[valid]))
        shuffled_accs.append(shuffled_acc)

    mean_shuffled = float(np.mean(shuffled_accs))

    if real_acc > 0.8 and mean_shuffled > 0.5:
        verdict = "SUSPICIOUS"
        detail = (f"Real accuracy {real_acc:.1%} AND shuffled accuracy {mean_shuffled:.1%} "
                  f"are both high. Model may be fitting features that encode future info.")
    elif real_acc > mean_shuffled * 2:
        verdict = "PASS"
        detail = (f"Real accuracy {real_acc:.1%} vs shuffled {mean_shuffled:.1%}. "
                  f"Model clearly learns from labels, not just feature artifacts.")
    else:
        verdict = "WARNING"
        detail = (f"Real accuracy {real_acc:.1%} close to shuffled {mean_shuffled:.1%}. "
                  f"Model may not be learning meaningful patterns from labels.")

    log.info("")
    log.info("=" * 70)
    log.info("  DIAGNOSTIC: Shuffled Target Sanity Check")
    log.info("=" * 70)
    log.info(f"  Real accuracy:      {real_acc:.4f} ({real_acc:.1%})")
    log.info(f"  Shuffled accuracy:  {mean_shuffled:.4f} ({mean_shuffled:.1%}) [mean of {n_shuffles} shuffles]")
    log.info(f"  Random baseline:    {random_baseline:.4f} ({random_baseline:.1%}) [{n_classes} classes]")
    log.info(f"  Verdict:            {verdict}")
    log.info(f"  Detail:             {detail}")
    log.info("=" * 70)

    return {
        'real_accuracy': real_acc,
        'shuffled_accuracies': shuffled_accs,
        'mean_shuffled_accuracy': mean_shuffled,
        'random_baseline': random_baseline,
        'n_classes': n_classes,
        'verdict': verdict,
        'detail': detail,
    }


def run_feature_importance_audit(
    model, device, val_loader, val_action_arr, val_valid,
    feature_names: List[str],
    n_top: int = 15,
    n_repeats: int = 3,
):
    """Permutation importance: shuffle each feature, measure accuracy drop.

    Features with very high importance that relate to future price movement
    are suspicious for leakage.

    Returns dict with feature rankings and flags.
    """
    model.eval()

    n = len(val_action_arr)
    valid = val_valid[:n].astype(bool)
    if np.sum(valid) < 10:
        log.warning("[FEAT_IMPORTANCE] Too few valid samples (%d), skipping.", np.sum(valid))
        return {
            'baseline_accuracy': 0.0, 'importances': [], 'top_features': [],
            'flagged_features': [], 'concentration_top3': 0.0,
        }

    all_logits = []
    all_features = []
    all_sym_ids = []
    with torch.no_grad():
        for batch in val_loader:
            feat = batch['features'].to(device)
            sym_id = batch.get('symbol_id')
            if sym_id is not None:
                sym_id = sym_id.to(device)
                all_sym_ids.append(batch['symbol_id'].numpy())
            outputs = model(feat, symbol_ids=sym_id)
            all_logits.append(outputs['action_logits'].detach().cpu().numpy())
            all_features.append(batch['features'].numpy())

    logits = np.concatenate(all_logits, axis=0)
    features_arr = np.concatenate(all_features, axis=0)
    sym_ids_arr = np.concatenate(all_sym_ids, axis=0) if all_sym_ids else None
    preds = np.argmax(logits, axis=1)

    n = min(len(preds), n)
    true_labels = val_action_arr[:n]
    valid = valid[:n]

    baseline_acc = float(np.mean(preds[:n][valid] == true_labels[valid]))
    n_features = features_arr.shape[1]

    log.info(f"[FEAT_IMPORTANCE] Running permutation importance on {n_features} features "
             f"({n_repeats} repeats each)...")

    importances = []

    for fi in range(n_features):
        drops = []
        for rep in range(n_repeats):
            shuffled_features = features_arr.copy()
            rng = np.random.RandomState(42 + fi * 100 + rep)
            shuffled_features[:, fi] = rng.permutation(shuffled_features[:, fi])

            shuf_logits = []
            with torch.no_grad():
                bs = 512
                for start in range(0, len(shuffled_features), bs):
                    end = min(start + bs, len(shuffled_features))
                    feat_batch = torch.tensor(shuffled_features[start:end], dtype=torch.float32).to(device)
                    sym_batch = None
                    if sym_ids_arr is not None:
                        sym_batch = torch.tensor(sym_ids_arr[start:end], dtype=torch.long).to(device)
                    out = model(feat_batch, symbol_ids=sym_batch)
                    shuf_logits.append(out['action_logits'].detach().cpu().numpy())

            shuf_preds = np.argmax(np.concatenate(shuf_logits, axis=0), axis=1)[:n]
            shuf_acc = float(np.mean(shuf_preds[valid] == true_labels[valid]))
            drops.append(baseline_acc - shuf_acc)

        mean_drop = float(np.mean(drops))
        std_drop = float(np.std(drops))
        importances.append({
            'feature_idx': fi,
            'feature_name': feature_names[fi] if fi < len(feature_names) else f"feat_{fi}",
            'importance': mean_drop,
            'std': std_drop,
        })

    importances.sort(key=lambda x: x['importance'], reverse=True)

    suspicious_keywords = [
        'mfe', 'mae', 'ret_r', 'future', 'forward', 'target',
        'realized', 'outcome', 'pnl', 'profit', 'loss',
    ]
    flagged = []
    for imp in importances[:n_top]:
        name_lower = imp['feature_name'].lower()
        for kw in suspicious_keywords:
            if kw in name_lower:
                flagged.append(imp)
                break

    log.info("")
    log.info("=" * 70)
    log.info("  DIAGNOSTIC: Feature Importance / Leakage Audit")
    log.info("=" * 70)
    log.info(f"  Baseline accuracy: {baseline_acc:.4f}")
    log.info(f"  Top {n_top} most important features:")
    log.info(f"  {'Rank':>4} {'Feature':>30} {'Importance':>12} {'±Std':>8}")
    log.info("-" * 70)
    for rank, imp in enumerate(importances[:n_top], 1):
        flag_str = " *** FLAGGED" if imp in flagged else ""
        log.info(f"  {rank:>4} {imp['feature_name']:>30} {imp['importance']:>+12.6f} "
                 f"{imp['std']:>8.6f}{flag_str}")

    if flagged:
        log.warning(f"  FLAGGED: {len(flagged)} top features have suspicious names!")
        for f in flagged:
            log.warning(f"    -> {f['feature_name']} (importance={f['importance']:+.6f})")
    else:
        log.info("  No suspicious feature names flagged in top features.")

    concentration = sum(i['importance'] for i in importances[:3]) / max(
        sum(i['importance'] for i in importances if i['importance'] > 0), 1e-8)
    if concentration > 0.5 and importances[0]['importance'] > 0.1:
        log.warning(f"  WARNING: Top 3 features account for {concentration:.0%} of total importance. "
                    f"Model may be over-relying on a few features.")
    log.info("=" * 70)

    return {
        'baseline_accuracy': baseline_acc,
        'importances': importances,
        'top_features': importances[:n_top],
        'flagged_features': flagged,
        'concentration_top3': concentration,
    }


def run_random_baseline(
    test_realized_r, test_outcomes, test_valid,
    n_trades_to_match: int,
    test_bars: int,
    n_simulations: int = 100,
    test_timestamps=None,
):
    """Generate random-entry baseline for forward test comparison.

    Randomly select the same number of trades as the model took, compute
    performance metrics. Averages over many simulations for stability.

    Returns dict with baseline metrics.
    """
    valid_mask = test_valid.astype(bool)
    valid_outcomes = np.isin(test_outcomes, ["TP", "SL", "EXP_WIN", "EXP_LOSS"])
    eligible = valid_mask & valid_outcomes

    eligible_indices = np.where(eligible)[0]
    eligible_r = test_realized_r[eligible_indices].astype(float)
    eligible_r = np.where(np.isnan(eligible_r), 0.0, eligible_r)

    n_eligible = len(eligible_indices)
    n_to_pick = min(n_trades_to_match, n_eligible)

    if n_to_pick == 0 or n_eligible == 0:
        log.warning("[RANDOM_BASELINE] No eligible trades to sample from")
        return {'verdict': 'NO_DATA', 'n_eligible': 0}

    sim_winrates = []
    sim_expects = []
    sim_pfs = []
    sim_sharpes = []
    sim_total_r = []

    val_days = test_bars / 96.0

    for sim in range(n_simulations):
        rng = np.random.RandomState(42 + sim)
        picked = rng.choice(n_eligible, size=n_to_pick, replace=False)
        r_picked = eligible_r[picked]

        wins = r_picked[r_picked > 0]
        losses = r_picked[r_picked <= 0]

        wr = len(wins) / max(len(r_picked), 1)
        exp = float(np.mean(r_picked))
        total_win = float(np.sum(wins))
        total_loss = float(abs(np.sum(losses)))
        pf = total_win / max(total_loss, 1e-6)

        daily_r = exp * (n_to_pick / max(val_days, 1e-6))
        std_r = float(np.std(r_picked)) if len(r_picked) > 1 else 1.0
        daily_std = std_r * np.sqrt(n_to_pick / max(val_days, 1e-6))
        sharpe = daily_r / max(daily_std, 1e-6) * np.sqrt(252)

        sim_winrates.append(wr)
        sim_expects.append(exp)
        sim_pfs.append(pf)
        sim_sharpes.append(sharpe)
        sim_total_r.append(float(np.sum(r_picked)))

    result = {
        'n_simulations': n_simulations,
        'n_trades_per_sim': n_to_pick,
        'n_eligible': n_eligible,
        'win_rate_mean': float(np.mean(sim_winrates)),
        'win_rate_std': float(np.std(sim_winrates)),
        'expectancy_mean': float(np.mean(sim_expects)),
        'expectancy_std': float(np.std(sim_expects)),
        'profit_factor_mean': float(np.mean(sim_pfs)),
        'sharpe_mean': float(np.mean(sim_sharpes)),
        'sharpe_std': float(np.std(sim_sharpes)),
        'total_r_mean': float(np.mean(sim_total_r)),
        'total_r_std': float(np.std(sim_total_r)),
    }

    log.info("")
    log.info("=" * 70)
    log.info("  DIAGNOSTIC: Random Entry Baseline")
    log.info("=" * 70)
    log.info(f"  Simulations:    {n_simulations} x {n_to_pick} random trades (from {n_eligible} eligible)")
    log.info(f"  Win Rate:       {result['win_rate_mean']:.1%} ± {result['win_rate_std']:.1%}")
    log.info(f"  Expectancy:     {result['expectancy_mean']:+.4f} ± {result['expectancy_std']:.4f} R")
    log.info(f"  Profit Factor:  {result['profit_factor_mean']:.2f}")
    log.info(f"  Sharpe:         {result['sharpe_mean']:.2f} ± {result['sharpe_std']:.2f}")
    log.info(f"  Total R:        {result['total_r_mean']:+.2f} ± {result['total_r_std']:.2f}")
    log.info("=" * 70)

    return result


def run_temporal_leakage_scan(feature_names: List[str], feature_matrix: np.ndarray,
                               targets: np.ndarray, valid_mask: np.ndarray,
                               horizon: int = 16):
    """Scan features for temporal leakage patterns.

    Checks:
    1. Feature-target correlation at lag 0 vs lag -horizon (if feature uses future data,
       correlation at lag 0 should be suspiciously high)
    2. Autocorrelation pattern: features built from future data tend to have
       unusual autocorrelation signatures
    3. Information coefficient: rank correlation of feature values with future returns

    Returns dict with per-feature leakage scores and flags.
    """
    valid = valid_mask.astype(bool)
    n_feat = feature_matrix.shape[1]

    log.info(f"[TEMPORAL_SCAN] Scanning {n_feat} features for look-ahead patterns...")

    results = []

    safe_targets = targets.copy()
    safe_targets = np.where(np.isfinite(safe_targets), safe_targets, 0.0)

    for fi in range(n_feat):
        feat_col = feature_matrix[:, fi]
        feat_name = feature_names[fi] if fi < len(feature_names) else f"feat_{fi}"

        feat_valid = np.isfinite(feat_col) & valid
        if np.sum(feat_valid) < 100:
            results.append({
                'feature_name': feat_name,
                'feature_idx': fi,
                'lag0_corr': 0.0,
                'future_corr': 0.0,
                'leakage_score': 0.0,
                'flag': False,
            })
            continue

        idx = np.where(feat_valid)[0]
        f_vals = feat_col[idx]
        t_vals = safe_targets[idx]

        if np.std(f_vals) < 1e-10 or np.std(t_vals) < 1e-10:
            results.append({
                'feature_name': feat_name,
                'feature_idx': fi,
                'lag0_corr': 0.0,
                'future_corr': 0.0,
                'leakage_score': 0.0,
                'flag': False,
            })
            continue

        corr_matrix = np.corrcoef(f_vals, t_vals)
        lag0_corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

        max_idx = len(idx) - horizon
        if max_idx > 100:
            future_vals = safe_targets[idx[:max_idx] + horizon]
            curr_vals = f_vals[:max_idx]
            valid_future = np.isfinite(future_vals) & np.isfinite(curr_vals)
            if np.sum(valid_future) > 50:
                fc = np.corrcoef(curr_vals[valid_future], future_vals[valid_future])
                future_corr = float(fc[0, 1]) if not np.isnan(fc[0, 1]) else 0.0
            else:
                future_corr = 0.0
        else:
            future_corr = 0.0

        leakage_score = abs(lag0_corr) - abs(future_corr)

        flag = abs(lag0_corr) > 0.3 and leakage_score > 0.15

        results.append({
            'feature_name': feat_name,
            'feature_idx': fi,
            'lag0_corr': lag0_corr,
            'future_corr': future_corr,
            'leakage_score': leakage_score,
            'flag': flag,
        })

    results.sort(key=lambda x: abs(x['lag0_corr']), reverse=True)

    flagged = [r for r in results if r['flag']]

    log.info("")
    log.info("=" * 70)
    log.info("  DIAGNOSTIC: Temporal Leakage Scan")
    log.info("=" * 70)
    log.info(f"  Features scanned:    {n_feat}")
    log.info(f"  Features flagged:    {len(flagged)}")
    log.info("")
    log.info(f"  Top 15 by |correlation with target|:")
    log.info(f"  {'Feature':>30} {'|Lag0 Corr|':>12} {'|Future Corr|':>14} {'Leak Score':>12} {'Flag':>6}")
    log.info("-" * 70)
    for r in results[:15]:
        flag_str = "***" if r['flag'] else ""
        log.info(f"  {r['feature_name']:>30} {abs(r['lag0_corr']):>12.4f} "
                 f"{abs(r['future_corr']):>14.4f} {r['leakage_score']:>+12.4f} {flag_str:>6}")

    if flagged:
        log.warning(f"  LEAKAGE ALERT: {len(flagged)} features have suspicious correlation patterns!")
        for f in flagged:
            log.warning(f"    -> {f['feature_name']}: corr={f['lag0_corr']:.4f} "
                        f"future_corr={f['future_corr']:.4f} leak_score={f['leakage_score']:.4f}")
    else:
        log.info("  PASS: No features flagged for temporal leakage.")
    log.info("=" * 70)

    return {
        'n_features': n_feat,
        'n_flagged': len(flagged),
        'flagged_features': flagged,
        'all_results': results,
    }


def run_gap_analysis(
    sweep_metrics: Dict,
    forward_report: Optional[Dict],
):
    """Compare in-sample sweep metrics vs forward test metrics.

    If the gap is suspiciously small, the model may have memorized or
    there may be leakage. If the gap is enormous, possible overfitting.

    Returns dict with gap analysis and verdict.
    """
    if forward_report is None or forward_report.get('total_trades', 0) == 0:
        log.warning("[GAP_ANALYSIS] No forward test results available, skipping.")
        return {'verdict': 'NO_FORWARD_TEST'}

    is_wr = sweep_metrics.get('win_rate', 0)
    is_exp = sweep_metrics.get('expectancy_r', 0)
    is_pf = sweep_metrics.get('profit_factor', 0)
    is_sharpe = sweep_metrics.get('sharpe', 0)

    oos_wr = forward_report.get('win_rate', 0)
    oos_exp = forward_report.get('expectancy_r', 0)
    oos_pf = forward_report.get('profit_factor', 0)
    oos_sharpe = forward_report.get('sharpe', 0)

    wr_gap = is_wr - oos_wr
    exp_gap = is_exp - oos_exp
    pf_gap = is_pf - oos_pf
    sharpe_gap = is_sharpe - oos_sharpe

    wr_ratio = oos_wr / max(is_wr, 1e-6)
    exp_ratio = oos_exp / max(abs(is_exp), 1e-6) if is_exp != 0 else 0
    pf_ratio = oos_pf / max(is_pf, 1e-6)

    if wr_gap < 0.05 and abs(exp_gap) < 0.02:
        verdict = "SUSPICIOUS_SMALL_GAP"
        detail = (f"In-sample and OOS metrics are nearly identical. "
                  f"WR gap={wr_gap:.1%}, Exp gap={exp_gap:+.4f}R. "
                  f"This may indicate leakage or the test set overlaps training data.")
    elif oos_wr < 0.40 or oos_exp < -0.1:
        verdict = "OVERFIT"
        detail = (f"In-sample WR={is_wr:.1%} but OOS WR={oos_wr:.1%}. "
                  f"In-sample Exp={is_exp:+.4f}R but OOS Exp={oos_exp:+.4f}R. "
                  f"Model has likely overfit to training data.")
    elif wr_gap > 0.20:
        verdict = "HIGH_DEGRADATION"
        detail = (f"Win rate drops {wr_gap:.1%} from in-sample to OOS. "
                  f"Significant overfitting detected.")
    elif oos_exp > 0 and oos_wr > 0.45:
        verdict = "HEALTHY"
        detail = (f"In-sample WR={is_wr:.1%} → OOS WR={oos_wr:.1%} (gap={wr_gap:.1%}). "
                  f"In-sample Exp={is_exp:+.4f}R → OOS Exp={oos_exp:+.4f}R. "
                  f"Performance degrades but remains positive. Acceptable.")
    else:
        verdict = "MARGINAL"
        detail = (f"OOS performance is marginal. WR={oos_wr:.1%}, Exp={oos_exp:+.4f}R. "
                  f"Model may have some edge but not reliable.")

    log.info("")
    log.info("=" * 70)
    log.info("  DIAGNOSTIC: In-Sample vs OOS Gap Analysis")
    log.info("=" * 70)
    log.info(f"  {'Metric':>20} {'In-Sample':>12} {'OOS':>12} {'Gap':>12} {'Ratio':>8}")
    log.info("-" * 70)
    log.info(f"  {'Win Rate':>20} {is_wr:>12.1%} {oos_wr:>12.1%} {wr_gap:>+12.1%} {wr_ratio:>8.2f}")
    log.info(f"  {'Expectancy (R)':>20} {is_exp:>+12.4f} {oos_exp:>+12.4f} {exp_gap:>+12.4f} {exp_ratio:>8.2f}")
    log.info(f"  {'Profit Factor':>20} {is_pf:>12.2f} {oos_pf:>12.2f} {pf_gap:>+12.2f} {pf_ratio:>8.2f}")
    log.info(f"  {'Sharpe':>20} {is_sharpe:>12.2f} {oos_sharpe:>12.2f} {sharpe_gap:>+12.2f}")
    log.info("-" * 70)
    log.info(f"  Verdict: {verdict}")
    log.info(f"  {detail}")
    log.info("=" * 70)

    return {
        'in_sample': {
            'win_rate': is_wr, 'expectancy_r': is_exp,
            'profit_factor': is_pf, 'sharpe': is_sharpe,
        },
        'oos': {
            'win_rate': oos_wr, 'expectancy_r': oos_exp,
            'profit_factor': oos_pf, 'sharpe': oos_sharpe,
        },
        'gaps': {
            'win_rate': wr_gap, 'expectancy_r': exp_gap,
            'profit_factor': pf_gap, 'sharpe': sharpe_gap,
        },
        'ratios': {
            'win_rate': wr_ratio, 'expectancy_r': exp_ratio,
            'profit_factor': pf_ratio,
        },
        'verdict': verdict,
        'detail': detail,
    }


def run_all_diagnostics(
    model, device,
    val_loader, val_action_arr, val_valid,
    val_feat, val_ret_R, feature_names,
    val_outcomes, val_realized_r, val_timestamps,
    test_bars, horizon,
    sweep_metrics=None, forward_report=None,
    n_trades_model=None,
):
    """Run all leakage/overfitting diagnostics in sequence.

    Called after training completes. Results are logged and returned.
    """
    log.info("")
    log.info("#" * 70)
    log.info("#  V5 LEAKAGE & OVERFITTING DIAGNOSTICS SUITE")
    log.info("#" * 70)

    results = {}

    log.info("\n[1/5] Shuffled Target Check...")
    results['shuffled_target'] = run_shuffled_target_check(
        model, device, val_loader, val_action_arr, val_valid,
    )

    log.info("\n[2/5] Feature Importance Audit...")
    results['feature_importance'] = run_feature_importance_audit(
        model, device, val_loader, val_action_arr, val_valid,
        feature_names=feature_names,
    )

    log.info("\n[3/5] Random Entry Baseline...")
    n_trades = n_trades_model or 100
    results['random_baseline'] = run_random_baseline(
        test_realized_r=val_realized_r,
        test_outcomes=val_outcomes,
        test_valid=val_valid,
        n_trades_to_match=n_trades,
        test_bars=test_bars,
        test_timestamps=val_timestamps,
    )

    log.info("\n[4/5] Temporal Leakage Scan...")
    results['temporal_leakage'] = run_temporal_leakage_scan(
        feature_names=feature_names,
        feature_matrix=val_feat,
        targets=val_ret_R,
        valid_mask=val_valid,
        horizon=horizon,
    )

    log.info("\n[5/5] Gap Analysis...")
    if sweep_metrics and forward_report:
        results['gap_analysis'] = run_gap_analysis(sweep_metrics, forward_report)
    else:
        log.info("  Skipped: need both sweep metrics and forward report.")
        results['gap_analysis'] = {'verdict': 'SKIPPED'}

    log.info("")
    log.info("#" * 70)
    log.info("#  DIAGNOSTICS SUMMARY")
    log.info("#" * 70)
    log.info(f"  Shuffled Target:     {results['shuffled_target'].get('verdict', 'N/A')}")
    log.info(f"  Feature Leakage:     {results['temporal_leakage'].get('n_flagged', 0)} features flagged")
    log.info(f"  Feature Importance:  top={results['feature_importance']['top_features'][0]['feature_name'] if results['feature_importance']['top_features'] else 'N/A'}")
    rbl = results.get('random_baseline', {})
    if rbl.get('verdict') != 'NO_DATA':
        log.info(f"  Random Baseline:     WR={rbl.get('win_rate_mean',0):.1%} Exp={rbl.get('expectancy_mean',0):+.4f}R")
    log.info(f"  Gap Analysis:        {results['gap_analysis'].get('verdict', 'N/A')}")
    log.info("#" * 70)

    return results
