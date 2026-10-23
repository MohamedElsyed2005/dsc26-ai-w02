"""Decision metrics for a mule-risk score.

You write this file. Keep the signatures exactly as they are: the grader calls them by name.

Rules for every function:
- y_true and scores are lists or 1-D arrays. Raise ValueError if the lengths differ, the
  input is empty, a label is not 0 or 1, or a score is NaN.
- A case is flagged when score >= threshold.
- Precision is 0.0 when nothing is flagged; recall is 0.0 when there are no positives.
"""
from __future__ import annotations
import math
import numpy as np
 
def _validate_inputs(y_true, scores):
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
 
    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")
 
    if len(y) == 0 or len(s) == 0:
        raise ValueError("Inputs must not be empty")
 
    if len(y) != len(s):
        raise ValueError("Labels and scores must have equal lengths")
 
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("Labels must be 0 or 1")
 
    if np.isnan(s).any():
        raise ValueError("Scores must not contain NaN")
 
    return y.astype(int), s
 
def confusion_at(y_true, scores, threshold: float) -> dict:
    """Return {"tp": int, "fp": int, "tn": int, "fn": int} at `threshold`."""
    y, s = _validate_inputs(y_true, scores)
    flagged = s >= threshold
 
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    tn = int(np.sum(~flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
 
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
 
 
def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    """Return (precision, recall) at `threshold`."""
    y, s = _validate_inputs(y_true, scores)
    counts = confusion_at(y, s, threshold)
 
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
 
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
 
    return float(precision), float(recall)
 
 
def precision_at_k(y_true, scores, k: int) -> float:
    """Share of positives among the k highest scores.
 
    ValueError if k <= 0; if k > n, use n. Rank by score descending; ties keep their
    original order.
    """
    y, s = _validate_inputs(y_true, scores)
 
    if k <= 0:
        raise ValueError("k must be positive")
 
    k = min(k, len(y))
 
    # Stable sorting preserves original order for tied scores.
    order = np.argsort(-s, kind="stable")
    top_k = y[order[:k]]
 
    return float(np.mean(top_k))
 
 
def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    """fp * cost_fp + fn * cost_fn at `threshold`."""
    y, s = _validate_inputs(y_true, scores)
    counts = confusion_at(y, s, threshold)
 
    return float(counts["fp"] * cost_fp + counts["fn"] * cost_fn)
 
 
def _counts_per_distinct_threshold(y, s):
    """For each distinct score t (ascending): number flagged and true positives at s >= t."""
    uniq = np.unique(s)
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    y_sorted = y[order]
    # first index in the ascending order where score >= t
    first = np.searchsorted(s_sorted, uniq, side="left")
    pos_suffix = np.concatenate([np.cumsum(y_sorted[::-1])[::-1], [0]])
    n = len(s)
    flagged = n - first
    tp = pos_suffix[first]
    return uniq, flagged.astype(int), tp.astype(int)
 
 
def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    """Threshold with the lowest expected cost.
 
    Candidates are every distinct score plus math.inf ("flag nothing"). On a tie in cost,
    return the highest threshold (fewer flags for the same cost).
    """
    y, s = _validate_inputs(y_true, scores)
 
    uniq, flagged, tp = _counts_per_distinct_threshold(y, s)
    total_pos = int(y.sum())
 
    best_t = math.inf
    best_cost = float(total_pos * cost_fn)  # flag nothing: every positive is missed
 
    # Highest to lowest, so on a tie the highest threshold is kept.
    for i in range(len(uniq) - 1, -1, -1):
        fp = int(flagged[i] - tp[i])
        fn = total_pos - int(tp[i])
        cost = float(fp * cost_fp + fn * cost_fn)
        if cost < best_cost:
            best_cost = cost
            best_t = float(uniq[i])
 
    return float(best_t)
 
 
def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    """Lowest distinct score whose precision is >= min_precision, or None if none reaches it."""
    y, s = _validate_inputs(y_true, scores)
 
    uniq, flagged, tp = _counts_per_distinct_threshold(y, s)
 
    for i in range(len(uniq)):
        precision = int(tp[i]) / int(flagged[i])
        if precision >= min_precision:
            return float(uniq[i])
 
    return None
 