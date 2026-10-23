"""Data audit for Rawaj Pay transfer extracts.

You write this file. Keep the signature exactly as it is: the grader calls it by name.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _is_missing(value):
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    """Audit a transfer extract before anyone trains a model on it.

    Raises KeyError if `target`, `id_col` or `time_col` is not a column of `df`.

    Returns a dict with exactly these keys:

    - "n_rows" (int): number of rows.
    - "missing" ({column: float}): fraction missing, for every column. Missing means
      NaN/None, or a string that is empty or only whitespace.
    - "duplicate_ids" (list): sorted non-missing `id_col` values that appear more than once.
    - "duplicate_rows" (int): rows that exactly repeat an earlier row.
    - "constant_columns" (list): sorted columns other than `target` with at most one
      distinct non-missing value.
    - "unlabelled_rows" (int): rows where `target` is missing.
    - "positive_rate" (float): share of labelled rows with target == 1; 0.0 if none is labelled.
    - "bad_timestamps" (int): non-missing values in `time_col` that do not parse as dates.
    - "time_range" (list): [min, max] of the parsed timestamps as ISO strings, or [None, None].
    - "leakage_suspects" (list): sorted columns (not target, id_col or time_col) that look
      like leaks. See README.md for the exact rule.
    """
    required = [target, id_col, time_col]

    for col in required:
        if col not in df.columns:
            raise KeyError(col)

    n_rows = len(df)

    # Treat blank strings as missing, too.
    missing_mask = pd.DataFrame(
        {
            col: df[col].map(_is_missing)
            for col in df.columns
        },
        index=df.index,
    )

    missing = {
        col: float(missing_mask[col].mean()) if n_rows else 0.0
        for col in df.columns
    }

    # Duplicate IDs: ignore missing IDs.
    id_missing = missing_mask[id_col]
    duplicate_ids = sorted(
        df.loc[
            ~id_missing
            & df[id_col].duplicated(keep=False),
            id_col,
        ].unique().tolist()
    )

    duplicate_rows = int(df.duplicated().sum())

    # Constant columns ignore missing values; exclude target.
    constant_columns = []

    for col in df.columns:
        if col == target:
            continue

        non_missing_values = df.loc[~missing_mask[col], col]

        if non_missing_values.nunique() <= 1:
            constant_columns.append(col)

    constant_columns.sort()

    # Label statistics.
    labelled_mask = ~missing_mask[target]
    labelled = df.loc[labelled_mask, target]

    unlabelled_rows = int(missing_mask[target].sum())

    positive_rate = (
        float((labelled == 1).mean())
        if len(labelled)
        else 0.0
    )

    # Timestamp parsing.
    time_missing = missing_mask[time_col]
    parsed_times = pd.to_datetime(
        df.loc[~time_missing, time_col],
        errors="coerce",
    )

    bad_timestamps = int(parsed_times.isna().sum())
    valid_times = parsed_times.dropna()

    if len(valid_times):
        time_range = [
            pd.Timestamp(valid_times.min()).isoformat(),
            pd.Timestamp(valid_times.max()).isoformat(),
        ]
    else:
        time_range = [None, None]

    # Leakage checks.
    leakage_suspects = []

    if (
        int(labelled_mask.sum()) >= 20
        and labelled.nunique() == 2
    ):
        y = df.loc[labelled_mask, target].astype(int).to_numpy()

        for col in df.columns:
            if col in {target, id_col, time_col}:
                continue

            col_missing = missing_mask.loc[labelled_mask, col]
            values = df.loc[labelled_mask, col]

            # A column with only one value, counting missing
            # as a value, must never be flagged.
            distinct_values = values[~col_missing].nunique()
            has_missing = bool(col_missing.any())
            total_distinct = distinct_values + int(has_missing)

            if total_distinct <= 1:
                continue

            suspicious = False

            # (a) Numeric or boolean value leakage.
            is_numeric = (
                pd.api.types.is_numeric_dtype(df[col].dtype)
                or pd.api.types.is_bool_dtype(df[col].dtype)
            )

            if is_numeric:
                non_missing = ~col_missing
                x_values = values.loc[non_missing]
                y_values = y[non_missing.to_numpy()]

                if (
                    len(x_values) >= 20
                    and len(np.unique(y_values)) == 2
                ):
                    auc = roc_auc_score(
                        y_values,
                        x_values.astype(float).to_numpy(),
                    )

                    if auc >= 0.98 or auc <= 0.02:
                        suspicious = True

            # (b) Missingness leakage, for any column type.
            missing_indicator = col_missing.astype(int).to_numpy()

            if len(np.unique(missing_indicator)) == 2:
                missing_auc = roc_auc_score(
                    y,
                    missing_indicator,
                )

                if missing_auc >= 0.98 or missing_auc <= 0.02:
                    suspicious = True

            if suspicious:
                leakage_suspects.append(col)

    leakage_suspects.sort()

    return {
        "n_rows": int(n_rows),
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": float(positive_rate),
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": leakage_suspects,
    }
