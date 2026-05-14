from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd


def clean_text(text: str) -> str:
    """General-purpose social text cleaning."""
    text = str(text)
    text = text.lower()

    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    # keep basic punctuation that may carry sentiment
    text = re.sub(r"[^a-z0-9\s!?']", " ", text)

    # normalise repeated spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def _get_col(df: pd.DataFrame, col: str):
    #helper function that retrieves a column from pandas DataFrame
    #allows the user to specify the column either by name or integer (e.g. '5' or 5)
    if isinstance(col, int):
        return df.iloc[:, col]
    if isinstance(col, str) and col.isdigit():
        return df.iloc[:, int(col)]
    return df[col]

def _pick_text_column(df: pd.DataFrame) -> str:
    """Heuristic: pick a likely text column if user didn't specify one."""
    candidates = [c for c in df.columns if df[c].dtype == "object"]
    if not candidates:
        raise ValueError("No obvious text column found (no object/string columns).")
    # prefer common names
    common = ["text", "tweet", "content", "review", "message", "comment", "sentence"]
    for name in common:
        for c in candidates:
            if c.lower() == name:
                return c
    # otherwise pick the longest average length column
    avg_len = {}
    for c in candidates:
        s = df[c].dropna().astype(str)
        if len(s) == 0:
            continue
        avg_len[c] = s.str.len().mean()
    if not avg_len:
        raise ValueError("Text column heuristic failed (all candidate columns empty).")
    return max(avg_len, key=avg_len.get)


def _pick_label_column(df: pd.DataFrame, text_col: str) -> Optional[str]:
    """Heuristic: pick a likely label column if user didn't specify one."""
    candidates = [c for c in df.columns if c != text_col]
    if not candidates:
        return None

    # Prefer common label names
    common = ["label", "sentiment", "target", "class", "category", "y"]
    for name in common:
        for c in candidates:
            if c.lower() == name:
                return c

    # Else: find a low-cardinality column (like 2-10 unique values)
    best = None
    best_card = None
    for c in candidates:
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 10:
            if best is None or nunique < best_card:
                best = c
                best_card = nunique
    return best


def _normalize_binary_labels(series: pd.Series) -> Tuple[pd.Series, Dict]:
    """
    Map a label column to 0/1 if it looks binary.
    Returns (mapped_series, mapping_used).
    """
    s = series.dropna()

    # If numeric and contains {0,1} already
    if pd.api.types.is_numeric_dtype(s):
        uniq = sorted(set(s.astype(float).unique()))
        if set(uniq) == {0.0, 1.0}:
            return series.astype(float).astype(int), {"0": 0, "1": 1}
        # Sentiment140 style {0,4}
        if set(uniq) == {0.0, 4.0}:
            return series.map({0: 0, 4: 1}).astype("Int64"), {"0": 0, "4": 1}

    # If string-like
    s_str = s.astype(str).str.strip().str.lower()
    uniq = sorted(set(s_str.unique()))

    # Common sentiment words
    mapping_sets = [
        ({"neg", "negative", "0"}, 0, {"pos", "positive", "1"}, 1),
        ({"ham"}, 0, {"spam"}, 1),
        ({"no", "false"}, 0, {"yes", "true"}, 1),
    ]

    for left_set, left_val, right_set, right_val in mapping_sets:
        if any(u in left_set for u in uniq) and any(u in right_set for u in uniq):
            mapped = series.astype(str).str.strip().str.lower().map(
                lambda x: left_val if x in left_set else (right_val if x in right_set else None)
            )
            return mapped.astype("Int64"), {"left": list(left_set), "right": list(right_set)}

    # Fallback: if exactly 2 unique values, map alphabetically
    if len(uniq) == 2:
        mapped = series.astype(str).str.strip().str.lower().map({uniq[0]: 0, uniq[1]: 1})
        return mapped.astype("Int64"), {uniq[0]: 0, uniq[1]: 1}

    raise ValueError(
        f"Label column is not clearly binary. Unique values sample: {uniq[:10]}"
    )


def load_csv_dataset(
    csv_path: str | Path,
    text_col=None,
    label_col=None,
    encoding: Optional[str] = None,
    header="infer",
):
    """
    Generic CSV loader for text classification datasets.

    Supports:
    - CSVs with header row
    - CSVs without header row (header=None)
    - Column names OR numeric column indexes (e.g., 5 or "5")

    Returns:
        df_out: DataFrame with columns:
            - text_clean
            - label (optional if provided)
        chosen_text_col
        chosen_label_col
        label_mapping (if labels normalized)
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    if encoding:
        df = pd.read_csv(csv_path, encoding=encoding, header=header)
    else:
        df = pd.read_csv(csv_path, header=header)

    if df.empty:
        raise ValueError("CSV loaded but contains no rows.")

    if text_col is None:
        text_col = _pick_text_column(df)

    try:
        text_series = _get_col(df, text_col)
    except Exception:
        raise ValueError(
            f"text_col '{text_col}' not found. Columns: {list(df.columns)}"
        )

    if label_col is None:
        label_col = _pick_label_column(df, text_col)

    out = pd.DataFrame()
    out["text_clean"] = text_series.astype(str).map(clean_text)

    label_mapping = None

    if label_col is not None:
        try:
            label_series = _get_col(df, label_col)
        except Exception:
            raise ValueError(
                f"label_col '{label_col}' not found. Columns: {list(df.columns)}"
            )

        labels, label_mapping = _normalize_binary_labels(label_series)
        out["label"] = labels
        out = out.dropna(subset=["label"]).copy()
        out["label"] = out["label"].astype(int)

    return out, text_col, label_col, label_mapping

    