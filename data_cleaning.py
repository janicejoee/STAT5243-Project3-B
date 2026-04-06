"""
Project 2 - Part 2: Data Cleaning and Preprocessing
====================================================

Pure-function backend module providing a complete data-cleaning toolkit.

Every public function follows a **stateless, pure-function** contract:

    pd.DataFrame in  -->  pd.DataFrame (or summary dict) out

This means there is no UI code, no Dash/Shiny state, and no side-effects
beyond the returned object.  The module is designed to be orchestrated by
an external API layer, interactive notebook, or the built-in
``run_pipeline`` convenience function.

Sections
--------
1. Data Loading        – CSV, Excel, JSON, RDS, built-in Iris
2. Data Inspection     – shape, dtypes, missing counts, descriptive stats
3. Missing-Value Handling – drop or impute (mean / median / mode / constant)
4. Duplicate Handling  – detect and remove duplicate rows
5. Scaling             – StandardScaler, MinMaxScaler, RobustScaler
6. Categorical Encoding – LabelEncoder, one-hot (get_dummies)
7. Outlier Handling    – IQR-based detection, removal, and winsorization
8. Export              – write cleaned data to CSV
9. Pipeline Runner     – declarative step-by-step cleaning pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Union


# ---------------------------------------------------------------------------
#  1. Data Loading
# ---------------------------------------------------------------------------


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the ``.csv`` file.
    **kwargs
        Additional keyword arguments forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse CSV file: {exc}. "
            "Ensure the file is a valid, UTF-8 encoded CSV."
        ) from exc


def load_excel(filepath: str, **kwargs) -> pd.DataFrame:
    """Load an Excel file (.xlsx / .xls) into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the Excel file.
    **kwargs
        Additional keyword arguments forwarded to ``pd.read_excel``.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    try:
        return pd.read_excel(filepath, **kwargs)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse Excel file: {exc}. "
            "Ensure the file is a valid .xlsx or .xls workbook."
        ) from exc


def load_json(filepath: str, **kwargs) -> pd.DataFrame:
    """Load a JSON file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the ``.json`` file.
    **kwargs
        Additional keyword arguments forwarded to ``pd.read_json``.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    try:
        return pd.read_json(filepath, **kwargs)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse JSON as a flat table: {exc}. "
            "Ensure the JSON file contains a flat array of records or a column-oriented object."
        ) from exc


# ---------------------------------------------------------------------------
#  2. Data Overview / Inspection
# ---------------------------------------------------------------------------


def get_overview(df: pd.DataFrame) -> dict:
    """Return a summary dict with basic dataset information.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    dict
        Keys: ``n_rows``, ``n_cols``, ``n_missing``, ``n_duplicates``,
        ``numeric_columns``, ``categorical_columns``.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "n_missing": int(df.isnull().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
    }


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column summary of dtype, nulls, and unique counts.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    pd.DataFrame
        One row per column with fields ``Column``, ``Dtype``,
        ``Non-Null``, ``Missing``, ``Missing %``, and ``Unique``.
    """
    return pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Missing": df.isnull().sum().values,
        # Guard against zero-length DataFrames with max(..., 1)
        "Missing %": (df.isnull().sum().values / max(len(df), 1) * 100).round(2),
        "Unique": df.nunique().values,
    })


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for all columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    pd.DataFrame
        Transposed output of ``df.describe(include='all')`` with an
        added ``Column`` field.
    """
    return (
        df.describe(include="all")
        .T
        .reset_index()
        .rename(columns={"index": "Column"})
    )


# ---------------------------------------------------------------------------
#  3. Missing-Value Handling
# ---------------------------------------------------------------------------


def handle_missing(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    strategy: str = "drop_rows",
    constant_value: Optional[str] = None,
) -> pd.DataFrame:
    """Handle missing values and return a cleaned DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    columns : list[str] or None
        Column names to operate on.  ``None`` means all columns.
    strategy : str
        One of:

        * ``'drop_rows'``  -- drop rows that have NaN in *columns*
        * ``'drop_cols'``  -- drop columns (from *columns*) that contain any NaN
        * ``'mean'``       -- fill NaN with column mean  (numeric only)
        * ``'median'``     -- fill NaN with column median (numeric only)
        * ``'mode'``       -- fill NaN with column mode
        * ``'constant'``   -- fill NaN with *constant_value*
    constant_value : str or None
        Value used when ``strategy='constant'``.  Defaults to ``"0"``.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with missing values handled.
    """
    df = df.copy()
    cols = columns if columns else df.columns.tolist()

    if strategy == "drop_rows":
        df.dropna(subset=cols, inplace=True)
    elif strategy == "drop_cols":
        to_drop = [c for c in cols if df[c].isnull().any()]
        df.drop(columns=to_drop, inplace=True, errors="ignore")
    elif strategy == "mean":
        for c in cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                fill_val = df[c].mean()
                if pd.notna(fill_val):
                    df[c] = df[c].fillna(fill_val)
                # Skip all-null columns where mean is NaN
    elif strategy == "median":
        for c in cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                fill_val = df[c].median()
                if pd.notna(fill_val):
                    df[c] = df[c].fillna(fill_val)
    elif strategy == "mode":
        for c in cols:
            if c in df.columns and not df[c].mode().empty:
                # mode() returns a Series; take the first (most frequent) value
                df[c] = df[c].fillna(df[c].mode()[0])
    elif strategy == "constant":
        val = constant_value if constant_value is not None else "0"
        for c in cols:
            if c in df.columns:
                df[c] = df[c].fillna(val)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
#  3b. k-NN Imputation
# ---------------------------------------------------------------------------


def knn_impute(
    df: pd.DataFrame,
    columns: list[str],
    k: int = 5,
) -> tuple["pd.DataFrame", Optional[str]]:
    """Impute missing values in *columns* using k-Nearest-Neighbour averaging.

    Uses ``sklearn.neighbors.NearestNeighbors`` with a KD-tree / ball-tree
    internally, so memory usage is O(n · k) rather than O(n²), making it safe
    for large datasets (tens of thousands of rows).  Features are
    StandardScaler-normalised before distance computation so that columns with
    large numeric ranges do not dominate.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    columns : list[str]
        Numeric columns whose missing values should be imputed.
    k : int, default 5
        Number of nearest neighbours to average over.

    Returns
    -------
    tuple[pd.DataFrame, str | None]
        ``(result_df, warning_message)`` where *warning_message* is ``None``
        when all rows were successfully imputed, or a descriptive string when
        fewer than 50 % of rows could be imputed (the un-imputable rows are
        dropped before returning).

    Raises
    ------
    ValueError
        If no other numeric columns exist to use as features, or if none of
        those feature columns contain ≥ 80 % valid values in the rows that
        need imputation.
    """
    df = df.copy()

    # Validate that all target columns exist and are numeric
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Column(s) not found: {missing_cols}")
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"k-NN imputation requires numeric target columns. "
            f"Column(s) {non_numeric} are not numeric."
        )

    # Step 1 — find all other numeric columns (not in target set)
    all_other_numeric = [
        c for c in df.columns
        if c not in columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not all_other_numeric:
        raise ValueError(
            f"k-NN imputation cannot be performed for column(s) {columns}: "
            "no other numeric-valued columns exist in the dataset."
        )

    # Step 2 — identify rows that need imputation
    impute_mask = df[columns].isnull().any(axis=1)
    n_impute = int(impute_mask.sum())

    if n_impute == 0:
        return df, None  # nothing to impute

    # Step 3 — keep only feature columns with ≥ 80 % valid values
    # in the rows that need imputation
    feature_cols: list[str] = []
    for c in all_other_numeric:
        n_valid = int(df.loc[impute_mask, c].notna().sum())
        if n_impute > 0 and n_valid / n_impute >= 0.80:
            feature_cols.append(c)

    if not feature_cols:
        raise ValueError(
            f"k-NN imputation cannot be performed for column(s) {columns}: "
            "no other numeric-valued columns contain valid values for ≥ 80 % "
            f"of the {n_impute} entries where {columns} has missing values."
        )

    # Step 4 — count rows where ALL feature columns are valid (can be imputed)
    can_impute_mask = impute_mask & df[feature_cols].notna().all(axis=1)
    n_can_impute = int(can_impute_mask.sum())
    ratio = n_can_impute / n_impute if n_impute > 0 else 1.0

    warning_msg: Optional[str] = None
    if ratio < 0.50:
        warning_msg = (
            f"Warning: For column(s) {columns} imputed using {feature_cols}, "
            f"only {n_can_impute}/{n_impute} ({ratio:.1%}) of rows requiring "
            f"imputation have valid values in all feature columns. "
            f"The {n_impute - n_can_impute} rows that cannot be k-NN imputed "
            "will be dropped."
        )

    # Step 5 — drop rows that cannot be imputed, then reset the index
    cannot_impute_mask = impute_mask & ~can_impute_mask
    df = df[~cannot_impute_mask].reset_index(drop=True)

    # Re-identify rows that still need imputation after dropping
    still_impute_mask = df[columns].isnull().any(axis=1)
    need_impute_idx = df.index[still_impute_mask].tolist()

    if not need_impute_idx:
        return df, warning_msg

    # Step 6 — build the reference set: rows with valid targets AND features
    ref_mask = df[columns].notna().all(axis=1) & df[feature_cols].notna().all(axis=1)
    df_ref = df[ref_mask]

    if df_ref.empty:
        raise ValueError(
            "No reference rows with valid values in both target and feature columns "
            "are available after filtering. k-NN imputation cannot proceed."
        )

    # Step 7 — scale features to unit variance so no single column dominates
    # distances, then use sklearn NearestNeighbors (KD-tree) for O(n log n)
    # neighbour lookup — avoids the O(n²·d) pairwise matrix that would OOM
    # on datasets with tens of thousands of rows.
    scaler = StandardScaler()
    ref_features_scaled = scaler.fit_transform(
        df_ref[feature_cols].values.astype(float)
    )
    ref_targets_np = df_ref[columns].values.astype(float)  # (n_ref, t)

    impute_features_raw = df.loc[need_impute_idx, feature_cols].values.astype(float)
    impute_features_scaled = scaler.transform(impute_features_raw)

    k_actual = min(k, len(df_ref))
    nn_model = NearestNeighbors(
        n_neighbors=k_actual,
        algorithm="auto",   # picks ball_tree / kd_tree / brute automatically
        metric="euclidean",
        n_jobs=1,
    )
    nn_model.fit(ref_features_scaled)
    # kneighbors returns (distances, indices) — indices into df_ref
    _, nn_idx = nn_model.kneighbors(impute_features_scaled)  # (n_imp, k_actual)

    # Average target values over k neighbours: (n_imp, t)
    neighbor_targets = ref_targets_np[nn_idx]
    imputed_means = neighbor_targets.mean(axis=1)

    # Vectorised assignment — avoid per-cell pandas .loc which is very slow
    imp_arr = df.loc[need_impute_idx, columns].values.astype(float)  # (n_imp, t)
    nan_mask = np.isnan(imp_arr)
    imp_arr[nan_mask] = imputed_means[nan_mask]
    df.loc[need_impute_idx, columns] = imp_arr

    return df, warning_msg


# ---------------------------------------------------------------------------
#  4. Duplicate Handling
# ---------------------------------------------------------------------------


def get_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return all duplicate rows, including every copy.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    pd.DataFrame
        Subset of *df* where ``duplicated(keep=False)`` is True.
    """
    return df[df.duplicated(keep=False)]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and return a new DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    pd.DataFrame
        De-duplicated DataFrame with a reset integer index.
    """
    return df.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
#  5. Scaling / Normalization
# ---------------------------------------------------------------------------


def scale_columns(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "standard",
) -> pd.DataFrame:
    """Scale numeric columns and return a new DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    columns : list[str]
        Numeric column names to scale.
    method : str
        Scaling algorithm to apply:

        * ``'standard'`` -- **StandardScaler**: centres each column to
          mean = 0, std = 1 using ``(x - mean) / std``.
        * ``'minmax'``   -- **MinMaxScaler**: rescales each column to
          the [0, 1] range via ``(x - min) / (max - min)``.
        * ``'robust'``   -- **RobustScaler**: centres using the median
          and scales by the IQR ``(x - median) / IQR``, making it
          resistant to outliers.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the specified columns scaled.
    """
    df = df.copy()
    scaler_map = {
        "standard": StandardScaler,   # z-score: (x - mean) / std
        "minmax": MinMaxScaler,       # rescale to [0, 1]: (x - min) / (max - min)
        "robust": RobustScaler,       # outlier-resistant: (x - median) / IQR
    }
    if method not in scaler_map:
        raise ValueError(f"Unknown method: {method}. Choose from {list(scaler_map)}")

    # Instantiate the chosen scaler, fit on the selected columns, and
    # replace those columns with the transformed values.
    # Validate that all selected columns are numeric before scaling
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"Cannot scale non-numeric columns: {non_numeric}. Select only numeric columns."
        )
    all_null = [c for c in columns if df[c].isnull().all()]
    if all_null:
        raise ValueError(
            f"Cannot scale all-null columns: {all_null}. These columns have no valid values."
        )

    scaler = scaler_map[method]()
    df[columns] = scaler.fit_transform(df[columns])
    return df


# ---------------------------------------------------------------------------
#  6. Categorical Encoding
# ---------------------------------------------------------------------------


def encode_columns(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "label",
) -> pd.DataFrame:
    """Encode categorical columns and return a new DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    columns : list[str]
        Categorical column names to encode.
    method : str
        Encoding strategy:

        * ``'label'``  -- **LabelEncoder**: maps each unique category to
          a consecutive integer (0, 1, 2, ...).  Each column is encoded
          independently.
        * ``'onehot'`` -- **One-hot (pd.get_dummies)**: creates a new
          binary (0/1) column for *every* category in the original column.
          The original column is dropped.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the specified columns encoded.
    """
    df = df.copy()
    if method == "label":
        # LabelEncoder works on a single column at a time, so we loop.
        # Values are cast to str first to handle mixed types gracefully.
        le = LabelEncoder()
        for c in columns:
            if c in df.columns:
                df[c] = le.fit_transform(df[c].astype(str))
    elif method == "onehot":
        # get_dummies expands each categorical column into k binary columns
        # (one per unique value).  dtype=int ensures 0/1 integers rather
        # than True/False booleans.
        df = pd.get_dummies(df, columns=columns, drop_first=False, dtype=int)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'label' or 'onehot'.")
    return df


# ---------------------------------------------------------------------------
#  7. Outlier Handling
# ---------------------------------------------------------------------------


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    iqr_multiplier: float = 1.5,
) -> dict:
    """Detect outliers via the IQR method and return diagnostic info.

    The **Interquartile Range (IQR)** method works as follows:

    1. Compute Q1 (25th percentile) and Q3 (75th percentile).
    2. IQR = Q3 - Q1  (the spread of the middle 50 % of data).
    3. Lower bound = Q1 - ``iqr_multiplier`` * IQR.
    4. Upper bound = Q3 + ``iqr_multiplier`` * IQR.
    5. Any value below the lower bound or above the upper bound is
       flagged as an outlier.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    column : str
        Numeric column to inspect.
    iqr_multiplier : float, default 1.5
        Multiplier applied to the IQR to determine the fence width.
        A value of 1.5 flags mild outliers; 3.0 flags extreme outliers.

    Returns
    -------
    dict
        Keys: ``q1``, ``q3``, ``iqr``, ``lower_bound``, ``upper_bound``,
        ``n_outliers``, ``outlier_mask`` (boolean Series).
    """
    # Step 1: Compute the first and third quartiles
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    # Step 2: IQR is the spread of the middle 50% of the distribution
    iqr = q3 - q1

    # Step 3-4: Fences define the acceptable range; anything outside is an outlier
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    # Step 5: Boolean mask -- True for rows that fall outside the fences
    mask = (df[column] < lower) | (df[column] > upper)

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_outliers": int(mask.sum()),
        "outlier_mask": mask,
    }


def handle_outliers(
    df: pd.DataFrame,
    column: str,
    action: str = "remove",
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Handle outliers in a single numeric column.

    Uses the IQR method (see ``detect_outliers``) to identify outlier
    rows, then either removes them or caps (winsorizes) the values.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    column : str
        Numeric column to clean.
    action : str
        * ``'remove'`` -- drop rows whose value falls outside the IQR
          fences.
        * ``'cap'``    -- clip (winsorize) outlier values to the nearest
          fence boundary so no data rows are lost.
    iqr_multiplier : float, default 1.5
        Multiplier forwarded to ``detect_outliers``.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with outliers handled.
    """
    df = df.copy()

    # Reuse detect_outliers to get the IQR-based fence boundaries
    info = detect_outliers(df, column, iqr_multiplier)
    lower, upper = info["lower_bound"], info["upper_bound"]

    if action == "remove":
        # Keep only rows within [lower, upper]
        df = df[(df[column] >= lower) & (df[column] <= upper)].reset_index(drop=True)
    elif action == "cap":
        # Winsorize: clip values so they sit at the fence boundaries
        df[column] = df[column].clip(lower, upper)
    else:
        raise ValueError(f"Unknown action: {action}. Choose 'remove' or 'cap'.")
    return df


# ---------------------------------------------------------------------------
#  7b. Text Standardization & Type Coercion
# ---------------------------------------------------------------------------


def standardize_text(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    strip: bool = True,
    case: str = "lower",
) -> pd.DataFrame:
    """Standardize whitespace and letter case in string columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    columns : list[str] or None
        String columns to standardize.  ``None`` means all object/string columns.
    strip : bool
        If ``True``, strip leading/trailing whitespace and collapse multiple
        internal spaces into one.
    case : str
        Letter-case transform: ``'lower'``, ``'upper'``, ``'title'``, or
        ``'none'`` (skip case change).

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the specified text columns cleaned.
    """
    df = df.copy()
    cols = columns if columns else df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].astype(str)
        if strip:
            s = s.str.strip().str.replace(r"\s+", " ", regex=True)
        if case == "lower":
            s = s.str.lower()
        elif case == "upper":
            s = s.str.upper()
        elif case == "title":
            s = s.str.title()
        df[c] = s
    return df


def coerce_column_types(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    target: str = "numeric",
) -> pd.DataFrame:
    """Coerce columns to a consistent data type.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    columns : list[str] or None
        Columns to coerce.  ``None`` means all columns.
    target : str
        Target type: ``'numeric'`` (via ``pd.to_numeric``, non-convertible
        become NaN) or ``'string'`` (via ``astype(str)``).

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the specified columns coerced.
    """
    df = df.copy()
    cols = columns if columns else df.columns.tolist()
    for c in cols:
        if c not in df.columns:
            continue
        if target == "numeric":
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif target == "string":
            df[c] = df[c].astype(str)
        else:
            raise ValueError(f"Unknown target type: {target}. Use 'numeric' or 'string'.")
    return df


# ---------------------------------------------------------------------------
#  8. Export
# ---------------------------------------------------------------------------


def export_csv(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """Save the DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to export.
    filepath : str
        Destination path for the ``.csv`` file.
    **kwargs
        Additional keyword arguments forwarded to ``df.to_csv``.
    """
    df.to_csv(filepath, index=False, **kwargs)


# ---------------------------------------------------------------------------
#  9. Pipeline Runner (convenience helper)
# ---------------------------------------------------------------------------


def run_pipeline(
    df: pd.DataFrame,
    steps: list[dict],
) -> pd.DataFrame:
    """Execute a sequence of cleaning steps declaratively.

    Each step is a dict whose ``"action"`` key selects a cleaning
    function, and whose remaining keys are forwarded as keyword
    arguments to that function.

    The pipeline processes steps **sequentially**: the output DataFrame
    of step *n* becomes the input of step *n + 1*.

    Parameters
    ----------
    df : pd.DataFrame
        The initial (raw) dataset.
    steps : list[dict]
        Ordered cleaning steps.  Example::

            [
                {"action": "handle_missing", "strategy": "mean"},
                {"action": "remove_duplicates"},
                {"action": "scale_columns",
                 "columns": ["col1"], "method": "standard"},
                {"action": "encode_columns",
                 "columns": ["species"], "method": "onehot"},
                {"action": "handle_outliers",
                 "column": "col1", "action_type": "cap"},
            ]

    Returns
    -------
    pd.DataFrame
        The fully cleaned dataset after all steps have been applied.
    """
    # Dispatch table: maps action names to callables.
    # Each lambda receives (DataFrame, step_dict) and returns a new DataFrame.
    # The "action" key is stripped from kwargs before forwarding so it does not
    # collide with actual function parameters.
    dispatch = {
        "handle_missing": lambda d, p: handle_missing(d, **{k: v for k, v in p.items() if k != "action"}),
        "remove_duplicates": lambda d, _: remove_duplicates(d),
        "scale_columns": lambda d, p: scale_columns(d, **{k: v for k, v in p.items() if k != "action"}),
        "encode_columns": lambda d, p: encode_columns(d, **{k: v for k, v in p.items() if k != "action"}),
        # handle_outliers needs special treatment because the step dict uses
        # "action_type" (to avoid clashing with the top-level "action" key)
        # while the function parameter is named "action".
        "handle_outliers": lambda d, p: handle_outliers(
            d,
            column=p["column"],
            action=p.get("action_type", "remove"),
            iqr_multiplier=p.get("iqr_multiplier", 1.5),
        ),
    }

    # Walk through each step in order, threading the DataFrame through
    for step in steps:
        action = step["action"]
        if action not in dispatch:
            raise ValueError(f"Unknown pipeline action: {action}")
        df = dispatch[action](df, step)
    return df


# ---------------------------------------------------------------------------
#  10. RDS Loading
# ---------------------------------------------------------------------------


def load_rds(filepath: str, **kwargs) -> pd.DataFrame:
    """Load an RDS file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the .rds file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    try:
        import pyreadr
    except ImportError:
        raise ImportError(
            "RDS support requires the 'pyreadr' package. "
            "Install it with: pip install pyreadr"
        )
    try:
        result = pyreadr.read_r(filepath)
        if not result:
            raise ValueError("RDS file is empty or contains no readable R objects.")
        return list(result.values())[0]
    except ImportError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Failed to read RDS file: {exc}. "
            "Ensure the file is a valid R .rds file."
        ) from exc


# ---------------------------------------------------------------------------
#  Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    raw = load_iris(as_frame=True).frame
    print("Raw shape:", raw.shape)
    print("Overview:", get_overview(raw))

    cleaned = run_pipeline(raw, [
        {"action": "remove_duplicates"},
        {"action": "handle_missing", "strategy": "mean"},
        {"action": "scale_columns", "columns": ["sepal_length", "sepal_width",
                                                  "petal_length", "petal_width"],
         "method": "standard"},
        {"action": "encode_columns", "columns": ["species"], "method": "label"},
    ])
    print("Cleaned shape:", cleaned.shape)
    print(cleaned.head())
