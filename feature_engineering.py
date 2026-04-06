"""Feature engineering backend for the Shiny data-exploration app.

Provides a single dispatcher function, ``apply_feature_engineering_to_df``,
that delegates to specialised private helpers for each transformation type
(log, square, cube, interaction, ratio, binning, one-hot encoding,
standardization, normalization, missing-value imputation, and row dropping).

Every helper returns a **(DataFrame, metadata-dict)** tuple so the UI layer
can display provenance information (formula, columns created, etc.) without
re-inspecting the data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Type-checking helpers
# ---------------------------------------------------------------------------

def _is_numeric(series: pd.Series) -> bool:
    """Check whether a pandas Series has a numeric dtype.

    Parameters
    ----------
    series : pd.Series
        The column to inspect.

    Returns
    -------
    bool
        True if the series dtype is numeric (int, float, etc.).
    """
    return pd.api.types.is_numeric_dtype(series)


def _is_categorical(series: pd.Series) -> bool:
    """Check whether a pandas Series is categorical-like.

    A column is considered categorical if its dtype is object, Categorical,
    boolean, or the pandas StringDtype.

    Parameters
    ----------
    series : pd.Series
        The column to inspect.

    Returns
    -------
    bool
        True if the series can be treated as a categorical variable.
    """
    return (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ``ValueError`` if any of *columns* are not numeric.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe whose columns are validated.
    columns : list[str]
        Column names that must have a numeric dtype.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Lists the non-numeric column names.
    """
    bad = [col for col in columns if not _is_numeric(df[col])]
    if bad:
        raise ValueError(f"These columns must be numeric: {bad}")


def _require_categorical(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ``ValueError`` if any of *columns* are not categorical-like.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe whose columns are validated.
    columns : list[str]
        Column names that must be categorical-like.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Lists the non-categorical column names.
    """
    bad = [col for col in columns if not _is_categorical(df[col])]
    if bad:
        raise ValueError(f"These columns must be categorical-like: {bad}")


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ``ValueError`` if any of *columns* are missing from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to check against.
    columns : list[str]
        Column names that must be present.  Empty/falsy strings are skipped.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Lists the column names that do not exist in *df*.
    """
    missing = [col for col in columns if col and col not in df.columns]
    if missing:
        raise ValueError(f"Invalid column(s): {missing}")


# ---------------------------------------------------------------------------
# Individual feature-engineering transforms
# ---------------------------------------------------------------------------

def _apply_log_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a log-transformed feature using ``numpy.log1p``.

    ``log1p(x)`` computes ``log(1 + x)``, which is preferred over plain
    ``log`` because it gracefully handles zeros (log1p(0) == 0) and avoids
    the -inf that ``log(0)`` would produce.  Values must be >= -1.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (not modified in place).
    col1 : str
        Name of the numeric column to transform.
    new_column : str or None, optional
        Name for the output column.  Defaults to ``"log_{col1}"``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A copy of *df* with the new column appended, and a metadata dict
        describing the transformation.
    """
    _require_numeric(df, [col1])
    series = pd.to_numeric(df[col1], errors="coerce")
    valid = series.dropna()
    # log1p is defined for x >= -1; reject data that would produce NaN.
    if (valid < -1).any():
        raise ValueError(f"log1p requires all non-null values in '{col1}' to be >= -1.")

    out = df.copy()
    output_name = new_column or f"log_{col1}"
    # Use log1p instead of log to safely handle zeros.
    out[output_name] = np.log1p(series)
    return out, {
        "feature_type": "log",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = log1p({col1})",
    }


def _apply_square_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a squared feature (x^2).

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (not modified in place).
    col1 : str
        Name of the numeric column to square.
    new_column : str or None, optional
        Name for the output column.  Defaults to ``"{col1}_squared"``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A copy of *df* with the new column appended, and a metadata dict.
    """
    _require_numeric(df, [col1])
    out = df.copy()
    output_name = new_column or f"{col1}_squared"
    out[output_name] = pd.to_numeric(out[col1], errors="coerce") ** 2
    return out, {
        "feature_type": "square",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = {col1}^2",
    }


def _apply_cube_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a cubed feature (x^3).

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (not modified in place).
    col1 : str
        Name of the numeric column to cube.
    new_column : str or None, optional
        Name for the output column.  Defaults to ``"{col1}_cubed"``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A copy of *df* with the new column appended, and a metadata dict.
    """
    _require_numeric(df, [col1])
    out = df.copy()
    output_name = new_column or f"{col1}_cubed"
    out[output_name] = pd.to_numeric(out[col1], errors="coerce") ** 3
    return out, {
        "feature_type": "cube",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = {col1}^3",
    }


def _apply_interaction_feature(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create an interaction feature by multiplying two numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (not modified in place).
    col1 : str
        First numeric column.
    col2 : str
        Second numeric column.
    new_column : str or None, optional
        Name for the output column.  Defaults to ``"{col1}_x_{col2}"``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A copy of *df* with the new column appended, and a metadata dict.
    """
    _require_numeric(df, [col1, col2])
    out = df.copy()
    output_name = new_column or f"{col1}_x_{col2}"
    out[output_name] = pd.to_numeric(out[col1], errors="coerce") * pd.to_numeric(
        out[col2], errors="coerce"
    )
    return out, {
        "feature_type": "interaction",
        "input_columns": [col1, col2],
        "output_columns": [output_name],
        "formula": f"{output_name} = {col1} * {col2}",
    }


def _apply_ratio_feature(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a ratio feature (col1 / col2) with zero-denominator protection.

    Rows where *col2* is exactly zero are replaced with ``NaN`` before
    division to avoid infinities.  The metadata dict reports how many
    zero-denominator rows were encountered.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (not modified in place).
    col1 : str
        Numerator column.
    col2 : str
        Denominator column.
    new_column : str or None, optional
        Name for the output column.  Defaults to ``"{col1}_div_{col2}"``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A copy of *df* with the new column appended, and a metadata dict
        that includes ``n_zero_denominator``.
    """
    _require_numeric(df, [col1, col2])
    out = df.copy()
    numerator = pd.to_numeric(out[col1], errors="coerce")
    denominator = pd.to_numeric(out[col2], errors="coerce")
    # Mask zeros in the denominator to NaN so division yields NaN instead of inf.
    zero_mask = denominator == 0
    output_name = new_column or f"{col1}_div_{col2}"
    out[output_name] = numerator / denominator.mask(zero_mask, np.nan)
    return out, {
        "feature_type": "ratio",
        "input_columns": [col1, col2],
        "output_columns": [output_name],
        "formula": f"{output_name} = {col1} / {col2}",
        "n_zero_denominator": int(zero_mask.sum()),
        "note": "Zero denominators are converted to NaN.",
    }


def _apply_binning_feature(
    df: pd.DataFrame,
    col1: str,
    bins: int,
    *,
    new_column: str | None = None,
    labels: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bin a numeric column into equal-width intervals using ``pd.cut``.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (not modified in place).
    col1 : str
        Numeric column to bin.
    bins : int
        Number of equal-width bins (must be >= 2).
    new_column : str or None, optional
        Name for the output column.  Defaults to ``"{col1}_binned"``.
    labels : bool, optional
        If True, bin labels are interval strings (e.g. ``(0.5, 1.5]``).
        If False (the default), labels are integer codes 0 .. bins-1.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A copy of *df* with the new column appended, and a metadata dict.

    Raises
    ------
    ValueError
        If *bins* < 2, since at least two intervals are needed for a
        meaningful partition.
    """
    _require_numeric(df, [col1])
    # At least 2 bins are needed; a single bin would be degenerate.
    if bins < 2:
        raise ValueError("bins must be at least 2.")

    out = df.copy()
    series = pd.to_numeric(out[col1], errors="coerce")
    output_name = new_column or f"{col1}_binned"
    if labels:
        # include_lowest=True ensures the minimum value falls inside the first bin.
        out[output_name] = pd.cut(series, bins=bins, include_lowest=True)
    else:
        # labels=False returns integer bin codes (0-indexed) instead of intervals.
        out[output_name] = pd.cut(series, bins=bins, labels=False, include_lowest=True)
    return out, {
        "feature_type": "binning",
        "input_columns": [col1],
        "output_columns": [output_name],
        "bins": int(bins),
        "labels": bool(labels),
        "formula": f"{output_name} = pd.cut({col1}, bins={bins})",
    }


def _apply_one_hot_encoding(
    df: pd.DataFrame,
    col1: str,
    *,
    prefix: str | None = None,
    drop_first: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One-hot encode a categorical column.

    Each unique value in *col1* becomes a new 0/1 indicator column.
    The original column is kept; the dummies are appended.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (not modified in place).
    col1 : str
        Categorical column to encode.
    prefix : str or None, optional
        Prefix for the generated dummy column names.  Defaults to *col1*.
    drop_first : bool, optional
        If True, drop the first category to avoid perfect multicollinearity
        (useful for regression models).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A copy of *df* with dummy columns appended, and a metadata dict
        listing all generated column names.
    """
    _require_categorical(df, [col1])
    # Guard against high-cardinality columns that would explode the column count
    n_unique = df[col1].nunique()
    if n_unique > 50:
        raise ValueError(
            f"Column '{col1}' has {n_unique} unique values. One-hot encoding "
            f"would create {n_unique} new columns, which is likely unintended. "
            f"Consider using label encoding or binning the column first."
        )
    out = df.copy()
    dummy_prefix = prefix or col1
    dummies = pd.get_dummies(
        out[col1],
        prefix=dummy_prefix,
        drop_first=drop_first,
        dummy_na=False,
    )
    out = pd.concat([out, dummies], axis=1)
    return out, {
        "feature_type": "one_hot",
        "input_columns": [col1],
        "output_columns": [str(c) for c in dummies.columns],
        "drop_first": bool(drop_first),
        "prefix": dummy_prefix,
        "formula": f"get_dummies({col1}, prefix='{dummy_prefix}', drop_first={drop_first})",
    }


def _apply_standardize_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardize a numeric column to zero mean and unit variance (z-score).

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (not modified in place).
    col1 : str
        Numeric column to standardize.
    new_column : str or None, optional
        Name for the output column.  Defaults to ``"{col1}_zscore"``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A copy of *df* with the new column appended, and a metadata dict
        that includes the computed mean and standard deviation.

    Raises
    ------
    ValueError
        If the standard deviation is zero or NaN (constant or all-null column).
    """
    _require_numeric(df, [col1])
    series = pd.to_numeric(df[col1], errors="coerce")
    mean_val = series.mean()
    std_val = series.std()
    # A zero or NaN std means the column is constant or empty; z-score is undefined.
    if pd.isna(std_val) or std_val == 0:
        raise ValueError(f"Column '{col1}' cannot be standardized because std is 0 or NaN.")

    out = df.copy()
    output_name = new_column or f"{col1}_zscore"
    out[output_name] = (series - mean_val) / std_val
    return out, {
        "feature_type": "standardize",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = ({col1} - mean({col1})) / std({col1})",
        "mean": float(mean_val),
        "std": float(std_val),
    }


def _apply_normalize_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize a numeric column to the [0, 1] range via min-max scaling.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (not modified in place).
    col1 : str
        Numeric column to normalize.
    new_column : str or None, optional
        Name for the output column.  Defaults to ``"{col1}_normalized"``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A copy of *df* with the new column appended, and a metadata dict
        that includes the observed min and max values.

    Raises
    ------
    ValueError
        If min equals max (constant column) or values are all-null.
    """
    _require_numeric(df, [col1])
    series = pd.to_numeric(df[col1], errors="coerce")
    min_val = series.min()
    max_val = series.max()
    # min == max means zero range; division by zero would occur.
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        raise ValueError(
            f"Column '{col1}' cannot be normalized because min=max or values are invalid."
        )

    out = df.copy()
    output_name = new_column or f"{col1}_normalized"
    out[output_name] = (series - min_val) / (max_val - min_val)
    return out, {
        "feature_type": "normalize",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = ({col1} - min({col1})) / (max({col1}) - min({col1}))",
        "min": float(min_val),
        "max": float(max_val),
    }


def _apply_fillna_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    strategy: str = "mean",
    fill_value: Any | None = None,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill missing values in a column using the chosen imputation strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (not modified in place).
    col1 : str
        Column whose NaN values should be filled.
    strategy : str, optional
        One of ``"mean"``, ``"median"``, ``"mode"``, or ``"constant"``.
        ``"mean"`` and ``"median"`` require a numeric column.
    fill_value : Any or None, optional
        The literal value to fill when ``strategy="constant"``.  Required
        for that strategy; ignored otherwise.
    new_column : str or None, optional
        Name for the output column.  If None, the column is filled in place
        (i.e. the original column name is reused).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A copy of *df* with the filled column, and a metadata dict that
        includes the actual ``fill_value_used``.
    """
    if col1 not in df.columns:
        raise ValueError(f"Column '{col1}' does not exist.")

    out = df.copy()
    series = out[col1]
    output_name = new_column or col1
    strategy = strategy.lower()

    if strategy == "mean":
        _require_numeric(out, [col1])
        value = pd.to_numeric(series, errors="coerce").mean()
    elif strategy == "median":
        _require_numeric(out, [col1])
        value = pd.to_numeric(series, errors="coerce").median()
    elif strategy == "mode":
        mode_vals = series.mode(dropna=True)
        # mode() returns an empty Series when all values are NaN.
        if len(mode_vals) == 0:
            raise ValueError(f"Column '{col1}' has no mode available for filling.")
        value = mode_vals.iloc[0]
    elif strategy == "constant":
        if fill_value is None:
            raise ValueError("fill_value is required when strategy='constant'.")
        value = fill_value
    else:
        raise ValueError("strategy must be one of: mean, median, mode, constant.")

    if output_name == col1:
        out[col1] = series.fillna(value)
    else:
        out[output_name] = series.fillna(value)
    return out, {
        "feature_type": "fillna",
        "input_columns": [col1],
        "output_columns": [output_name],
        "strategy": strategy,
        "fill_value_used": value,
        "formula": f"{output_name} = fillna({col1}, strategy='{strategy}')",
    }


def _apply_dropna_feature(
    df: pd.DataFrame,
    col1: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop rows that contain missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (not modified in place).
    col1 : str or None, optional
        If provided, only rows where *col1* is NaN are dropped.
        If None, any row with at least one NaN in *any* column is dropped.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A copy of *df* with the affected rows removed, and a metadata dict
        that includes ``rows_removed``.
    """
    out = df.copy()
    before_rows = len(out)
    if col1:
        if col1 not in out.columns:
            raise ValueError(f"Column '{col1}' does not exist.")
        out = out.dropna(subset=[col1])
        scope = [col1]
    else:
        out = out.dropna()
        scope = "all_columns"
    return out, {
        "feature_type": "dropna",
        "input_columns": [col1] if col1 else [],
        "output_columns": [],
        "scope": scope,
        "rows_removed": int(before_rows - len(out)),
        "formula": f"dropna(subset={scope})",
    }


# ---------------------------------------------------------------------------
# Custom algebraic expression transform
# ---------------------------------------------------------------------------

def _apply_custom_expr_feature(
    df: pd.DataFrame,
    expr: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a new column from a custom algebraic expression evaluated with ``df.eval``.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (not modified in place).
    expr : str
        A pandas-eval-compatible expression referencing existing column names.
        Examples: ``"col_a * 2 + col_b"``, ``"(price - cost) / price"``.
    new_column : str or None, optional
        Name for the output column.  Required — if not provided the function
        raises ``ValueError``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A copy of *df* with the new column appended, and a metadata dict.

    Raises
    ------
    ValueError
        If *new_column* is empty, if the expression references non-existent
        columns, or if ``df.eval`` cannot evaluate the expression.
    """
    if not new_column or not new_column.strip():
        raise ValueError(
            "A column name is required when using Custom New Column. "
            "Enter a name in the 'New column name' field."
        )
    if not expr or not expr.strip():
        raise ValueError("Enter an algebraic expression to evaluate.")

    # Check that the expression only references existing columns
    # by trying df.eval first and catching helpful errors.
    try:
        result = df.eval(expr)
    except pd.errors.UndefinedVariableError as exc:
        raise ValueError(
            f"UndefinedVariableError: column not found — {exc}. "
            "Check column names and expression."
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Could not evaluate expression '{expr}': {exc}. "
            "Check column names and operators (use +, -, *, /, **, // etc.)."
        ) from exc

    if not isinstance(result, pd.Series):
        raise ValueError(
            "The expression must produce a single column (Series). "
            "Make sure it is a scalar-per-row computation, not a DataFrame transform."
        )

    out = df.copy()
    col_name = new_column.strip()
    out[col_name] = result
    return out, {
        "feature_type": "custom_expr",
        "input_columns": [],
        "output_columns": [col_name],
        "formula": f"{col_name} = {expr}",
    }


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def apply_feature_engineering_to_df(
    df: pd.DataFrame,
    method: str,
    col1: str | None = None,
    *,
    col2: str | None = None,
    bins: int = 4,
    new_column: str | None = None,
    labels: bool = False,
    prefix: str | None = None,
    drop_first: bool = False,
    strategy: str = "mean",
    fill_value: Any | None = None,
    expr: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a single feature-engineering transformation to a DataFrame.

    This is the main entry point used by the Shiny UI.  It validates inputs,
    normalises the *method* name, and delegates to the appropriate private
    ``_apply_*`` helper.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (not modified in place).
    method : str
        Transformation name.  One of: ``log``, ``square``, ``cube``,
        ``interaction``, ``ratio``, ``binning``, ``one_hot`` (aliases
        ``onehot`` / ``one-hot``), ``standardize``, ``normalize``,
        ``fillna``, ``dropna``.
    col1 : str or None, optional
        Primary column for the transformation.  Required for all methods
        except ``dropna``.
    col2 : str or None, optional
        Secondary column, required only for ``interaction`` and ``ratio``.
    bins : int, optional
        Number of bins (only used by ``binning``).  Default 4.
    new_column : str or None, optional
        Custom name for the generated column.
    labels : bool, optional
        If True, binning returns interval labels instead of integer codes.
    prefix : str or None, optional
        Prefix for one-hot encoded dummy columns.
    drop_first : bool, optional
        Whether to drop the first dummy column (one-hot encoding only).
    strategy : str, optional
        Imputation strategy for ``fillna``.  Default ``"mean"``.
    fill_value : Any or None, optional
        Literal fill value when ``strategy="constant"``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A transformed copy of *df* and a metadata dict describing what was
        done (feature type, formula, output columns, etc.).

    Raises
    ------
    ValueError
        On invalid method name, missing required columns, or dtype mismatches.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    method_norm = method.lower()
    cols_to_check = []
    if col1:
        cols_to_check.append(col1)
    if col2:
        cols_to_check.append(col2)
    _validate_columns(df, cols_to_check)

    if method_norm == "log":
        if not col1:
            raise ValueError("col1 is required for log.")
        return _apply_log_feature(df, col1, new_column=new_column)
    if method_norm == "square":
        if not col1:
            raise ValueError("col1 is required for square.")
        return _apply_square_feature(df, col1, new_column=new_column)
    if method_norm == "cube":
        if not col1:
            raise ValueError("col1 is required for cube.")
        return _apply_cube_feature(df, col1, new_column=new_column)
    if method_norm == "interaction":
        if not col1 or not col2:
            raise ValueError("col1 and col2 are required for interaction.")
        return _apply_interaction_feature(df, col1, col2, new_column=new_column)
    if method_norm == "ratio":
        if not col1 or not col2:
            raise ValueError("col1 and col2 are required for ratio.")
        return _apply_ratio_feature(df, col1, col2, new_column=new_column)
    if method_norm == "binning":
        if not col1:
            raise ValueError("col1 is required for binning.")
        return _apply_binning_feature(
            df,
            col1,
            bins=bins,
            new_column=new_column,
            labels=labels,
        )
    if method_norm in {"one_hot", "onehot", "one-hot"}:
        if not col1:
            raise ValueError("col1 is required for one_hot.")
        return _apply_one_hot_encoding(df, col1, prefix=prefix, drop_first=drop_first)
    if method_norm == "standardize":
        if not col1:
            raise ValueError("col1 is required for standardize.")
        return _apply_standardize_feature(df, col1, new_column=new_column)
    if method_norm == "normalize":
        if not col1:
            raise ValueError("col1 is required for normalize.")
        return _apply_normalize_feature(df, col1, new_column=new_column)
    if method_norm == "fillna":
        if not col1:
            raise ValueError("col1 is required for fillna.")
        return _apply_fillna_feature(
            df,
            col1,
            strategy=strategy,
            fill_value=fill_value,
            new_column=new_column,
        )
    if method_norm == "dropna":
        return _apply_dropna_feature(df, col1=col1)

    if method_norm == "custom_expr":
        return _apply_custom_expr_feature(df, expr or "", new_column=new_column)

    raise ValueError(
        "Invalid feature engineering method. Allowed: "
        "log, square, cube, interaction, ratio, binning, one_hot, "
        "standardize, normalize, fillna, dropna, custom_expr."
    )


# ---------------------------------------------------------------------------
# Capability introspection (used by the UI to build dynamic controls)
# ---------------------------------------------------------------------------

def feature_engineering_capabilities() -> dict[str, Any]:
    """Return a machine-readable catalogue of available transformations.

    The UI layer reads this to dynamically build dropdown choices and to
    decide which secondary inputs (col2, bins, strategy, etc.) to show
    for each method.

    Parameters
    ----------
    None

    Returns
    -------
    dict[str, Any]
        A dict with a single key ``"methods"`` whose value is a list of
        dicts, each containing ``method`` (internal name), ``label``
        (human-friendly name), ``requires`` (which column inputs are
        needed), and ``options`` (which extra parameters apply).
    """
    return {
        "methods": [
            {
                "method": "log",
                "label": "Log Transform",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
            },
            {
                "method": "square",
                "label": "Square Feature",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
            },
            {
                "method": "cube",
                "label": "Cube Feature",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
            },
            {
                "method": "interaction",
                "label": "Interaction (col1 x col2)",
                "requires": {"col1": True, "col2": True},
                "options": {"new_column": True},
            },
            {
                "method": "ratio",
                "label": "Ratio (col1 / col2)",
                "requires": {"col1": True, "col2": True},
                "options": {"new_column": True},
            },
            {
                "method": "binning",
                "label": "Binning",
                "requires": {"col1": True, "col2": False},
                "options": {"bins": True, "new_column": True, "labels": True},
            },
            {
                "method": "one_hot",
                "label": "One-Hot Encoding",
                "requires": {"col1": True, "col2": False},
                "options": {"prefix": True, "drop_first": True},
            },
            {
                "method": "standardize",
                "label": "Standardize (Z-score)",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
            },
            {
                "method": "normalize",
                "label": "Normalize (Min-Max)",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
            },
            {
                "method": "fillna",
                "label": "Fill Missing Values",
                "requires": {"col1": True, "col2": False},
                "options": {
                    "strategy": ["mean", "median", "mode", "constant"],
                    "fill_value": True,
                    "new_column": True,
                },
            },
            {
                "method": "dropna",
                "label": "Drop Missing Rows",
                "requires": {"col1": False, "col2": False},
                "options": {"col1_optional": True},
            },
        ]
    }
