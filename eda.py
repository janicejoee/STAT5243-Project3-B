# EDA.py

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover
    gaussian_kde = None

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
except Exception:  # pragma: no cover
    sm_lowess = None

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None


"""
Frontend integration note
-------------------------
1. For "column click + filter input" behavior, the frontend constructs a
   pandas-query-style filter string and calls the API layer
   ``filter_and_save_dataset(dataset_id, filter_expr)`` (api.py).
   That function calls ``EDA.apply_filter`` for the pure filtering step,
   then ``dataset_store.register_dataframe`` to persist the result.

   Example filter expressions:
       column = "age"
       operator = ">"
       value = 30
       frontend builds: "age > 30"

   For string values:
       column = "city"
       operator = "=="
       value = "New York"
       frontend builds: 'city == "New York"'

2. All public functions in this file return JSON-friendly Python dictionaries
   (serializable to JSON by FastAPI / Flask / similar frameworks).

3. Plotting functions return plot specifications and data payloads for frontend rendering.
   The backend here does NOT return PNGs.

4. For each 1D / 2D histogram-like output, `logx_available` / `logy_available` indicate
   whether that axis can safely be displayed on a log scale under the rule:
   "not available if there exist non-positive values".
"""


# ============================================================================
# Generic helpers
# ============================================================================

def _success(data: dict[str, Any], message: str | None = None) -> dict[str, Any]:
    """Wrap a successful payload in a consistent JSON-friendly response."""
    payload = {"status": "success", "data": data}
    if message:
        payload["message"] = message
    return payload


def _error(message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wrap an error payload in a consistent JSON-friendly response."""
    payload = {"status": "error", "message": message}
    if details:
        payload["details"] = details
    return payload


def _warning(message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wrap a warning payload in a consistent JSON-friendly response."""
    payload = {"status": "warning", "message": message}
    if data is not None:
        payload["data"] = data
    return payload


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> str | None:
    """Return None if all columns exist; otherwise return an error message."""
    missing = [col for col in columns if col and col not in df.columns]
    if missing:
        return f"Invalid column(s): {missing}"
    return None


def _is_numeric(series: pd.Series) -> bool:
    """Return True if a Series is numeric."""
    return pd.api.types.is_numeric_dtype(series)


def _is_categorical(series: pd.Series) -> bool:
    """Return True if a Series should be treated as categorical for plotting."""
    return (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def _replace_nan_with_none(obj: Any) -> Any:
    """Recursively replace NaN / inf-like values with None for JSON compatibility."""
    if isinstance(obj, dict):
        return {k: _replace_nan_with_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_nan_with_none(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_replace_nan_with_none(v) for v in obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)

    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    if pd.isna(obj):
        return None

    return obj


def _records_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-friendly records."""
    out = df.where(df.notna(), None).to_dict(orient="records")
    return _replace_nan_with_none(out)


def _json_ready(value: Any) -> Any:
    """Convert nested pandas / numpy structures into JSON-friendly Python objects."""
    if isinstance(value, pd.DataFrame):
        return _records_from_df(value)
    if isinstance(value, pd.Series):
        return _replace_nan_with_none(value.to_dict())
    return _replace_nan_with_none(value)


def _series_log_available(series: pd.Series) -> bool:
    """Return whether a series can be shown on a log scale under the positive-only rule."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.dropna()
    if numeric.empty:
        return False
    return bool((numeric > 0).all())


def _counts_log_available(counts: np.ndarray | pd.Series) -> bool:
    """Return whether histogram / count values are all strictly positive."""
    arr = np.asarray(counts, dtype=float)
    if arr.size == 0:
        return False
    return bool(np.all(arr > 0))


def _maybe_normalize_counts(counts: np.ndarray, normalize: bool) -> np.ndarray:
    """Normalize counts to unity if requested and if the total is positive."""
    counts = counts.astype(float)
    if normalize:
        total = counts.sum()
        if total > 0:
            counts = counts / total
    return counts


def _sample_df(df: pd.DataFrame, max_points: int | None) -> pd.DataFrame:
    """Return either the full DataFrame or a random sample capped at max_points."""
    if max_points is None or len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=0)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric-only version of a column with NaNs dropped."""
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _sort_count_series(series: pd.Series, sort_order: str | None) -> pd.Series:
    """Sort a count series according to the requested order."""
    if sort_order is None:
        return series
    order = sort_order.lower()
    if order == "asc":
        return series.sort_values(ascending=True)
    if order == "desc":
        return series.sort_values(ascending=False)
    return series


def _category_box_stats(series: pd.Series) -> dict[str, Any]:
    """Compute box-plot-like summary statistics for one numeric group."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
        }

    return _replace_nan_with_none(
        {
            "count": int(clean.count()),
            "mean": float(clean.mean()),
            "std": float(clean.std(ddof=1)) if clean.count() > 1 else None,
            "min": float(clean.min()),
            "q1": float(clean.quantile(0.25)),
            "median": float(clean.quantile(0.50)),
            "q3": float(clean.quantile(0.75)),
            "max": float(clean.max()),
        }
    )


def _maybe_apply_hue_default(hue: str | None, fallback: str) -> str:
    """Use fallback hue when hue is empty or None."""
    return hue if hue else fallback


# ============================================================================
# Dataframe viewing
# ============================================================================

def show_head(df: pd.DataFrame, n: int = 5) -> dict[str, Any]:
    """
    Return the first n rows of a DataFrame in JSON-friendly table form.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    n:
        Number of rows to display, matching the spirit of pandas `df.head(n)`.

    Returns
    -------
    dict
        JSON-friendly response with columns, row records, and shape metadata.
    """
    out = df.head(n)
    return _success(
        {
            "view_type": "head",
            "n_requested": int(n),
            "n_returned": int(len(out)),
            "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "columns": [str(c) for c in out.columns],
            "rows": _records_from_df(out),
        }
    )


def describe_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return DataFrame summary statistics similar to `df.describe(include='all')`.

    Parameters
    ----------
    df:
        Input pandas DataFrame.

    Returns
    -------
    dict
        JSON-friendly response with describe-table columns and rows.
    """
    desc = df.describe(include="all").transpose().reset_index()
    desc = desc.rename(columns={"index": "column"})
    return _success(
        {
            "view_type": "describe",
            "columns": [str(c) for c in desc.columns],
            "rows": _records_from_df(desc),
        }
    )


def column_types(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return DataFrame column dtypes similar to `df.dtypes()`.

    Parameters
    ----------
    df:
        Input pandas DataFrame.

    Returns
    -------
    dict
        JSON-friendly response with one row per column and its dtype.
    """
    out = pd.DataFrame(
        {
            "column": df.columns.astype(str),
            "dtype": [str(dtype) for dtype in df.dtypes],
            "is_numeric": [bool(_is_numeric(df[col])) for col in df.columns],
            "is_categorical": [bool(_is_categorical(df[col])) for col in df.columns],
        }
    )
    return _success(
        {
            "view_type": "dtypes",
            "columns": [str(c) for c in out.columns],
            "rows": _records_from_df(out),
        }
    )


# ============================================================================
# Filtering
# ============================================================================

def apply_filter(df: pd.DataFrame, filter_expr: str) -> pd.DataFrame:
    """
    Filter a DataFrame using a pandas-query-compatible expression.

    This is a **pure function**: it does not produce JSON, does not interact
    with any dataset_id or storage layer, and raises ``ValueError`` on any
    error so the caller can handle it in whatever way is appropriate for
    the context (API layer, test harness, etc.).

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    filter_expr:
        pandas ``DataFrame.query()``-compatible expression, e.g.::

            "age > 30"
            'city == "New York"'
            "salary >= 50000 and department == 'Physics'"
            "`column with spaces` > 10"

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame (a subset of rows from *df*).

    Raises
    ------
    ValueError
        If *filter_expr* is empty, references no known column, or
        ``DataFrame.query()`` raises any exception.
    """
    if not filter_expr or not filter_expr.strip():
        raise ValueError("filter_expr must be a non-empty string.")

    # Lightweight column-presence check: reject expressions that mention
    # no column of df, which almost always indicates a typo or wrong dataset.
    matched_cols = [
        col for col in df.columns
        if col in filter_expr or f"`{col}`" in filter_expr
    ]
    if not matched_cols:
        raise ValueError(
            f"No valid column names found in filter expression: {filter_expr!r}. "
            f"Available columns: {list(df.columns)}"
        )

    try:
        filtered = df.query(filter_expr, engine="python")
    except Exception as exc:
        raise ValueError(
            f"Failed to apply filter expression {filter_expr!r}: {exc}"
        ) from exc

    return filtered


# ============================================================================
# 1D plots
# ============================================================================

def plot_categorical_1d(
    df: pd.DataFrame,
    column: str,
    normalize: bool = False,
    sort_order: str | None = "desc",
) -> dict[str, Any]:
    """
    Return a categorical count-plot specification for one column.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    column:
        Categorical column to count.
    normalize:
        If True, normalize counts to unity.
    sort_order:
        "asc", "desc", or None.

    Returns
    -------
    dict
        JSON-friendly categorical plot spec for frontend rendering.
    """
    err = _validate_columns(df, [column])
    if err:
        return _error(err)

    series = df[column].fillna("<<MISSING>>").astype(str)
    counts = series.value_counts(dropna=False)
    counts = _sort_count_series(counts, sort_order)

    values = counts.to_numpy(dtype=float)
    values = _maybe_normalize_counts(values, normalize)

    payload = {
        "plot_family": "1d",
        "plot_type": "categorical_count",
        "column": column,
        "normalize": bool(normalize),
        "sort_order": sort_order,
        "categories": counts.index.tolist(),
        "counts": values.tolist(),
        "logx_available": False,
        "logy_available": _counts_log_available(values),
    }
    return _success(_json_ready(payload))


def plot_numeric_1d(
    df: pd.DataFrame,
    column: str,
    bins: int | str = 30,
    normalize: bool = False,
) -> dict[str, Any]:
    """
    Return a 1D histogram specification for one numerical column.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    column:
        Numeric column to histogram.
    bins:
        Histogram bin specification accepted by numpy / pandas, typically int or "auto".
    normalize:
        If True, normalize counts to unity.

    Returns
    -------
    dict
        JSON-friendly histogram spec with bin edges and counts.
    """
    err = _validate_columns(df, [column])
    if err:
        return _error(err)
    if not _is_numeric(df[column]):
        return _error(f"Column '{column}' is not numeric.")

    series = _numeric_series(df, column)
    if series.empty:
        return _error(f"Column '{column}' has no valid numeric values.")

    counts, edges = np.histogram(series.to_numpy(), bins=bins)
    counts = _maybe_normalize_counts(counts, normalize)

    payload = {
        "plot_family": "1d",
        "plot_type": "histogram",
        "column": column,
        "normalize": bool(normalize),
        "bins": edges.tolist(),
        "counts": counts.tolist(),
        "logx_available": _series_log_available(series),
        "logy_available": _counts_log_available(counts),
    }
    return _success(_json_ready(payload))


# ============================================================================
# Numerical x Numerical
# ============================================================================

def _numeric_numeric_hist2d(
    df: pd.DataFrame,
    x: str,
    y: str,
    bins: int | tuple[int, int] = 40,
) -> dict[str, Any]:
    """Build a 2D histogram specification for two numeric columns."""
    plot_df = df[[x, y]].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna()

    if plot_df.empty:
        return _error("No valid numeric pairs remain after dropping NaNs.")

    counts, x_edges, y_edges = np.histogram2d(
        plot_df[x].to_numpy(),
        plot_df[y].to_numpy(),
        bins=bins,
    )

    payload = {
        "plot_family": "2d",
        "plot_type": "hist2d",
        "x": x,
        "y": y,
        "x_bins": x_edges.tolist(),
        "y_bins": y_edges.tolist(),
        "counts": counts.tolist(),
        "colorbar_label": "count",
        "logx_available": _series_log_available(plot_df[x]),
        "logy_available": _series_log_available(plot_df[y]),
    }
    return _success(_json_ready(payload))


def _numeric_numeric_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    max_points: int = 5000,
) -> dict[str, Any]:
    """Build a scatter-style point payload for two numeric columns."""
    cols = [x, y] + ([hue] if hue else [])
    plot_df = df[cols].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])
    plot_df = _sample_df(plot_df, max_points)

    payload = {
        "plot_family": "2d",
        "plot_type": "scatter",
        "x": x,
        "y": y,
        "hue": hue,
        "points": _records_from_df(plot_df),
        "logx_available": _series_log_available(plot_df[x]),
        "logy_available": _series_log_available(plot_df[y]),
    }
    return _success(_json_ready(payload))


def _bin_xy(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    nbins: int,
    shared_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin x into *nbins* equal-width bins and return (midpoints, mean_y).

    If *shared_edges* is provided those edges are used instead of computing
    new ones from x_arr (enables alignment across groups).  Empty bins are
    dropped from the output.
    """
    if shared_edges is None:
        shared_edges = np.linspace(float(x_arr.min()), float(x_arr.max()), nbins + 1)
    midpoints = 0.5 * (shared_edges[:-1] + shared_edges[1:])
    bin_idx = np.clip(np.digitize(x_arr, shared_edges) - 1, 0, nbins - 1)
    mean_y = np.array(
        [y_arr[bin_idx == i].mean() if np.any(bin_idx == i) else np.nan for i in range(nbins)]
    )
    mask = ~np.isnan(mean_y)
    return midpoints[mask], mean_y[mask]


def _numeric_numeric_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    max_points: int = 5000,
    nbins: int | None = 30,
) -> dict[str, Any]:
    """
    Build a line-plot payload for two numeric columns.

    Single-line mode (``hue=None``)
        When *nbins* is set (default 30), x is divided into that many
        equal-width bins and the mean y per bin is returned, producing a
        smooth trend line.  Set ``nbins=None`` for raw sorted data.

    Multi-line mode (``hue`` provided, must be categorical)
        Returns a ``plot_type="multiline"`` response — one line per hue
        group — using the same x-binning as above for each group with
        shared bin edges.  This response shape is identical to
        ``plot_multiline`` 2D mode and is rendered the same way.
    """
    err = _validate_columns(df, [x, y] + ([hue] if hue else []))
    if err:
        return _error(err)
    if not _is_numeric(df[x]) or not _is_numeric(df[y]):
        return _error(f"Both '{x}' and '{y}' must be numeric for a line plot.")

    # ── Multi-line branch (hue splits into separate lines) ──────────────────
    if hue is not None:
        if not _is_categorical(df[hue]):
            return _error(
                f"hue column '{hue}' must be categorical. "
                "Use plot_multiline with group_by for a numeric grouping variable."
            )
        groups, _ = _multiline_groups_categorical(df, hue)
        line_specs = _multiline_2d_lines(
            groups, x, y, sort_x=True, max_points_per_line=max_points, nbins=nbins
        )
        if not line_specs:
            return _error("No valid (x, y) pairs remain for any hue group after dropping NaNs.")
        all_x = pd.to_numeric(df[x], errors="coerce").dropna()
        all_y = pd.to_numeric(df[y], errors="coerce").dropna()
        payload = {
            "plot_family": "multiline",
            "plot_type": "multiline",
            "mode": "2d",
            "x_column": x,
            "y_column": y,
            "group_source": "categorical",
            "group_column": hue,
            "n_lines": len(line_specs),
            "lines": _json_ready(line_specs),
            "logx_available": _series_log_available(all_x),
            "logy_available": _series_log_available(all_y),
            "warnings": [],
        }
        return _success(_json_ready(payload))

    # ── Single-line branch ───────────────────────────────────────────────────
    plot_df = df[[x, y]].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna()

    if plot_df.empty:
        return _error("No valid numeric pairs remain after dropping NaNs.")

    x_arr = plot_df[x].to_numpy(dtype=float)
    y_arr = plot_df[y].to_numpy(dtype=float)

    if nbins is not None and nbins > 1:
        x_out_arr, y_out_arr = _bin_xy(x_arr, y_arr, nbins)
        x_out = x_out_arr.tolist()
        y_out = y_out_arr.tolist()
        n_pts = int(len(x_out))
        aggregated = True
    else:
        plot_df = _sample_df(plot_df, max_points)
        plot_df = plot_df.sort_values(by=x).reset_index(drop=True)
        x_out = plot_df[x].tolist()
        y_out = plot_df[y].tolist()
        n_pts = len(plot_df)
        aggregated = False

    payload = {
        "plot_family": "2d",
        "plot_type": "line",
        "x": x,
        "y": y,
        "hue": None,
        "x_values": x_out,
        "y_values": y_out,
        "n_points": n_pts,
        "aggregated": aggregated,
        "nbins": nbins if aggregated else None,
        "logx_available": _series_log_available(pd.Series(x_out)),
        "logy_available": _series_log_available(pd.Series(y_out)),
    }
    return _success(_json_ready(payload))


def _numeric_numeric_joint(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    bins: int = 30,
    max_points: int = 5000,
) -> dict[str, Any]:
    """
    Build a 'jointplot-like' payload:
    scatter points + x-marginal histogram + y-marginal histogram.
    """
    cols = [x, y] + ([hue] if hue else [])
    plot_df = df[cols].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])

    if plot_df.empty:
        return _error("No valid numeric pairs remain after dropping NaNs.")

    scatter_df = _sample_df(plot_df, max_points)
    x_counts, x_edges = np.histogram(plot_df[x].to_numpy(), bins=bins)
    y_counts, y_edges = np.histogram(plot_df[y].to_numpy(), bins=bins)

    payload = {
        "plot_family": "2d",
        "plot_type": "joint",
        "x": x,
        "y": y,
        "hue": hue,
        "points": _records_from_df(scatter_df),
        "x_marginal": {
            "bins": x_edges.tolist(),
            "counts": x_counts.tolist(),
            "logx_available": _series_log_available(plot_df[x]),
            "logy_available": _counts_log_available(x_counts),
        },
        "y_marginal": {
            "bins": y_edges.tolist(),
            "counts": y_counts.tolist(),
            "logx_available": _series_log_available(plot_df[y]),
            "logy_available": _counts_log_available(y_counts),
        },
        "logx_available": _series_log_available(plot_df[x]),
        "logy_available": _series_log_available(plot_df[y]),
    }
    return _success(_json_ready(payload))


def _numeric_numeric_contour(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    gridsize: int = 60,
    max_points: int = 5000,
) -> dict[str, Any]:
    """
    Build a KDE contour payload for two numeric columns.

    If scipy is unavailable, returns an error response.
    Hue is returned for frontend grouping / legend use, but the density grid here
    is computed only for x-y jointly, not separately per hue level.
    """
    if gaussian_kde is None:
        return _error("Contour / KDE output requires scipy.stats.gaussian_kde, which is unavailable.")

    cols = [x, y] + ([hue] if hue else [])
    plot_df = df[cols].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])

    if len(plot_df) < 2:
        return _error("At least two valid points are required for contour / KDE output.")

    x_vals = plot_df[x].to_numpy(dtype=float)
    y_vals = plot_df[y].to_numpy(dtype=float)

    x_grid = np.linspace(float(np.min(x_vals)), float(np.max(x_vals)), gridsize)
    y_grid = np.linspace(float(np.min(y_vals)), float(np.max(y_vals)), gridsize)
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x_vals, y_vals])

    try:
        kde = gaussian_kde(values)
        density = np.reshape(kde(positions).T, xx.shape)
    except Exception as exc:
        return _error("Failed to compute KDE contour grid.", details={"exception": str(exc)})

    payload = {
        "plot_family": "2d",
        "plot_type": "contour",
        "x": x,
        "y": y,
        "hue": hue,
        "x_grid": x_grid.tolist(),
        "y_grid": y_grid.tolist(),
        "z_grid": density.tolist(),
        "points_preview": _records_from_df(_sample_df(plot_df, max_points)),
        "logx_available": _series_log_available(plot_df[x]),
        "logy_available": _series_log_available(plot_df[y]),
    }
    return _success(_json_ready(payload))


def plot_numeric_numeric(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    kind: str = "hist",
    bins: int | tuple[int, int] = 40,
    max_points: int = 5000,
    gridsize: int = 60,
    nbins: int | None = 30,
) -> dict[str, Any]:
    """
    Plot two numeric columns with one of: "hist", "joint", "scatter", "contour", "line".

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    x, y:
        Numeric columns.
    hue:
        Optional hue column.  For ``kind="line"`` a categorical hue splits
        the data into one line per group (returns a multiline response).
    kind:
        One of "hist", "joint", "scatter", "contour", "line".
    bins:
        Bin specification for histogram-based outputs (hist, joint).
    max_points:
        Maximum raw points returned for point-based rendering (scatter, contour).
    gridsize:
        Grid resolution for contour / KDE output.
    nbins:
        Number of x-axis bins for ``kind="line"``.  Defaults to 30, giving a
        smooth trend line.  Set ``None`` to return raw sorted data instead.

    Returns
    -------
    dict
        JSON-friendly 2D plot specification.
    """
    err = _validate_columns(df, [x, y] + ([hue] if hue else []))
    if err:
        return _error(err)
    if not _is_numeric(df[x]) or not _is_numeric(df[y]):
        return _error(f"Both '{x}' and '{y}' must be numeric columns.")

    kind = kind.lower()
    if kind == "hist":
        return _numeric_numeric_hist2d(df, x, y, bins=bins)
    if kind == "joint":
        return _numeric_numeric_joint(df, x, y, hue=hue, bins=int(bins) if not isinstance(bins, tuple) else 40, max_points=max_points)
    if kind == "scatter":
        return _numeric_numeric_scatter(df, x, y, hue=hue, max_points=max_points)
    if kind == "contour":
        return _numeric_numeric_contour(df, x, y, hue=hue, gridsize=gridsize, max_points=max_points)
    if kind == "line":
        return _numeric_numeric_line(df, x, y, hue=hue, max_points=max_points, nbins=nbins)

    return _error("Invalid kind for numeric-numeric plot.", details={"allowed": ["hist", "joint", "scatter", "contour", "line"]})


# ============================================================================
# Numeric x Categorical
# ============================================================================

def plot_numeric_categorical(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    kind: str = "box",
    max_points: int = 5000,
) -> dict[str, Any]:
    """
    Plot a numeric column against a categorical column with one of:
    "bar", "violin", "box", "swarm", "swarmonbox".

    Convention:
    - x is numeric
    - y is categorical

    If hue is not provided, y is also used as hue for frontend color grouping.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    x:
        Numeric column.
    y:
        Categorical column.
    hue:
        Optional hue column. If absent, defaults to y.
    kind:
        One of "bar", "violin", "box", "swarm", "swarmonbox".
    max_points:
        Maximum number of raw points returned for point-based styles.

    Returns
    -------
    dict
        JSON-friendly grouped plot specification.
    """
    err = _validate_columns(df, [x, y] + ([hue] if hue else []))
    if err:
        return _error(err)
    if not _is_numeric(df[x]):
        return _error(f"Column '{x}' must be numeric.")
    if not _is_categorical(df[y]):
        return _error(f"Column '{y}' must be categorical.")

    effective_hue = _maybe_apply_hue_default(hue, y)
    # When hue resolves to the same column as y, avoid selecting / grouping on it twice.
    hue_same_as_y = (effective_hue == y)
    select_cols = [x, y] + ([] if hue_same_as_y else ([effective_hue] if effective_hue else []))
    plot_df = df[select_cols].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])

    if plot_df.empty:
        return _error("No valid rows remain after dropping NaNs in required columns.")

    kind = kind.lower()
    # Use a single key when hue and y are the same column to avoid the pandas
    # "not 1-dimensional" error that occurs when groupby receives duplicate keys.
    group_keys = [y] if (hue_same_as_y or not effective_hue) else [y, effective_hue]
    grouped = plot_df.groupby(group_keys, dropna=False)

    summary_rows = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        group_y = key[0]
        group_hue = group_y if hue_same_as_y else (key[1] if len(key) > 1 else None)

        stats = _category_box_stats(group[x])
        stats.update({y: group_y, "hue_value": group_hue})
        summary_rows.append(stats)

    payload: dict[str, Any] = {
        "plot_family": "2d",
        "plot_type": kind,
        "x": x,
        "y": y,
        "hue": effective_hue,
        "summary": _json_ready(summary_rows),
        "logx_available": _series_log_available(plot_df[x]),
        "logy_available": False,
    }

    if kind == "bar":
        bar_rows = (
            plot_df.groupby(group_keys, dropna=False)[x]
            .agg(["mean", "count", "std"])
            .reset_index()
            .rename(columns={"mean": "value"})
        )
        payload["bars"] = _records_from_df(bar_rows)
        return _success(_json_ready(payload))

    if kind in {"violin", "box", "swarm", "swarmonbox"}:
        payload["points"] = _records_from_df(_sample_df(plot_df, max_points))
        return _success(_json_ready(payload))

    return _error(
        "Invalid kind for numeric-categorical plot.",
        details={"allowed": ["bar", "violin", "box", "swarm", "swarmonbox"]},
    )


# ============================================================================
# Categorical x Categorical
# ============================================================================

def plot_categorical_categorical(
    df: pd.DataFrame,
    x: str,
    y: str,
) -> dict[str, Any]:
    """
    Plot two categorical columns against each other as a heatmap-like contingency table.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    x, y:
        Categorical columns.

    Returns
    -------
    dict
        JSON-friendly heatmap specification using a contingency table.
    """
    err = _validate_columns(df, [x, y])
    if err:
        return _error(err)
    if not _is_categorical(df[x]) or not _is_categorical(df[y]):
        return _error(f"Both '{x}' and '{y}' must be categorical columns.")

    x_series = df[x].fillna("<<MISSING>>").astype(str)
    y_series = df[y].fillna("<<MISSING>>").astype(str)
    table = pd.crosstab(index=y_series, columns=x_series, dropna=False)

    payload = {
        "plot_family": "2d",
        "plot_type": "heatmap",
        "x": x,
        "y": y,
        "x_categories": [str(c) for c in table.columns],
        "y_categories": [str(i) for i in table.index],
        "values": table.to_numpy(dtype=int).tolist(),
        "logx_available": False,
        "logy_available": False,
    }
    return _success(_json_ready(payload))


# ============================================================================
# Generic 2-column dispatcher
# ============================================================================

def plot_two_columns(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    kind: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Dispatch to the appropriate 2-column plotting function based on column types.

    Rules
    -----
    - If both x and y are numeric:
        call `plot_numeric_numeric(...)`
    - If one is numeric and one is categorical:
        call `plot_numeric_categorical(...)`
        If x is categorical and y is numeric, swap them internally so the numeric
        variable remains on x and the categorical variable remains on y.
    - If both are categorical:
        ignore hue and call `plot_categorical_categorical(...)`

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    x, y:
        Column names.
    hue:
        Optional hue column.
    kind:
        Plot-style option passed through to the chosen function.
    **kwargs:
        Additional keyword arguments passed through to the chosen function.

    Returns
    -------
    dict
        JSON-friendly plotting response from the selected specialized function.
    """
    err = _validate_columns(df, [x, y] + ([hue] if hue else []))
    if err:
        return _error(err)

    x_is_num = _is_numeric(df[x])
    y_is_num = _is_numeric(df[y])
    x_is_cat = _is_categorical(df[x])
    y_is_cat = _is_categorical(df[y])

    if x_is_num and y_is_num:
        return plot_numeric_numeric(df, x=x, y=y, hue=hue, kind=kind or "hist", **kwargs)

    if x_is_num and y_is_cat:
        return plot_numeric_categorical(df, x=x, y=y, hue=hue, kind=kind or "box", **kwargs)

    if x_is_cat and y_is_num:
        # Switch so numeric stays on x and categorical stays on y.
        return plot_numeric_categorical(df, x=y, y=x, hue=hue, kind=kind or "box", **kwargs)

    if x_is_cat and y_is_cat:
        return plot_categorical_categorical(df, x=x, y=y)

    return _error(
        "Could not determine valid plot dispatch from the input column types.",
        details={
            "x": x,
            "y": y,
            "x_dtype": str(df[x].dtype),
            "y_dtype": str(df[y].dtype),
        },
    )


# ============================================================================
# Regression
# ============================================================================

def regression_analysis(
    df: pd.DataFrame,
    x: str,
    y: str,
    order: int = 1,
    logx: bool = False,
    robust: bool = False,
    lowess: bool = False,
    max_points: int = 5000,
    fit_points: int = 200,
) -> dict[str, Any]:
    """
    Compute regression-oriented output for two numeric columns.

    This is intended to back a frontend option similar in spirit to seaborn.regplot:
    - polynomial fit via `order`
    - optional log-x fit
    - optional robust linear fit
    - optional LOWESS smoothing

    Notes
    -----
    - `lowess=True` requires statsmodels.
    - `robust=True` for linear fits requires statsmodels.
    - `logx=True` requires x > 0 for all used fit points.
    - Pearson correlation is returned as part of the JSON output.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    x, y:
        Numeric columns.
    order:
        Polynomial order for the fit. Ignored by lowess. For robust fits, only linear
        order=1 is supported here.
    logx:
        If True, fit against log(x) while still returning original x values for plotting.
    robust:
        If True, attempt a robust linear fit.
    lowess:
        If True, attempt LOWESS smoothing.
    max_points:
        Maximum number of raw points returned for scatter rendering.
    fit_points:
        Number of x-grid points used for returned fitted curves.

    Returns
    -------
    dict
        JSON-friendly regression output including point preview, fit curve, and
        Pearson correlation.
    """
    err = _validate_columns(df, [x, y])
    if err:
        return _error(err)
    if not _is_numeric(df[x]) or not _is_numeric(df[y]):
        return _error(
            "Regression requires both columns to be numeric.",
            details={"x_dtype": str(df[x].dtype), "y_dtype": str(df[y].dtype)},
        )

    plot_df = df[[x, y]].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna()

    if plot_df.empty:
        return _error("No valid numeric pairs remain after dropping NaNs.")

    if logx and not _series_log_available(plot_df[x]):
        return _error(f"logx=True is invalid because column '{x}' contains non-positive values.")

    x_vals = plot_df[x].to_numpy(dtype=float)
    y_vals = plot_df[y].to_numpy(dtype=float)

    x_fit = np.log(x_vals) if logx else x_vals
    x_grid_original = np.linspace(float(np.min(x_vals)), float(np.max(x_vals)), fit_points)
    x_grid_fit = np.log(x_grid_original) if logx else x_grid_original

    pearson = float(plot_df[[x, y]].corr(method="pearson").iloc[0, 1])

    fit_payload: dict[str, Any] = {
        "fit_type": None,
        "x_fit": None,
        "y_fit": None,
        "coefficients": None,
        "warnings": [],
    }

    try:
        if lowess:
            if sm_lowess is None:
                return _error("LOWESS requested but statsmodels is unavailable.")
            fitted = sm_lowess(endog=y_vals, exog=x_fit, frac=0.66, return_sorted=True)
            x_fit_sorted = fitted[:, 0]
            y_fit_sorted = fitted[:, 1]

            if logx:
                x_fit_sorted = np.exp(x_fit_sorted)

            fit_payload.update(
                {
                    "fit_type": "lowess",
                    "x_fit": x_fit_sorted.tolist(),
                    "y_fit": y_fit_sorted.tolist(),
                }
            )

        elif robust:
            if order != 1:
                return _error("Robust fitting is only implemented here for order=1.")
            if sm is None:
                return _error("Robust fit requested but statsmodels is unavailable.")

            X = sm.add_constant(x_fit)
            model = sm.RLM(y_vals, X)
            result = model.fit()

            y_grid = result.predict(sm.add_constant(x_grid_fit))
            fit_payload.update(
                {
                    "fit_type": "robust_linear",
                    "x_fit": x_grid_original.tolist(),
                    "y_fit": y_grid.tolist(),
                    "coefficients": result.params.tolist(),
                }
            )

        else:
            coeffs = np.polyfit(x_fit, y_vals, deg=order)
            y_grid = np.polyval(coeffs, x_grid_fit)
            fit_payload.update(
                {
                    "fit_type": "polynomial",
                    "x_fit": x_grid_original.tolist(),
                    "y_fit": y_grid.tolist(),
                    "coefficients": coeffs.tolist(),
                }
            )

    except Exception as exc:
        return _error("Regression fit failed.", details={"exception": str(exc)})

    payload = {
        "plot_family": "2d",
        "plot_type": "regression",
        "x": x,
        "y": y,
        "order": int(order),
        "logx": bool(logx),
        "robust": bool(robust),
        "lowess": bool(lowess),
        "pearson_correlation": pearson,
        "points": _records_from_df(_sample_df(plot_df, max_points)),
        "fit": fit_payload,
        "logx_available": _series_log_available(plot_df[x]),
        "logy_available": _series_log_available(plot_df[y]),
    }
    return _success(_json_ready(payload))


# ============================================================================
# Multi-line plotting helpers
# ============================================================================

def _multiline_groups_categorical(
    df: pd.DataFrame,
    group_by: str,
) -> tuple[list[tuple[str, pd.DataFrame]], list[str]]:
    """Split df into one sub-DataFrame per unique value of *group_by*."""
    groups: list[tuple[str, pd.DataFrame]] = []
    for label, sub in df.groupby(group_by, dropna=False):
        label_str = "<<MISSING>>" if pd.isna(label) else str(label)
        groups.append((label_str, sub.copy()))
    return groups, []


def _multiline_groups_filters(
    df: pd.DataFrame,
    filter_strings: list[str],
    filter_labels: list[str],
) -> tuple[list[tuple[str, pd.DataFrame]], list[str]]:
    """
    Apply each filter string to df and collect (label, sub_df) pairs.

    Returns (valid_groups, warnings).  Warnings are accumulated for:
    - empty or column-free filter strings
    - syntax errors from df.query()
    - filters that select zero rows (valid but empty)
    Valid groups are returned even if some filters failed.
    """
    valid_groups: list[tuple[str, pd.DataFrame]] = []
    warnings: list[str] = []

    for label, fstr in zip(filter_labels, filter_strings):
        if not fstr or not fstr.strip():
            warnings.append(f"['{label}'] Filter expression is empty — skipped.")
            continue

        # Lightweight column-presence check (same logic as apply_filter).
        matched_cols = [col for col in df.columns if col in fstr or f"`{col}`" in fstr]
        if not matched_cols:
            warnings.append(
                f"['{label}'] No valid column names found in filter '{fstr}' — skipped."
            )
            continue

        try:
            filtered = df.query(fstr, engine="python")
        except Exception as exc:
            warnings.append(
                f"['{label}'] Filter '{fstr}' raised an error: {exc} — skipped."
            )
            continue

        if filtered.empty:
            warnings.append(
                f"['{label}'] Filter '{fstr}' produced an empty selection — skipped."
            )
            continue

        valid_groups.append((label, filtered.copy()))

    return valid_groups, warnings


def _multiline_1d_lines(
    groups: list[tuple[str, pd.DataFrame]],
    column: str,
    normalize: bool,
    nbins: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """
    Compute histogram-line specs for each group using shared bin edges.

    Sharing bin edges across groups ensures the x-axes are aligned for
    direct visual comparison.  Returns (line_specs, shared_edges).
    """
    all_vals = pd.concat(
        [pd.to_numeric(sub[column], errors="coerce").dropna() for _, sub in groups],
        ignore_index=True,
    )
    _, shared_edges = np.histogram(all_vals.to_numpy(), bins=nbins)
    midpoints = 0.5 * (shared_edges[:-1] + shared_edges[1:])

    line_specs: list[dict[str, Any]] = []
    for label, sub_df in groups:
        vals = pd.to_numeric(sub_df[column], errors="coerce").dropna()
        counts, _ = np.histogram(vals.to_numpy(), bins=shared_edges)
        counts = _maybe_normalize_counts(counts.astype(float), normalize)
        line_specs.append(
            {
                "label": label,
                "x": midpoints.tolist(),
                "y": counts.tolist(),
                "n_points": int(len(vals)),
            }
        )
    return line_specs, shared_edges


def _multiline_2d_lines(
    groups: list[tuple[str, pd.DataFrame]],
    x_column: str,
    y_column: str,
    sort_x: bool,
    max_points_per_line: int | None,
    nbins: int | None = 30,
) -> list[dict[str, Any]]:
    """
    Compute line specs (x, y) for each group.

    When *nbins* is set (default 30), x values are binned into that many
    equal-width bins and y is aggregated as the within-bin mean, using
    **shared edges** computed from all groups combined.  This produces smooth,
    directly comparable trend lines even when the raw data are noisy.

    When *nbins* is ``None``, each group's raw data are returned (after
    optional sampling and x-sorting).
    """
    line_specs: list[dict[str, Any]] = []

    if nbins is not None and nbins > 1:
        # Shared bin edges from the combined x range of all groups.
        all_x = pd.concat(
            [pd.to_numeric(sub[x_column], errors="coerce").dropna() for _, sub in groups],
            ignore_index=True,
        )
        if all_x.empty:
            return line_specs
        shared_edges = np.linspace(float(all_x.min()), float(all_x.max()), nbins + 1)

        for label, sub_df in groups:
            pair_df = sub_df[[x_column, y_column]].copy()
            pair_df[x_column] = pd.to_numeric(pair_df[x_column], errors="coerce")
            pair_df[y_column] = pd.to_numeric(pair_df[y_column], errors="coerce")
            pair_df = pair_df.dropna()
            if pair_df.empty:
                continue
            x_mid, y_mean = _bin_xy(
                pair_df[x_column].to_numpy(dtype=float),
                pair_df[y_column].to_numpy(dtype=float),
                nbins,
                shared_edges=shared_edges,
            )
            line_specs.append(
                {
                    "label": label,
                    "x": x_mid.tolist(),
                    "y": y_mean.tolist(),
                    "n_points": int(len(pair_df)),
                }
            )
    else:
        for label, sub_df in groups:
            pair_df = sub_df[[x_column, y_column]].copy()
            pair_df[x_column] = pd.to_numeric(pair_df[x_column], errors="coerce")
            pair_df[y_column] = pd.to_numeric(pair_df[y_column], errors="coerce")
            pair_df = pair_df.dropna()
            if pair_df.empty:
                continue
            pair_df = _sample_df(pair_df, max_points_per_line)
            if sort_x:
                pair_df = pair_df.sort_values(by=x_column)
            line_specs.append(
                {
                    "label": label,
                    "x": pair_df[x_column].tolist(),
                    "y": pair_df[y_column].tolist(),
                    "n_points": int(len(pair_df)),
                }
            )
    return line_specs


# ============================================================================
# plot_multiline — public API
# ============================================================================

def plot_multiline(
    df: pd.DataFrame,
    column: str,
    x_column: str | None = None,
    *,
    group_by: str | None = None,
    filter_strings: list[str] | None = None,
    filter_labels: list[str] | None = None,
    normalize: bool = False,
    nbins: int | None = 30,
    sort_x: bool = True,
    max_points_per_line: int | None = None,
) -> dict[str, Any]:
    """
    Plot multiple lines on a shared canvas — one line per group.

    Groups are defined by exactly **one** of:

    - ``group_by``: a categorical column; one line per unique value.
    - ``filter_strings``: a list of pandas-query expressions; one line per
      valid, non-empty filter.  Filters referencing absent columns, causing
      syntax errors, or yielding empty selections are *skipped* — each
      generates a warning entry in ``data["warnings"]``.  If at least one
      valid group remains the function returns ``status="warning"``; if
      all groups fail it returns ``status="error"``.

    Each line represents either:

    - **2D mode** (``x_column`` provided): *column* (y-axis) vs *x_column*
      (x-axis).  When *nbins* is set (default 30), x is binned into that many
      equal-width bins (shared across groups) and y is aggregated as the
      within-bin mean, producing smooth comparable trend lines.
    - **1D mode** (``x_column`` is ``None``): histogram / count distribution
      of *column* plotted as a line.  All groups share the same bin edges.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    column:
        2D mode — numeric y-axis column.
        1D mode — numeric column whose distribution is plotted.
    x_column:
        Numeric x-axis column.  ``None`` selects 1D mode.
    group_by:
        Categorical column to split *df* into groups.
    filter_strings:
        List of pandas-query-compatible filter expressions.
    filter_labels:
        Display labels for each filter; auto-generated if absent or shorter
        than ``filter_strings``.
    normalize:
        (1D only) Normalize each group's histogram counts to sum to 1.
    nbins:
        Number of bins.  1D: shared histogram bins.  2D: x-axis bins used
        for mean-aggregation (set ``None`` for raw sorted data).
    sort_x:
        (2D only, only when ``nbins`` is ``None``) Sort raw data by x.
    max_points_per_line:
        (2D only, only when ``nbins`` is ``None``) Downsample each group.

    Returns
    -------
    dict
        JSON-friendly payload with ``"lines"`` (one spec per group),
        ``"mode"`` (``"1d"`` or ``"2d"``), and ``"warnings"`` (skipped
        filter reasons).  ``status`` is ``"warning"`` when some filters
        were skipped, ``"success"`` otherwise.
    """
    # ── Mutual exclusivity ──────────────────────────────────────────────────
    if group_by is None and filter_strings is None:
        return _error(
            "Provide exactly one of 'group_by' or 'filter_strings'.",
            details={"hint": "group_by='col' for categorical split, or filter_strings=['expr1', …] for filter-based split."},
        )
    if group_by is not None and filter_strings is not None:
        return _error("Provide exactly one of 'group_by' or 'filter_strings', not both.")

    # ── Column existence ─────────────────────────────────────────────────────
    cols_to_check = [column] + ([x_column] if x_column else []) + ([group_by] if group_by else [])
    err = _validate_columns(df, [c for c in cols_to_check if c])
    if err:
        return _error(err)

    # ── Type checks ──────────────────────────────────────────────────────────
    if x_column is not None:
        if not _is_numeric(df[x_column]):
            return _error(f"x_column '{x_column}' must be numeric for 2D mode.")
        if not _is_numeric(df[column]):
            return _error(f"column '{column}' must be numeric for 2D mode.")
    else:
        if not _is_numeric(df[column]):
            return _error(
                f"column '{column}' must be numeric for 1D histogram mode. "
                "For categorical data use plot_categorical_1d per group instead."
            )

    # ── Build groups ─────────────────────────────────────────────────────────
    accumulated_warnings: list[str] = []

    if group_by is not None:
        if not _is_categorical(df[group_by]):
            return _error(
                f"group_by column '{group_by}' is not categorical.",
                details={"dtype": str(df[group_by].dtype)},
            )
        groups, accumulated_warnings = _multiline_groups_categorical(df, group_by)
        group_source = "categorical"
    else:
        if not isinstance(filter_strings, list) or len(filter_strings) == 0:
            return _error("filter_strings must be a non-empty list of filter expressions.")

        n = len(filter_strings)
        if filter_labels is None:
            resolved_labels = [f"Group {i + 1}" for i in range(n)]
        else:
            resolved_labels = list(filter_labels)
            if len(resolved_labels) < n:
                resolved_labels += [f"Group {i + 1}" for i in range(len(resolved_labels), n)]
            else:
                resolved_labels = resolved_labels[:n]

        groups, accumulated_warnings = _multiline_groups_filters(
            df, filter_strings, resolved_labels
        )
        group_source = "filter"

    if not groups:
        return _error(
            "No valid groups could be constructed.",
            details={"warnings": accumulated_warnings},
        )

    # ── Compute line specs ───────────────────────────────────────────────────
    if x_column is None:
        line_specs, shared_edges = _multiline_1d_lines(groups, column, normalize, nbins if nbins is not None else 30)
        all_numeric = pd.to_numeric(df[column], errors="coerce").dropna()
        data: dict[str, Any] = {
            "plot_family": "multiline",
            "plot_type": "multiline",
            "mode": "1d",
            "column": column,
            "normalize": bool(normalize),
            "bins": shared_edges.tolist(),
            "group_source": group_source,
            "group_column": group_by,
            "n_lines": len(line_specs),
            "lines": _json_ready(line_specs),
            "logx_available": _series_log_available(all_numeric),
            "logy_available": False,
            "warnings": accumulated_warnings,
        }
    else:
        line_specs_2d = _multiline_2d_lines(
            groups, x_column, column, sort_x, max_points_per_line, nbins=nbins
        )
        if not line_specs_2d:
            return _error(
                "No valid (x, y) pairs remain for any group after dropping NaNs.",
                details={"warnings": accumulated_warnings},
            )
        all_x = pd.to_numeric(df[x_column], errors="coerce").dropna()
        all_y = pd.to_numeric(df[column], errors="coerce").dropna()
        data = {
            "plot_family": "multiline",
            "plot_type": "multiline",
            "mode": "2d",
            "x_column": x_column,
            "y_column": column,
            "group_source": group_source,
            "group_column": group_by,
            "n_lines": len(line_specs_2d),
            "lines": _json_ready(line_specs_2d),
            "logx_available": _series_log_available(all_x),
            "logy_available": _series_log_available(all_y),
            "warnings": accumulated_warnings,
        }

    if accumulated_warnings:
        return {
            "status": "warning",
            "message": (
                f"{len(accumulated_warnings)} group(s) skipped; "
                "see data['warnings'] for details."
            ),
            "data": _json_ready(data),
        }
    return _success(_json_ready(data))


# ============================================================================
# correlation_matrix — public API
# ============================================================================

def correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
) -> dict[str, Any]:
    """
    Compute the correlation matrix for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str] | None
        Specific columns to include.  If ``None``, all numeric columns are used.
    method : str
        Correlation method: ``"pearson"``, ``"spearman"``, or ``"kendall"``.

    Returns
    -------
    dict
        JSON-friendly payload with ``plot_type="correlation_matrix"``,
        ``columns``, ``values`` (nested list), and ``method``.
    """
    if method not in {"pearson", "spearman", "kendall"}:
        return _error(f"Unsupported method '{method}'. Use pearson, spearman, or kendall.")

    if columns is not None:
        err = _validate_columns(df, columns)
        if err:
            return _error(err)
        numeric_cols = [c for c in columns if _is_numeric(df[c])]
    else:
        numeric_cols = [c for c in df.columns if _is_numeric(df[c])]

    if len(numeric_cols) < 2:
        return _error("Need at least 2 numeric columns for a correlation matrix.")

    corr_df = df[numeric_cols].corr(method=method)
    values = corr_df.values.tolist()
    # Replace any NaN from constant columns
    values = [[None if (v != v) else round(v, 4) for v in row] for row in values]

    return _success({
        "plot_type": "correlation_matrix",
        "columns": list(numeric_cols),
        "values": values,
        "method": method,
    })