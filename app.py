from __future__ import annotations

"""
STAT 5243 Project 2 — Interactive Data Workbench (Shiny for Python).
"""

from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any
import re

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris as sklearn_load_iris
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly
import shinyswatch

import eda as EDA
import feature_engineering
import data_cleaning as cleaning


# ---------------------------------------------------------------------------
# Plotly global template
# ---------------------------------------------------------------------------
_app_template = go.layout.Template(
    layout=go.Layout(
        font=dict(family="'Nunito Sans', system-ui, sans-serif", color="#1e293b", size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[
            "#1a1a2e", "#4361ee", "#7209b7", "#06d6a0", "#f77f00",
            "#d62828", "#0891b2", "#16a34a", "#e11d48", "#ca8a04",
        ],
        title=dict(font=dict(size=16, color="#1a1a2e")),
        xaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", zeroline=False),
        yaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", zeroline=False),
        margin=dict(t=50, r=16, b=40, l=50),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, bgcolor="rgba(255,255,255,0.8)",
        ),
    )
)
pio.templates["app_theme"] = _app_template
pio.templates.default = "app_theme"

# ---------------------------------------------------------------------------
# Google Analytics 4 Configuration
# ---------------------------------------------------------------------------
GA_MEASUREMENT_ID = "G-D8T3NJKCNN"
AB_VERSION = "B"

ga_head_tags = ui.TagList(
    ui.tags.script(
        src=f"https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}",
        async_="",
    ),
    ui.tags.script(
        f"""
        window.dataLayer = window.dataLayer || [];
        function gtag(){{ dataLayer.push(arguments); }}
        gtag('js', new Date());
        gtag('config', '{GA_MEASUREMENT_ID}');
        gtag('set', 'user_properties', {{
            ab_version: '{AB_VERSION}'
        }});
        """
    ),
    ui.tags.script(
        """
        $(document).on('shiny:inputchanged', function(event) {
            if (event.name === 'main_nav') {
                gtag('event', 'tab_switch', { 'tab_name': event.value });
            }
        });
        $(document).on('click', '.btn-dark, .btn-outline-dark, .btn-outline-primary', function() {
            gtag('event', 'button_click', { 'button_label': $(this).text().trim() });
        });
        var _tabEnterTime = Date.now(), _currentTab = '';
        $(document).on('shiny:inputchanged', function(event) {
            if (event.name === 'main_nav') {
                var now = Date.now();
                if (_currentTab) {
                    gtag('event', 'tab_duration', {
                        'tab_name': _currentTab,
                        'duration_seconds': Math.round((now - _tabEnterTime) / 1000)
                    });
                }
                _currentTab = event.value;
                _tabEnterTime = now;
            }
        });
        window._sessionStart = Date.now();
        window.addEventListener('beforeunload', function() {
            if (_currentTab) {
                gtag('event', 'tab_duration', {
                    'tab_name': _currentTab,
                    'duration_seconds': Math.round((Date.now() - _tabEnterTime) / 1000)
                });
            }
            gtag('event', 'session_duration', {
                'total_seconds': Math.round((Date.now() - window._sessionStart) / 1000)
            });
        });
        """
    ),
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEST_DATA_PATH = BASE_DIR / "test_data" / "sleep_mobile_stress_dataset_15000.csv"


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------
def load_builtin_dataset(name: str) -> pd.DataFrame:
    if name == "sleep_health":
        return pd.read_csv(TEST_DATA_PATH)
    if name == "iris":
        return sklearn_load_iris(as_frame=True).frame
    if name == "tips":
        import seaborn as sns
        return sns.load_dataset("tips")
    raise ValueError(f"Unknown built-in dataset: {name}")


def load_uploaded_dataset(path: str, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return cleaning.load_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return cleaning.load_excel(path)
    if suffix == ".json":
        return cleaning.load_json(path)
    if suffix == ".rds":
        return cleaning.load_rds(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def next_dataset_key(datasets: OrderedDict[str, dict[str, Any]], prefix: str) -> str:
    if prefix == "original" and "original" not in datasets:
        return "original"
    index = 1
    while True:
        key = f"{prefix}_{index:02d}"
        if key not in datasets:
            return key
        index += 1


def register_dataset_version(
    datasets: OrderedDict[str, dict[str, Any]],
    df: pd.DataFrame,
    *,
    prefix: str,
    label: str,
    source_key: str | None = None,
    transform: str | None = None,
    apply_kind: str | None = None,
) -> tuple[OrderedDict[str, dict[str, Any]], str]:
    key = next_dataset_key(datasets, prefix)
    new_datasets = OrderedDict(datasets)
    parent = new_datasets.get(source_key) if source_key else None
    cleaning_applied = bool(parent.get("cleaning_applied")) if parent else False
    feature_engineering_applied = bool(parent.get("feature_engineering_applied")) if parent else False
    if apply_kind == "cleaning":
        cleaning_applied = True
    elif apply_kind == "feature":
        feature_engineering_applied = True
    new_datasets[key] = {
        "label": label,
        "df": df.copy(),
        "source_key": source_key,
        "transform": transform,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cleaning_applied": cleaning_applied,
        "feature_engineering_applied": feature_engineering_applied,
    }
    return new_datasets, key


def overwrite_dataset_version(
    datasets: OrderedDict[str, dict[str, Any]],
    key: str,
    df: pd.DataFrame,
    *,
    transform: str | None = None,
    apply_kind: str | None = None,
) -> OrderedDict[str, dict[str, Any]]:
    new_datasets = OrderedDict(datasets)
    record = dict(new_datasets[key])
    record["df"] = df.copy()
    # Append so stage inference (keywords in `transform`) still sees prior cleaning / steps
    # when the user applies multiple "current version" operations on the same key.
    prev = (record.get("transform") or "").strip()
    new_t = (transform or "").strip()
    if prev and new_t:
        record["transform"] = f"{prev} | {new_t}"
    else:
        record["transform"] = new_t or prev
    cleaning_applied = bool(record.get("cleaning_applied"))
    feature_engineering_applied = bool(record.get("feature_engineering_applied"))
    if apply_kind == "cleaning":
        cleaning_applied = True
    elif apply_kind == "feature":
        feature_engineering_applied = True
    record["cleaning_applied"] = cleaning_applied
    record["feature_engineering_applied"] = feature_engineering_applied
    record["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_datasets[key] = record
    return new_datasets


def format_history_table(datasets: OrderedDict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, record in datasets.items():
        df = record["df"]
        rows.append({
            "key": key,
            "label": record["label"],
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "source": record["source_key"] or "-",
            "transform": record["transform"] or "-",
            "created_at": record["created_at"],
        })
    return pd.DataFrame(rows)


def _text_has_any_keyword(text: str | None, keywords: tuple[str, ...]) -> bool:
    if not text:
        return False
    lowered = str(text).lower()
    return any(keyword in lowered for keyword in keywords)


def infer_stage_flags_for_record(
    key: str,
    record: dict[str, Any] | None,
) -> dict[str, bool]:
    if record is None:
        return {"loaded": False, "cleaned": False, "feature_engineered": False}

    transform = str(record.get("transform") or "")
    key_l = key.lower()
    source_key = str(record.get("source_key") or "").lower()

    is_loaded = True
    explicit_clean = bool(record.get("cleaning_applied"))
    explicit_feature = bool(record.get("feature_engineering_applied"))
    is_cleaned = (
        explicit_clean
        or key_l.startswith("cleaned_")
        or _text_has_any_keyword(transform, (
            "handle_missing",
            "knn_impute",
            "remove_duplicate",
            "duplicate rows",  # remove_duplicates summary: "Found N duplicate rows ..."
            "scale_",
            "encode_",
            "outlier",
            "standardize_text",
            "coerce_column_types",
            "clean",
        ))
        or source_key.startswith("cleaned_")
    )
    is_feature_engineered = (
        explicit_feature
        or key_l.startswith("feature_")
        or _text_has_any_keyword(transform, (
            "feature",
            "interaction",
            "ratio",
            "bin",
            "polynomial",
            "log transform",
            "square transform",
            "cube transform",
            "custom",
        ))
        or source_key.startswith("feature_")
    )

    return {
        "loaded": is_loaded,
        "cleaned": is_cleaned,
        "feature_engineered": is_feature_engineered,
    }


def derive_active_stage_state(
    datasets: OrderedDict[str, dict[str, Any]],
    active_key: str | None,
) -> dict[str, Any]:
    base_state: dict[str, Any] = {
        "loaded": False,
        "cleaned": False,
        "feature_engineered": False,
        "lineage_keys": [],
    }
    if not active_key or active_key not in datasets:
        return base_state

    visited: set[str] = set()
    lineage_keys: list[str] = []
    stage_flags = {
        "loaded": False,
        "cleaned": False,
        "feature_engineered": False,
    }

    cursor_key: str | None = active_key
    while cursor_key and cursor_key in datasets and cursor_key not in visited:
        visited.add(cursor_key)
        lineage_keys.append(cursor_key)
        record = datasets.get(cursor_key)
        inferred = infer_stage_flags_for_record(cursor_key, record)
        stage_flags["loaded"] = stage_flags["loaded"] or inferred["loaded"]
        stage_flags["cleaned"] = stage_flags["cleaned"] or inferred["cleaned"]
        stage_flags["feature_engineered"] = (
            stage_flags["feature_engineered"] or inferred["feature_engineered"]
        )
        cursor_key = None if record is None else record.get("source_key")

    return {
        **stage_flags,
        "lineage_keys": lineage_keys,
    }


def coerce_text_value(value: str | None) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def dataframe_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    data = payload.get("data", {})
    columns = data.get("columns", [])
    rows = data.get("rows", [])
    if not rows and not columns:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=columns or None)


def current_overview(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None:
        return None
    return cleaning.get_overview(df)


def current_column_types(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["column", "dtype", "is_numeric", "is_categorical"])
    payload = EDA.column_types(df)
    return pd.DataFrame(payload["data"]["rows"])


def midpoints(edges: list[float]) -> list[float]:
    return [(float(edges[i]) + float(edges[i + 1])) / 2 for i in range(len(edges) - 1)]


def widths(edges: list[float]) -> list[float]:
    return [float(edges[i + 1]) - float(edges[i]) for i in range(len(edges) - 1)]


# ---------------------------------------------------------------------------
# Descriptive key generation
# ---------------------------------------------------------------------------
def _sanitize_col(col: str, max_len: int = 8) -> str:
    """Shorten a column name to a filesystem-safe token."""
    s = re.sub(r"[^a-zA-Z0-9]", "_", str(col))
    return s[:max_len]


def _cols_token(cols: list[str] | str | None, max_cols: int = 2) -> str:
    if not cols:
        return ""
    if isinstance(cols, str):
        cols = [cols]
    return "_".join(_sanitize_col(c) for c in list(cols)[:max_cols])


def _filter_expr_to_slug(expr: str) -> str:
    """Convert a filter expression to a compact slug for dataset naming.

    Examples
    --------
    ``age >= 5``               → ``age_geq_5``
    ``type == "race"``         → ``type_eq_race``
    ``(age >= 5) & (x != 0)`` → ``age_geq_5_x_neq_0``
    """
    s = expr
    # Replace multi-char operators before single-char ones
    for op, abbr in [(">=", "_geq_"), ("<=", "_leq_"), ("!=", "_neq_"),
                     ("==", "_eq_"), (">", "_gt_"), ("<", "_lt_")]:
        s = s.replace(op, abbr)
    # Strip quotes and parentheses
    s = re.sub(r'["\']', '', s)
    s = re.sub(r'[()&|~]', '_', s)
    # Collapse whitespace → underscore
    s = re.sub(r'\s+', '_', s.strip())
    # Remove any characters that aren't alphanumeric or underscore
    s = re.sub(r'[^a-zA-Z0-9_]', '', s)
    # Collapse repeated underscores and strip leading/trailing ones
    s = re.sub(r'_+', '_', s).strip('_')
    return s[:35]


def generate_descriptive_key(
    action: str,
    columns: list[str] | str | None = None,
    method: str | None = None,
    expr: str | None = None,
) -> str:
    """Return a short but meaningful dataset key prefix encoding the operation."""
    cc = _cols_token(columns)

    if action == "handle_missing":
        smap = {
            "knn": "knn", "drop_rows": "dropr", "drop_cols": "dropc",
            "mean": "fill_mn", "median": "fill_med", "mode": "fill_mo",
            "constant": "fill_c",
        }
        pfx = smap.get(method or "", "miss")
        raw = f"{pfx}_{cc}" if cc else pfx

    elif action == "remove_duplicates":
        raw = "dedup"

    elif action == "scale_columns":
        mmap = {"standard": "std", "minmax": "mm", "robust": "rob"}
        abbr = mmap.get(method or "", method or "scl")
        raw = f"scl_{abbr}_{cc}" if cc else f"scl_{abbr}"

    elif action == "encode_columns":
        mmap = {"label": "lbl", "onehot": "ohe"}
        abbr = mmap.get(method or "", method or "enc")
        raw = f"enc_{abbr}_{cc}" if cc else f"enc_{abbr}"

    elif action == "handle_outliers":
        mmap = {"remove": "rm", "cap": "cap"}
        abbr = mmap.get(method or "", method or "out")
        raw = f"out_{abbr}_{cc}" if cc else f"out_{abbr}"

    elif action == "standardize_text":
        mmap = {"lower": "lo", "upper": "up", "title": "ti", "none": ""}
        abbr = mmap.get(method or "", "")
        raw = f"txt_{abbr}_{cc}".rstrip("_") if cc else f"txt_{abbr}".rstrip("_")

    elif action == "coerce_types":
        mmap = {"numeric": "num", "string": "str"}
        abbr = mmap.get(method or "", method or "crc")
        raw = f"coerce_{abbr}_{cc}" if cc else f"coerce_{abbr}"

    elif action in {
        "log", "square", "cube", "interaction", "ratio",
        "binning", "one_hot", "standardize", "normalize",
        "fillna", "dropna",
    }:
        fe_abbr = {
            "log": "log", "square": "sq", "cube": "cu",
            "interaction": "int", "ratio": "rat", "binning": "bin",
            "one_hot": "ohe", "standardize": "zscore", "normalize": "norm",
            "fillna": "fill", "dropna": "dropna",
        }
        pfx = fe_abbr.get(action, action)
        raw = f"{pfx}_{cc}" if cc else pfx

    elif action == "custom_expr":
        raw = f"expr_{cc}" if cc else "expr"

    elif action == "filter":
        slug = _filter_expr_to_slug(expr) if expr else ""
        raw = f"filt_{slug}" if slug else "filt"

    else:
        raw = re.sub(r"[^a-zA-Z0-9_]", "_", action)[:20]

    # Truncate to 40 chars to keep keys readable
    return raw[:40]


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def empty_figure(title: str = "No plot yet.") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title)
    return fig


def build_comparison_figure(before: pd.Series, after: pd.Series, col_name: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Before", "After"])
    if pd.api.types.is_numeric_dtype(before):
        fig.add_histogram(x=before.dropna(), name="Before", marker_color="#94a3b8", row=1, col=1)
        fig.add_histogram(x=after.dropna(), name="After", marker_color="#4361ee", row=1, col=2)
    else:
        vc_before = before.value_counts().head(15)
        vc_after = after.value_counts().head(15)
        fig.add_bar(x=vc_before.index.astype(str), y=vc_before.values, name="Before",
                    marker_color="#94a3b8", row=1, col=1)
        fig.add_bar(x=vc_after.index.astype(str), y=vc_after.values, name="After",
                    marker_color="#4361ee", row=1, col=2)
    fig.update_layout(title=f"Before / After: {col_name}", showlegend=False,
                      height=280, margin=dict(t=50, b=30, l=40, r=20))
    return fig


def build_rowcount_figure(before_count: int, after_count: int, action: str) -> go.Figure:
    fig = go.Figure()
    removed = before_count - after_count
    fig.add_bar(
        x=["Before", "After"], y=[before_count, after_count],
        marker_color=["#94a3b8", "#4361ee"],
        text=[f"{before_count:,}", f"{after_count:,}"],
        textposition="outside", width=0.5,
    )
    fig.update_layout(
        title=f"{action}: {removed:,} rows removed ({before_count:,} → {after_count:,})",
        yaxis_title="Row count", height=280, margin=dict(t=50, b=30, l=60, r=20),
    )
    return fig


def figure_from_payload(payload: dict[str, Any]) -> go.Figure:
    status = payload.get("status")
    if status == "error":
        return empty_figure(payload.get("message", "Unable to render plot."))

    data = payload.get("data", {})
    plot_type = data.get("plot_type")
    fig = go.Figure()

    if plot_type == "categorical_count":
        fig.add_bar(x=data["categories"], y=data["counts"], marker_color="#4361ee")
        fig.update_layout(xaxis_title=data["column"], yaxis_title="Count")

    elif plot_type == "histogram":
        fig.add_bar(x=midpoints(data["bins"]), y=data["counts"], width=widths(data["bins"]),
                    marker_color="#1a1a2e")
        fig.update_layout(xaxis_title=data["column"], yaxis_title="Count")

    elif plot_type == "scatter":
        points = pd.DataFrame(data["points"])
        if points.empty:
            return empty_figure("No scatter data available.")
        if data.get("hue"):
            for label, group in points.groupby(data["hue"]):
                fig.add_scattergl(x=group[data["x"]], y=group[data["y"]], mode="markers",
                                  name=str(label), opacity=0.6)
        else:
            fig.add_scattergl(x=points[data["x"]], y=points[data["y"]], mode="markers",
                              name=f"{data['x']} vs {data['y']}", opacity=0.6)
        fig.update_layout(xaxis_title=data["x"], yaxis_title=data["y"])

    elif plot_type == "hist2d":
        fig.add_heatmap(x=midpoints(data["x_bins"]), y=midpoints(data["y_bins"]),
                        z=data["counts"], colorscale="YlOrRd")
        fig.update_layout(xaxis_title=data["x"], yaxis_title=data["y"])

    elif plot_type == "bar":
        bars = pd.DataFrame(data["bars"])
        hue_col = "hue_value"
        if not bars.empty and hue_col in bars.columns:
            for label, group in bars.groupby(hue_col):
                fig.add_bar(x=group[data["y"]], y=group["value"], name=str(label))
        else:
            fig.add_bar(x=bars[data["y"]], y=bars["value"], name="Value")
        fig.update_layout(xaxis_title=data["y"], yaxis_title=data["x"], barmode="group")

    elif plot_type == "box":
        points = pd.DataFrame(data["points"])
        if points.empty:
            return empty_figure("No box-plot data available.")
        hue_col = data.get("hue") or data["y"]
        for label, group in points.groupby(hue_col):
            fig.add_box(x=group[data["y"]], y=group[data["x"]], name=str(label),
                        boxpoints="outliers")
        fig.update_layout(xaxis_title=data["y"], yaxis_title=data["x"])

    elif plot_type == "heatmap":
        fig.add_heatmap(x=data["x_categories"], y=data["y_categories"], z=data["values"],
                        colorscale="Blues")
        fig.update_layout(xaxis_title=data["x"], yaxis_title=data["y"])

    elif plot_type == "regression":
        points = pd.DataFrame(data["points"])
        fig.add_scattergl(x=points[data["x"]], y=points[data["y"]], mode="markers",
                          name="Points", opacity=0.55)
        fit = data.get("fit") or {}
        if fit.get("x_fit") and fit.get("y_fit"):
            fig.add_scatter(x=fit["x_fit"], y=fit["y_fit"], mode="lines",
                            name=fit.get("fit_type", "Fit"),
                            line=dict(color="#d62828", width=3))
        fig.update_layout(
            xaxis_title=data["x"], yaxis_title=data["y"],
            annotations=[dict(
                xref="paper", yref="paper", x=0, y=1.12, showarrow=False,
                text=(f"Pearson r: {round(float(data['pearson_correlation']), 4)}"
                      f"  |  R\u00b2: {round(float(data['pearson_correlation'])**2, 4)}"),
            )],
        )

    elif plot_type == "multiline":
        for line in data["lines"]:
            fig.add_scatter(x=line["x"], y=line["y"], mode="lines", name=line["label"])
        x_title = data["column"] if data["mode"] == "1d" else data.get("x_column", "x")
        y_title = "Count" if data["mode"] == "1d" else data["column"]
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)

    elif plot_type == "correlation_matrix":
        cols = data["columns"]
        fig.add_heatmap(
            x=cols, y=cols, z=data["values"], colorscale="RdBu_r", zmid=0,
            text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in data["values"]],
            texttemplate="%{text}",
            hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
        )
        fig.update_layout(title=f"Correlation Matrix ({data['method'].title()})",
                          width=700, height=600)

    else:
        return empty_figure(f"Unsupported plot type: {plot_type}")

    return fig


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
APP_CSS = """
.tip-box {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border-left: 4px solid #1a1a2e;
  border-radius: 6px;
  padding: 12px 16px;
  margin-bottom: 12px;
  font-size: 0.93rem;
}
.small-note { color: #6c757d; font-size: 0.9rem; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 8px;
}
.metric {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border: 1px solid #dee2e6;
  border-radius: 8px;
  padding: 14px;
  text-align: center;
}
.metric .label {
  font-size: 0.72rem;
  color: #6c757d;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-weight: 600;
}
.metric .value {
  font-size: 1.3rem;
  font-weight: 700;
  color: #1a1a2e;
  margin-top: 2px;
}
.alert-stack { display: grid; gap: 6px; }
.bslib-sidebar-layout > .sidebar { border-right: 1px solid #dee2e6 !important; }
.card { transition: box-shadow 0.2s; }
.card:hover { box-shadow: 0 8px 24px rgba(0,0,0,0.08); }
.instr-box {
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
  border-left: 4px solid #4361ee;
  border-radius: 6px;
  padding: 12px 16px;
  margin-bottom: 14px;
  font-size: 0.92rem;
}
.workflow-header-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  padding: 0 12px 8px 12px;
}
.workflow-header-bar .step-tracker {
  flex: 1 1 auto;
  min-width: 0;
}
.step-tracker {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 0;
}
.step-pill {
  border: 1px solid #ced4da;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 0.82rem;
  background: #f8f9fa;
  color: #6c757d;
}
.step-pill.current {
  border-color: #4361ee;
  color: #1f3fbf;
  background: #eef2ff;
  font-weight: 700;
}
.step-pill.done {
  border-color: #198754;
  color: #146c43;
  background: #e9f7ef;
}
.workflow-top-right {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  flex-shrink: 0;
}
.workflow-footer-wrap {
  width: 100%;
  margin-top: auto;
  padding: 0;
  background: #f8f9fa;
  border-top: 1px solid #dee2e6;
}
.workflow-bottom-nav {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  max-width: 720px;
  margin: 0 auto;
}
.workflow-bottom-nav .btn {
  min-width: 100px;
}
/* Hide default page_navbar tab strip only; custom header keeps step pills + Open EDA */
.bslib-page-nav > nav .nav,
.bslib-page-nav > nav > ul.nav,
.navbar .navbar-nav,
.navbar-nav,
.nav-underline {
  display: none !important;
}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
app_ui = ui.page_navbar(
    # ── Load Tab ───────────────────────────────────────────────────────────
    ui.nav_panel(
        "Load",
        ui.layout_columns(
            ui.layout_columns(
                ui.card(
                    ui.card_header(ui.strong("Built-in Datasets")),
                    ui.input_select(
                        "builtin_dataset", "Choose built-in dataset",
                        {"sleep_health": "Sleep, Mobile and Stress",
                         "iris": "Iris", "tips": "Tips (Restaurant)"},
                    ),
                    ui.tooltip(
                        ui.input_action_button("load_builtin_btn", "Load Built-in Dataset",
                                               class_="btn-dark w-100"),
                        "Load the selected built-in dataset into session memory",
                    ),
                ),
                ui.card(
                    ui.card_header(ui.strong("Upload Dataset")),
                    ui.input_file("upload_file", "Upload CSV, Excel, JSON, or RDS",
                                  accept=[".csv", ".xlsx", ".xls", ".json", ".rds"]),
                    ui.tooltip(
                        ui.input_action_button("load_upload_btn", "Load Uploaded File",
                                               class_="btn-outline-dark w-100"),
                        "Parse and load the uploaded file",
                    ),
                ),
                col_widths=[12, 12],
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header(ui.strong("Active Dataset")),
                    ui.input_select("dataset_picker", "Active dataset version", {}),
                    ui.output_ui("active_dataset_summary"),
                    ui.download_button("download_active", "Download Active Dataset (CSV)",
                                       class_="btn-outline-dark btn-sm mt-2"),
                ),
                ui.card(
                    ui.card_header(ui.strong("Dataset History")),
                    ui.output_data_frame("history_table"),
                    full_screen=True,
                ),
                col_widths=[12, 12],
            ),
            col_widths=[4, 8],
        ),
        ui.output_ui("overview_ready_banner"),
        ui.output_ui("overview_priority_recommendation"),
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.strong("Missing Value Overview")),
                ui.output_ui("overview_missing_content"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("Duplicate Overview")),
                ui.output_ui("overview_duplicate_content"),
                full_screen=True,
            ),
            col_widths=[6, 6],
        ),
        ui.card(
            ui.card_header(ui.strong("Scale Review (Numeric Columns)")),
            ui.output_ui("overview_scale_content"),
            full_screen=True,
        ),
    ),

    # ── Cleaning Tab ───────────────────────────────────────────────────────
    ui.nav_panel(
        "Cleaning",
        ui.output_ui("cleaning_context_banner"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h6("Cleaning / Preprocessing", class_="text-uppercase fw-bold"),
                ui.hr(),
                ui.input_select("clean_df_picker", "Dataset to clean", {}),
                ui.hr(),
                ui.tooltip(
                    ui.input_select(
                        "clean_action", "Action",
                        {
                            "handle_missing": "Handle missing values",
                            "remove_duplicates": "Remove duplicates",
                            "scale_columns": "Scale numeric columns",
                            "encode_columns": "Encode categorical columns",
                            "handle_outliers": "Handle outliers",
                            "standardize_text": "Standardize text (whitespace & case)",
                            "coerce_types": "Coerce column types",
                        },
                    ),
                    "Choose a preprocessing operation to apply to the selected dataset",
                ),
                ui.input_selectize("clean_columns", "Columns", [], multiple=True),
                ui.panel_conditional(
                    "input.clean_action === 'handle_outliers'",
                    ui.input_select("clean_single_column", "Single column", {"": "— select a column —"}),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'handle_missing'",
                    ui.tooltip(
                        ui.input_select(
                            "clean_strategy", "Missing-value strategy",
                            {
                                "knn": "k-NN imputation (recommended)",
                                "drop_rows": "Drop rows",
                                "drop_cols": "Drop columns",
                                "mean": "Fill with mean",
                                "median": "Fill with median",
                                "mode": "Fill with mode",
                                "constant": "Fill with constant",
                            },
                            selected="knn",
                        ),
                        "How to handle missing values in selected columns",
                    ),
                    ui.input_text("clean_constant_value", "Constant value", "",
                                  placeholder="e.g., 0 or unknown"),
                    ui.panel_conditional(
                        "input.clean_strategy === 'knn'",
                        ui.input_numeric("clean_knn_k", "k (neighbours)", 5, min=1, max=100),
                        ui.div(
                            {"class": "tip-box", "style": "margin-top:8px;"},
                            ui.tags.small(
                                ui.strong("k-NN Imputation: "),
                                "Fills missing values by finding k similar rows using other "
                                "numeric columns as features, then averaging those neighbours' "
                                "values. Only numeric columns with ≥ 80 % valid values in the "
                                "rows that need imputation are used as features. Features are "
                                "automatically scaled to unit variance before distance "
                                "computation so no single column dominates. "
                                "Rows with no valid features are dropped. "
                                "Uses a KD-tree internally so it is efficient even for large "
                                "datasets (tens of thousands of rows).",
                            ),
                        ),
                    ),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'scale_columns'",
                    ui.tooltip(
                        ui.input_select(
                            "clean_scale_method", "Scaling method",
                            {"standard": "Standard", "minmax": "Min-Max", "robust": "Robust"},
                        ),
                        "Algorithm for rescaling numeric values to a standard range",
                    ),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'encode_columns'",
                    ui.tooltip(
                        ui.input_select(
                            "clean_encode_method", "Encoding method",
                            {"label": "Label encode", "onehot": "One-hot encode"},
                        ),
                        "Method for converting categorical values to numbers",
                    ),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'handle_outliers'",
                    ui.tooltip(
                        ui.input_select(
                            "clean_outlier_action", "Outlier action",
                            {"remove": "Remove rows", "cap": "Cap values"},
                        ),
                        "How to treat values outside the IQR fence boundaries",
                    ),
                    ui.input_numeric("clean_iqr", "IQR multiplier", 1.5, min=0.5, step=0.5),
                    ui.tags.small(
                        {"class": "small-note"},
                        "IQR = Q3 - Q1. Outliers lie beyond Q1 - k*IQR or Q3 + k*IQR. "
                        "Use 1.5 for mild, 3.0 for extreme.",
                    ),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'standardize_text'",
                    ui.tooltip(
                        ui.input_select(
                            "clean_text_case", "Case transform",
                            {"lower": "Lowercase", "upper": "Uppercase",
                             "title": "Title Case", "none": "No change"},
                        ),
                        "Letter case normalization to apply to string columns",
                    ),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'coerce_types'",
                    ui.tooltip(
                        ui.input_select(
                            "clean_coerce_target", "Target type",
                            {"numeric": "Numeric (non-convertible → NaN)", "string": "String"},
                        ),
                        "Target data type — non-convertible values become NaN for numeric",
                    ),
                ),
                ui.input_radio_buttons(
                    "clean_save_mode", "Apply mode",
                    {"derived": "Save as derived version", "current": "Apply to current version"},
                    selected="current", inline=False,
                ),
                ui.hr(),
                ui.layout_columns(
                    ui.tooltip(
                        ui.input_action_button("preview_clean_btn", "Preview",
                                               class_="btn-outline-dark w-100"),
                        "Preview the result without changing the active dataset",
                    ),
                    ui.tooltip(
                        ui.input_action_button("apply_clean_btn", "Apply",
                                               class_="btn-dark w-100"),
                        "Apply transformation and save the result",
                    ),
                    col_widths=[6, 6],
                ),
                ui.output_ui("cleaning_result_summary"),
                ui.output_ui("cleaning_whats_next"),
                width="380px",
            ),
            ui.card(
                ui.card_header(ui.strong("Cleaning Preview")),
                ui.output_data_frame("cleaning_preview_table"),
                ui.download_button("download_cleaned", "Download Cleaned Preview (CSV)",
                                   class_="btn-outline-dark btn-sm mt-2"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("Before / After Comparison")),
                output_widget("plot_clean_comparison", height="300px"),
            ),
        ),
    ),

    # ── Feature Engineering Tab ────────────────────────────────────────────
    ui.nav_panel(
        "Feature Engineering",
        ui.output_ui("feature_unlock_banner"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h6("Feature Engineering", class_="text-uppercase fw-bold"),
                ui.hr(),
                ui.input_select("feature_df_picker", "Dataset to transform", {}),
                ui.hr(),
                ui.tooltip(
                    ui.input_select(
                        "feature_method", "Method",
                        {
                            "log": "Log transform",
                            "square": "Square",
                            "cube": "Cube",
                            "interaction": "Interaction",
                            "ratio": "Ratio",
                            "binning": "Binning",
                            "one_hot": "One-hot encoding",
                            "standardize": "Standardize",
                            "normalize": "Normalize",
                            "fillna": "Fill missing values",
                            "dropna": "Drop missing rows",
                            "custom_expr": "Custom New Column",
                        },
                    ),
                    "Type of feature transformation — see explanation below",
                ),
                ui.output_ui("feature_explanation"),
                ui.panel_conditional(
                    "input.feature_method !== 'custom_expr'",
                    ui.tooltip(
                        ui.input_select("feature_col1", "Primary column", {}),
                        "Column to transform (required for all methods except Custom)",
                    ),
                    ui.tooltip(
                        ui.input_select("feature_col2", "Secondary column", {}),
                        "Second column — only used for Interaction and Ratio transforms",
                    ),
                ),
                ui.panel_conditional(
                    "input.feature_method === 'custom_expr'",
                    ui.input_text(
                        "feature_custom_expr", "Algebraic expression",
                        placeholder="e.g., col_a * 2 + col_b",
                    ),
                    ui.div(
                        {"class": "tip-box", "style": "margin-top:8px;"},
                        ui.tags.small(
                            ui.strong("Custom New Column: "),
                            "Enter a pandas-eval expression referencing existing column names "
                            "(e.g., ",
                            ui.tags.code("(price - cost) / price"),
                            "). The result is added as a new column. If a column name "
                            "contains spaces, wrap it in backticks. Errors for non-existent "
                            "columns or invalid syntax are reported immediately.",
                        ),
                        ui.output_ui("feature_custom_expr_columns"),
                    ),
                ),
                ui.input_text("feature_new_column", "New column name (optional)", ""),
                ui.panel_conditional(
                    "input.feature_method === 'binning'",
                    ui.input_numeric("feature_bins", "Number of bins", 4, min=2, step=1),
                    ui.input_checkbox("feature_labels", "Use interval labels", False),
                ),
                ui.panel_conditional(
                    "input.feature_method === 'one_hot'",
                    ui.input_text("feature_prefix", "Dummy prefix (optional)", ""),
                    ui.input_checkbox("feature_drop_first", "Drop first category", False),
                ),
                ui.panel_conditional(
                    "input.feature_method === 'fillna'",
                    ui.input_select(
                        "feature_fill_strategy", "Fill strategy",
                        {"mean": "Mean", "median": "Median", "mode": "Mode", "constant": "Constant"},
                    ),
                    ui.input_text("feature_fill_value", "Constant fill value", "",
                                  placeholder="e.g., 0 or missing"),
                ),
                ui.input_radio_buttons(
                    "feature_save_mode", "Apply mode",
                    {"derived": "Save as derived version", "current": "Apply to current version"},
                    selected="current",
                ),
                ui.hr(),
                ui.layout_columns(
                    ui.tooltip(
                        ui.input_action_button("preview_feature_btn", "Preview",
                                               class_="btn-outline-dark w-100"),
                        "Preview the transformation without saving",
                    ),
                    ui.tooltip(
                        ui.input_action_button("apply_feature_btn", "Apply",
                                               class_="btn-dark w-100"),
                        "Apply transformation and save the result",
                    ),
                    col_widths=[6, 6],
                ),
                ui.output_ui("feature_result_summary"),
                width="380px",
            ),
            ui.card(
                ui.card_header(ui.strong("Feature Preview")),
                ui.output_data_frame("feature_preview_table"),
                ui.download_button("download_featured", "Download Feature Preview (CSV)",
                                   class_="btn-outline-dark btn-sm mt-2"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("Before / After Comparison")),
                output_widget("plot_feature_comparison", height="300px"),
            ),
        ),
    ),

    # ── EDA Tab ────────────────────────────────────────────────────────────
    ui.nav_panel(
        "EDA",
        ui.output_ui("eda_context_banner"),
        ui.card(
            ui.card_header(ui.strong("Dataset")),
            ui.input_select("eda_df_picker", "Dataset to explore", {}),
        ),
        # Filtering
        ui.card(
            ui.card_header(ui.strong("Filtering")),
            ui.layout_columns(
                ui.input_text_area(
                    "filter_expr", "Pandas query expression",
                    placeholder=(
                        'Example: (age > 30) & (gender == "Female")  '
                        '|  Use `backticks` for column names with spaces'
                    ),
                    rows=2,
                ),
                ui.div(
                    ui.input_radio_buttons(
                        "filter_save_mode", "Filter mode",
                        {"derived": "Save filtered version",
                         "current": "Replace current version"},
                        selected="derived", inline=True,
                    ),
                    ui.input_action_button("apply_filter_btn", "Apply Filter",
                                           class_="btn-dark"),
                ),
                col_widths=[8, 4],
            ),
        ),
        # Summary tables
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.strong("Data Preview")),
                ui.input_numeric("head_rows", "Rows", 8, min=1, max=50, step=1),
                ui.output_data_frame("head_table"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("Describe — Numeric")),
                ui.output_data_frame("describe_num_table"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("Describe — Categorical")),
                ui.output_data_frame("describe_cat_table"),
                full_screen=True,
            ),
            col_widths=[4, 4, 4],
        ),
        ui.card(
            ui.card_header(ui.strong("Column Types")),
            ui.output_data_frame("column_types_table"),
            full_screen=True,
        ),
        # 1D and 2D plots
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.strong("1D Plot")),
                ui.input_select("plot1d_column", "Column", {}),
                ui.input_numeric("plot1d_bins", "Bins for numeric histogram", 30, min=5, max=100),
                ui.tags.small({"class": "small-note"}, "More bins = finer detail; fewer = smoother shape"),
                ui.input_checkbox("plot1d_normalize", "Normalize counts", False),
                ui.input_checkbox("plot1d_logx", "Log-scale X", False),
                ui.input_checkbox("plot1d_logy", "Log-scale Y", False),
                ui.input_action_button("render_1d_btn", "Render 1D Plot", class_="btn-dark btn-sm"),
                output_widget("plot_1d", height="380px"),
                ui.output_ui("plot1d_stats"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("2D Plot")),
                ui.input_select("plot2d_x", "X column", {}),
                ui.input_select("plot2d_y", "Y column", {}),
                ui.input_select("plot2d_hue", "Hue (optional)", {"": "None"}),
                ui.tooltip(
                    ui.input_select(
                        "plot2d_kind", "2D plot kind",
                        {"auto": "Auto", "hist": "2D histogram", "scatter": "Scatter",
                         "line": "Line", "bar": "Bar", "box": "Box", "heatmap": "Heatmap"},
                    ),
                    "Chart type — Auto detects based on column types",
                ),
                ui.input_checkbox("plot2d_logx", "Log-scale X", False),
                ui.input_checkbox("plot2d_logy", "Log-scale Y", False),
                ui.input_action_button("render_2d_btn", "Render 2D Plot", class_="btn-dark btn-sm"),
                output_widget("plot_2d", height="380px"),
                full_screen=True,
            ),
            col_widths=[6, 6],
        ),
        # Regression and Multiline
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.strong("Regression")),
                ui.layout_columns(
                    ui.input_select("regression_x", "X column", {}),
                    ui.input_select("regression_y", "Y column", {}),
                    col_widths=[6, 6],
                ),
                ui.layout_columns(
                    ui.tooltip(
                        ui.input_numeric("regression_order", "Polynomial order", 1, min=1, max=5),
                        "Degree of the polynomial fit: 1=linear, 2=quadratic, 3=cubic, etc.",
                    ),
                    ui.div(
                        ui.input_checkbox("regression_logx", "Log-scale x", False),
                        ui.input_checkbox("regression_robust", "Robust fit", False),
                        ui.input_checkbox("regression_lowess", "LOWESS fit", False),
                    ),
                    col_widths=[6, 6],
                ),
                ui.input_action_button("render_regression_btn", "Render Regression",
                                       class_="btn-dark btn-sm"),
                output_widget("plot_regression", height="380px"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("Multiline")),
                ui.input_select("multiline_value", "Value column", {}),
                ui.input_select("multiline_group", "Group by", {}),
                ui.input_numeric("multiline_bins", "Histogram bins", 20, min=5, max=80),
                ui.tags.small({"class": "small-note"},
                              "Number of equal-width bins for grouping the distribution"),
                ui.input_checkbox("multiline_normalize", "Normalize counts", False),
                ui.input_action_button("render_multiline_btn", "Render Multiline",
                                       class_="btn-dark btn-sm"),
                output_widget("plot_multiline", height="380px"),
                full_screen=True,
            ),
            col_widths=[6, 6],
        ),
        # Correlation
        ui.card(
            ui.card_header(ui.strong("Correlation Matrix")),
            ui.layout_columns(
                ui.tooltip(
                    ui.input_select(
                        "corr_method", "Method",
                        {"pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall"},
                    ),
                    "Pearson measures linear correlation; Spearman/Kendall measure monotonic association",
                ),
                ui.input_action_button("render_corr_btn", "Render Correlation Matrix",
                                       class_="btn-dark btn-sm"),
                col_widths=[4, 4],
            ),
            output_widget("plot_correlation", height="520px"),
            full_screen=True,
        ),
    ),

    # ── Navbar configuration ───────────────────────────────────────────────
    title=ui.tags.span("STAT 5243 Data Workbench",
                       style="font-weight:800; letter-spacing:0.5px;"),
    id="main_nav",
    theme=shinyswatch.theme.lux,
    fillable=False,
    header=ui.div(
        ga_head_tags,
        ui.busy_indicators.use(),
        ui.tags.style(APP_CSS),
        ui.div(
            {"class": "workflow-header-bar"},
            ui.output_ui("workflow_step_tracker"),
            ui.output_ui("workflow_top_right_control"),
        ),
        ui.output_ui("message_stack"),
    ),
    footer=ui.output_ui("workflow_bottom_nav"),
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def server(input, output, session):
    datasets_state = reactive.value(OrderedDict())
    active_key_state = reactive.value(None)
    messages_state = reactive.value([])

    cleaning_preview_df = reactive.value(pd.DataFrame())
    cleaning_preview_meta = reactive.value("")
    feature_preview_df = reactive.value(pd.DataFrame())
    feature_preview_meta = reactive.value("")

    plot1d_payload = reactive.value(None)
    plot1d_stats_text = reactive.value("")
    plot2d_payload = reactive.value(None)
    regression_payload = reactive.value(None)
    multiline_payload = reactive.value(None)
    corr_payload = reactive.value(None)
    clean_comparison_fig = reactive.value(None)
    feature_comparison_fig = reactive.value(None)
    overview_ready_banner_state = reactive.value("")
    overview_recommendation_target = reactive.value("Cleaning")
    cleaning_next_steps_state = reactive.value([])
    shown_instruction_tabs_state = reactive.value([])
    last_non_eda_tab_state = reactive.value("Load")

    # Tracks whether a multi-source conflict modal is pending
    _conflict_pending = reactive.value(False)

    def push_message(level: str, text: str) -> None:
        """Show one banner at a time; each new message replaces the previous."""
        messages_state.set([{"level": level, "text": text}])

    def clear_previews() -> None:
        cleaning_preview_df.set(pd.DataFrame())
        cleaning_preview_meta.set("")
        cleaning_next_steps_state.set([])
        feature_preview_df.set(pd.DataFrame())
        feature_preview_meta.set("")

    def _select_main_tab(tab_name: str) -> None:
        """Best-effort tab selection across Shiny API variants."""
        update_candidates = (
            getattr(ui, "update_navs", None),
            getattr(ui, "update_navset", None),
        )
        for updater in update_candidates:
            if updater is None:
                continue
            try:
                updater("main_nav", selected=tab_name, session=session)
                return
            except Exception:
                continue

    def _post_load_success(readiness_text: str) -> None:
        overview_ready_banner_state.set(readiness_text)
        _select_main_tab("Load")

    def _show_tab_instruction_modal(tab_name: str) -> None:
        modal_content: dict[str, tuple[str, list[str]]] = {
            "Load": (
                "Load + Overview Quick Start",
                [
                    "Load a built-in dataset or upload your own file first.",
                    "Review the missing, duplicate, and scale summaries before cleaning.",
                    "Use the recommendation card to jump to the next best step.",
                ],
            ),
            "Cleaning": (
                "Cleaning Quick Start",
                [
                    "Pick the dataset version to clean from the sidebar selector.",
                    "Preview before apply whenever possible.",
                    "Start with missing values or duplicates from the Load overview.",
                ],
            ),
            "Feature Engineering": (
                "Feature Engineering Quick Start",
                [
                    "Choose a dataset and method from the left sidebar.",
                    "Preview the transformation before applying it.",
                    "If data has not been cleaned yet, consider cleaning first.",
                ],
            ),
            "EDA": (
                "EDA Quick Start",
                [
                    "Use EDA at any stage, not just at the end.",
                    "Start with Data Preview and Describe, then move to plots.",
                    "Use the Correlation Matrix to validate post-cleaning changes.",
                ],
            ),
        }
        if tab_name not in modal_content:
            return
        title, bullets = modal_content[tab_name]
        ui.modal_show(
            ui.modal(
                ui.tags.ul(*[ui.tags.li(line) for line in bullets]),
                title=title,
                easy_close=True,
                footer=ui.modal_button("Got it"),
            )
        )

    @reactive.effect
    @reactive.event(input.main_nav)
    def _show_guided_instructions():
        if _conflict_pending.get():
            return
        tab_name = input.main_nav()
        if not tab_name:
            return
        shown = list(shown_instruction_tabs_state.get())
        if tab_name in shown:
            return
        if tab_name not in ("Load", "Cleaning", "Feature Engineering", "EDA"):
            return
        _show_tab_instruction_modal(tab_name)
        shown.append(tab_name)
        shown_instruction_tabs_state.set(shown)

    @reactive.effect
    @reactive.event(input.main_nav)
    def _remember_last_non_eda_tab():
        tab_name = input.main_nav()
        if tab_name and tab_name != "EDA":
            last_non_eda_tab_state.set(tab_name)

    @reactive.effect
    @reactive.event(input.workflow_open_eda_btn)
    def _open_eda_from_guided_nav():
        _select_main_tab("EDA")

    @reactive.effect
    @reactive.event(input.workflow_prev_btn)
    def _go_prev_workflow_tab():
        workflow_tabs = ["Load", "Cleaning", "Feature Engineering"]
        current_tab = input.main_nav() or "Load"
        if current_tab == "EDA":
            _select_main_tab(last_non_eda_tab_state.get())
            return
        if current_tab not in workflow_tabs:
            _select_main_tab("Load")
            return
        current_idx = workflow_tabs.index(current_tab)
        if current_idx > 0:
            _select_main_tab(workflow_tabs[current_idx - 1])

    @reactive.effect
    @reactive.event(input.workflow_next_btn)
    def _go_next_workflow_tab():
        workflow_tabs = ["Load", "Cleaning", "Feature Engineering"]
        current_tab = input.main_nav() or "Load"
        if current_tab == "EDA":
            _select_main_tab(last_non_eda_tab_state.get())
            return
        if current_tab not in workflow_tabs:
            _select_main_tab("Load")
            return
        current_idx = workflow_tabs.index(current_tab)
        if current_idx < len(workflow_tabs) - 1:
            _select_main_tab(workflow_tabs[current_idx + 1])
        else:
            push_message("info", "You are at the end of the guided workflow. Open EDA anytime.")

    def _cleaning_next_steps(action: str) -> list[dict[str, str]]:
        action_map: dict[str, list[dict[str, str]]] = {
            "handle_missing": [
                {"text": "Imputation complete. Check Correlation Matrix in EDA to validate shifted relationships.", "target": "EDA"},
                {"text": "Review duplicate rows in Load before additional transforms.", "target": "Load"},
                {"text": "Proceed to Feature Engineering once the data quality looks stable.", "target": "Feature Engineering"},
            ],
            "remove_duplicates": [
                {"text": "Duplicates removed. Recheck missing-value percentages in Load.", "target": "Load"},
                {"text": "Inspect distributions in EDA to confirm repeated patterns are gone.", "target": "EDA"},
                {"text": "Continue with feature transforms after quality checks.", "target": "Feature Engineering"},
            ],
            "scale_columns": [
                {"text": "Scaling applied. Validate transformed distributions in EDA 1D plots.", "target": "EDA"},
                {"text": "If missingness remains, return to Cleaning before modeling features.", "target": "Cleaning"},
                {"text": "Move to Feature Engineering for downstream transforms.", "target": "Feature Engineering"},
            ],
            "encode_columns": [
                {"text": "Encoding complete. Verify output columns and cardinality in EDA.", "target": "EDA"},
                {"text": "Review active dataset summary to confirm expected shape changes.", "target": "Load"},
                {"text": "Continue into Feature Engineering for interactions or ratios.", "target": "Feature Engineering"},
            ],
            "handle_outliers": [
                {"text": "Outlier handling complete. Compare before/after spread in EDA.", "target": "EDA"},
                {"text": "Revisit Load scale review for updated ranges.", "target": "Load"},
                {"text": "Proceed to Feature Engineering after confirming stability.", "target": "Feature Engineering"},
            ],
            "standardize_text": [
                {"text": "Text standardization complete. Check categorical levels in EDA tables.", "target": "EDA"},
                {"text": "If duplicates were caused by case/spacing, rerun duplicate check in Load.", "target": "Load"},
                {"text": "Continue with encoding or feature transforms next.", "target": "Feature Engineering"},
            ],
            "coerce_types": [
                {"text": "Type coercion complete. Validate column types in EDA.", "target": "EDA"},
                {"text": "Re-evaluate missing values created by coercion in Load.", "target": "Load"},
                {"text": "Proceed to feature creation once type checks pass.", "target": "Feature Engineering"},
            ],
        }
        return action_map.get(action, [
            {"text": "Cleaning operation applied successfully.", "target": "Cleaning"},
            {"text": "Validate outcomes with EDA before the next step.", "target": "EDA"},
            {"text": "Continue to Feature Engineering when ready.", "target": "Feature Engineering"},
        ])

    # ── Core reactive calcs ─────────────────────────────────────────────

    @reactive.calc
    def current_record() -> dict[str, Any] | None:
        datasets = datasets_state.get()
        active_key = active_key_state.get()
        if active_key is None:
            return None
        return datasets.get(active_key)

    @reactive.calc
    def current_df() -> pd.DataFrame | None:
        record = current_record()
        return None if record is None else record["df"]

    @reactive.calc
    def current_stage_state() -> dict[str, Any]:
        return derive_active_stage_state(
            datasets_state.get(),
            active_key_state.get(),
        )

    @reactive.calc
    def clean_active_df() -> pd.DataFrame | None:
        datasets = datasets_state.get()
        key = input.clean_df_picker()
        if key and key in datasets:
            return datasets[key]["df"]
        return current_df()

    @reactive.calc
    def feature_active_df() -> pd.DataFrame | None:
        datasets = datasets_state.get()
        key = input.feature_df_picker()
        if key and key in datasets:
            return datasets[key]["df"]
        return current_df()

    @reactive.calc
    def eda_active_df() -> pd.DataFrame | None:
        datasets = datasets_state.get()
        key = input.eda_df_picker()
        if key and key in datasets:
            return datasets[key]["df"]
        return current_df()

    def _tab_picker_key(picker_input: str) -> str | None:
        """Return the key currently selected in a tab picker (or active_key)."""
        datasets = datasets_state.get()
        try:
            val = getattr(input, picker_input)()
        except Exception:
            val = None
        if val and val in datasets:
            return val
        return active_key_state.get()

    # ── Sync dataset picker dropdowns ───────────────────────────────────

    @reactive.effect
    def _sync_dataset_picker() -> None:
        datasets = datasets_state.get()
        choices = {key: f"{record['label']} [{key}]"
                   for key, record in datasets.items()}
        ui.update_select("dataset_picker", choices=choices,
                         selected=active_key_state.get(), session=session)

    @reactive.effect
    def _sync_tab_pickers() -> None:
        datasets = datasets_state.get()
        active_key = active_key_state.get()
        choices = {key: f"{record['label']} [{key}]"
                   for key, record in datasets.items()}
        for picker in ("clean_df_picker", "feature_df_picker", "eda_df_picker"):
            try:
                current_val = getattr(input, picker)()
            except Exception:
                current_val = None
            selected = current_val if (current_val and current_val in datasets) else active_key
            ui.update_select(picker, choices=choices, selected=selected, session=session)

    @reactive.effect
    @reactive.event(input.dataset_picker)
    def _activate_from_picker() -> None:
        key = input.dataset_picker()
        if key and key in datasets_state.get():
            active_key_state.set(key)
            clear_previews()

    # ── Column input sync (split by tab) ────────────────────────────────

    @reactive.effect
    def _sync_clean_col_inputs() -> None:
        df = clean_active_df()
        action = input.clean_action()
        if df is None:
            empty: dict[str, str] = {}
            ui.update_selectize("clean_columns", choices=empty, selected=[], session=session)
            ui.update_select("clean_single_column",
                             choices={"": "— select a column —"},
                             selected="", session=session)
            return

        all_cols = [str(c) for c in df.columns]
        numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df[c])]

        if action == "scale_columns":
            col_choices = {c: c for c in numeric_cols}
        elif action in ("encode_columns", "standardize_text"):
            col_choices = {c: c for c in categorical_cols}
        else:
            col_choices = {c: c for c in all_cols}

        ui.update_selectize("clean_columns", choices=col_choices, selected=[], session=session)

        single_base = numeric_cols if numeric_cols else all_cols
        ui.update_select(
            "clean_single_column",
            choices={"": "— select a column —", **{c: c for c in single_base}},
            selected="",
            session=session,
        )

    @reactive.effect
    def _sync_feature_col_inputs() -> None:
        df = feature_active_df()
        if df is None:
            empty: dict[str, str] = {}
            for w in ("feature_col1", "feature_col2"):
                ui.update_select(w, choices=empty, session=session)
            return
        all_cols = [str(c) for c in df.columns]
        ui.update_select("feature_col1", choices={c: c for c in all_cols}, session=session)
        ui.update_select(
            "feature_col2",
            choices={"": "None", **{c: c for c in all_cols}},
            selected="",
            session=session,
        )

    @reactive.effect
    def _sync_eda_col_inputs() -> None:
        df = eda_active_df()
        if df is None:
            empty: dict[str, str] = {}
            for w in ("plot1d_column", "plot2d_x", "plot2d_y",
                      "regression_x", "regression_y",
                      "multiline_value", "multiline_group"):
                ui.update_select(w, choices=empty, session=session)
            ui.update_select("plot2d_hue", choices={"": "None"}, selected="", session=session)
            return

        all_cols = [str(c) for c in df.columns]
        numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df[c])]
        typed_cols = {c: f"{c} (num)" if c in numeric_cols else f"{c} (cat)" for c in all_cols}

        ui.update_select("plot1d_column", choices={c: c for c in all_cols}, session=session)
        ui.update_select("plot2d_x", choices=typed_cols, session=session)
        ui.update_select("plot2d_y", choices=typed_cols, session=session)
        ui.update_select("plot2d_hue", choices={"": "None", **typed_cols},
                         selected="", session=session)
        ui.update_select("regression_x", choices={c: c for c in numeric_cols}, session=session)
        ui.update_select("regression_y", choices={c: c for c in numeric_cols}, session=session)
        ui.update_select("multiline_value", choices={c: c for c in numeric_cols}, session=session)
        ui.update_select("multiline_group", choices={c: c for c in categorical_cols},
                         session=session)

    # ── Dataset loading ─────────────────────────────────────────────────

    def _check_and_prompt_conflict(new_datasets: OrderedDict, new_key: str) -> bool:
        """Show a conflict modal if ≥2 source datasets exist. Returns True if conflict."""
        source_keys = [k for k in new_datasets.keys()
                       if k == "original" or k.startswith("loaded_")]
        if len(source_keys) >= 2:
            choices = {k: f"{new_datasets[k]['label']} [{k}]" for k in source_keys}
            ui.modal_show(
                ui.modal(
                    ui.p(
                        "You have loaded more than one source dataset. "
                        "Please choose one to keep. All derived versions "
                        "(cleaned, feature-engineered, filtered) will be removed "
                        "and you will start fresh from the chosen dataset."
                    ),
                    ui.input_radio_buttons(
                        "modal_dataset_choice",
                        "Keep this dataset:",
                        choices=choices,
                        selected=new_key,
                    ),
                    title="Multiple Datasets Loaded",
                    footer=ui.div(
                        ui.input_action_button(
                            "modal_confirm_btn", "Confirm", class_="btn-dark me-2"
                        ),
                        ui.modal_button("Cancel"),
                    ),
                    easy_close=False,
                )
            )
            _conflict_pending.set(True)
            return True
        return False

    @reactive.effect
    @reactive.event(input.load_builtin_btn)
    def _load_builtin() -> None:
        name = input.builtin_dataset()
        try:
            df = load_builtin_dataset(name)
            prefix = "original" if not datasets_state.get() else "loaded"
            datasets, key = register_dataset_version(
                datasets_state.get(), df,
                prefix=prefix, label=f"Built-in: {name}", transform="built-in load",
            )
            datasets_state.set(datasets)
            active_key_state.set(key)
            clear_previews()
            if not _check_and_prompt_conflict(datasets, key):
                _post_load_success(
                    "Your dataset is ready — here is what we found before you start cleaning."
                )
                push_message("success", f"Loaded built-in dataset '{name}' as [{key}].")
        except Exception as exc:
            push_message("error", f"Failed to load built-in dataset: {exc}")

    @reactive.effect
    @reactive.event(input.load_upload_btn)
    def _load_uploaded() -> None:
        files = input.upload_file()
        if not files:
            push_message("warning", "Choose a file before loading.")
            return
        info = files[0]
        try:
            df = load_uploaded_dataset(info["datapath"], info["name"])
            prefix = "original" if not datasets_state.get() else "loaded"
            datasets, key = register_dataset_version(
                datasets_state.get(), df,
                prefix=prefix, label=f"Upload: {info['name']}",
                transform=f"uploaded {Path(info['name']).suffix.lower()}",
            )
            datasets_state.set(datasets)
            active_key_state.set(key)
            clear_previews()
            if not _check_and_prompt_conflict(datasets, key):
                _post_load_success(
                    "Your dataset is ready — here is what we found before you start cleaning."
                )
                push_message("success", f"Loaded '{info['name']}' as [{key}].")
        except Exception as exc:
            push_message("error", f"Failed to load uploaded file: {exc}")

    @reactive.effect
    @reactive.event(input.modal_confirm_btn)
    def _modal_confirm() -> None:
        chosen_key = input.modal_dataset_choice()
        datasets = datasets_state.get()
        if not chosen_key or chosen_key not in datasets:
            return
        # Keep only the chosen source dataset
        new_datasets: OrderedDict[str, dict[str, Any]] = OrderedDict()
        new_datasets[chosen_key] = datasets[chosen_key]
        datasets_state.set(new_datasets)
        active_key_state.set(chosen_key)
        _conflict_pending.set(False)
        clear_previews()
        ui.modal_remove()
        _post_load_success(
            "Your dataset is ready — here is what we found before you start cleaning."
        )
        push_message("success",
                     f"Kept dataset [{chosen_key}]. All other versions removed. "
                     "Starting fresh from this dataset.")

    # ── Cleaning operations ─────────────────────────────────────────────

    def compute_cleaning_result() -> tuple[pd.DataFrame, str, list[tuple[str, str]]]:
        """Run the selected cleaning op. Returns (transformed_df, summary, extra_messages)."""
        df = clean_active_df()
        if df is None:
            raise ValueError("Load a dataset first.")

        action = input.clean_action()
        extra_msgs: list[tuple[str, str]] = []

        if action == "handle_missing":
            columns = list(input.clean_columns() or [])
            strategy = input.clean_strategy()

            if strategy in ("drop_rows", "drop_cols") and not columns:
                raise ValueError(
                    f"Select at least one column before using '{strategy}'. "
                    "Without a column selection the operation would scan ALL columns and "
                    "drop far more rows than expected."
                )

            if strategy == "knn":
                if not columns:
                    raise ValueError("Select at least one column for k-NN imputation.")
                k = int(input.clean_knn_k())
                transformed, warning_msg = cleaning.knn_impute(df, columns, k=k)
                n_dropped = len(df) - len(transformed)
                summary = (
                    f"k-NN imputation (k={k}) on {columns}: "
                    f"{n_dropped} rows dropped, {len(transformed)}/{len(df)} rows retained."
                )
                if warning_msg:
                    extra_msgs.append(("warning", warning_msg))
            else:
                transformed = cleaning.handle_missing(
                    df, columns=columns or None,
                    strategy=strategy,
                    constant_value=coerce_text_value(input.clean_constant_value()),
                )
                summary = f"handle_missing strategy='{strategy}' → shape {transformed.shape}."

        elif action == "remove_duplicates":
            transformed = cleaning.remove_duplicates(df)
            n_dupes = len(df) - len(transformed)
            return (
                transformed,
                f"remove_duplicates: removed {n_dupes} duplicate rows ({len(transformed)}/{len(df)} rows kept).",
                [],
            )

        elif action == "scale_columns":
            columns = list(input.clean_columns() or [])
            if not columns:
                raise ValueError("Select one or more numeric columns to scale.")
            transformed = cleaning.scale_columns(df, columns=columns,
                                                  method=input.clean_scale_method())
            summary = f"scale_columns method='{input.clean_scale_method()}' on {columns}."

        elif action == "encode_columns":
            columns = list(input.clean_columns() or [])
            if not columns:
                raise ValueError("Select one or more categorical columns to encode.")
            transformed = cleaning.encode_columns(df, columns=columns,
                                                   method=input.clean_encode_method())
            summary = f"encode_columns method='{input.clean_encode_method()}' on {columns}."

        elif action == "handle_outliers":
            column = input.clean_single_column()
            if not column:
                raise ValueError("Select a column for outlier handling.")
            diagnostics = cleaning.detect_outliers(df, column=column,
                                                    iqr_multiplier=float(input.clean_iqr()))
            transformed = cleaning.handle_outliers(df, column=column,
                                                    action=input.clean_outlier_action(),
                                                    iqr_multiplier=float(input.clean_iqr()))
            return transformed, (
                f"Outlier handling on '{column}': {diagnostics['n_outliers']} outliers "
                f"(Q1={diagnostics['q1']:.2f}, Q3={diagnostics['q3']:.2f}, "
                f"IQR={diagnostics['iqr']:.2f}, "
                f"bounds=[{diagnostics['lower_bound']:.2f}, {diagnostics['upper_bound']:.2f}], "
                f"multiplier={input.clean_iqr()})."
            ), []

        elif action == "standardize_text":
            columns = list(input.clean_columns() or [])
            numeric_selected = [c for c in columns
                                 if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
            if numeric_selected:
                extra_msgs.append(("warning",
                    f"Columns {numeric_selected} are numeric and will be converted to strings."))
            transformed = cleaning.standardize_text(
                df, columns=columns or None, case=input.clean_text_case(),
            )
            summary = f"standardize_text case='{input.clean_text_case()}' → shape {transformed.shape}."

        elif action == "coerce_types":
            columns = list(input.clean_columns() or [])
            if not columns:
                raise ValueError("Select one or more columns to coerce.")
            transformed = cleaning.coerce_column_types(df, columns=columns,
                                                        target=input.clean_coerce_target())
            summary = f"coerce_types target='{input.clean_coerce_target()}' on {columns}."

        else:
            raise ValueError(f"Unsupported cleaning action: {action}")

        return transformed, summary, extra_msgs

    def apply_transformed_result(
        transformed: pd.DataFrame,
        *,
        mode: str,
        prefix: str,
        label: str,
        transform: str,
        source_key: str | None = None,
        apply_kind: str = "filter",
    ) -> str:
        datasets = datasets_state.get()
        src_key = source_key or active_key_state.get()
        if src_key is None:
            raise ValueError("No active dataset to update.")
        if mode == "current":
            datasets_state.set(overwrite_dataset_version(
                datasets, src_key, transformed,
                transform=transform,
                apply_kind=apply_kind,
            ))
            active_key_state.set(src_key)
            return src_key
        datasets, key = register_dataset_version(
            datasets, transformed,
            prefix=prefix, label=label,
            source_key=src_key, transform=transform,
            apply_kind=apply_kind,
        )
        datasets_state.set(datasets)
        active_key_state.set(key)
        return key

    @reactive.effect
    @reactive.event(input.preview_clean_btn)
    def _preview_cleaning() -> None:
        try:
            transformed, summary, extra_msgs = compute_cleaning_result()
            for level, msg in extra_msgs:
                push_message(level, msg)
            df = clean_active_df()
            action = input.clean_action()
            if action == "remove_duplicates" and df is not None:
                dupes = cleaning.get_duplicates(df)
                cleaning_preview_df.set(
                    dupes.head(20) if not dupes.empty else transformed.head(20)
                )
            else:
                cleaning_preview_df.set(transformed.head(20))
            cleaning_preview_meta.set(summary)

            is_row_removal = (
                action == "remove_duplicates"
                or (action == "handle_missing"
                    and input.clean_strategy() in ("drop_rows", "drop_cols", "knn"))
                or (action == "handle_outliers" and input.clean_outlier_action() == "remove")
            )
            if is_row_removal and df is not None:
                clean_comparison_fig.set(
                    build_rowcount_figure(len(df), len(transformed),
                                         action.replace("_", " ").title())
                )
            else:
                if action == "handle_outliers":
                    col = input.clean_single_column()
                else:
                    cols = list(input.clean_columns() or [])
                    col = cols[0] if cols else None
                if (col and df is not None and col in df.columns
                        and col in transformed.columns):
                    clean_comparison_fig.set(
                        build_comparison_figure(df[col], transformed[col], col)
                    )
                else:
                    clean_comparison_fig.set(None)
            push_message("info", "Cleaning preview updated.")
        except Exception as exc:
            push_message("error", f"Cleaning preview failed: {exc}")

    @reactive.effect
    @reactive.event(input.apply_clean_btn)
    def _apply_cleaning() -> None:
        try:
            transformed, summary, extra_msgs = compute_cleaning_result()
            for level, msg in extra_msgs:
                push_message(level, msg)
            # Build descriptive key
            action = input.clean_action()
            columns = list(input.clean_columns() or [])
            method = None
            if action == "handle_missing":
                method = input.clean_strategy()
            elif action == "scale_columns":
                method = input.clean_scale_method()
            elif action == "encode_columns":
                method = input.clean_encode_method()
            elif action == "handle_outliers":
                method = input.clean_outlier_action()
                columns = [input.clean_single_column()]
            elif action == "standardize_text":
                method = input.clean_text_case()
            elif action == "coerce_types":
                method = input.clean_coerce_target()

            desc_key = generate_descriptive_key(action, columns or None, method)
            src_key = _tab_picker_key("clean_df_picker")
            desc_label = f"{desc_key.replace('_', ' ')} (from {src_key})"

            target_key = apply_transformed_result(
                transformed,
                mode=input.clean_save_mode(),
                prefix=desc_key,
                label=desc_label,
                transform=summary,
                source_key=src_key,
                apply_kind="cleaning",
            )
            cleaning_preview_df.set(transformed.head(20))
            cleaning_preview_meta.set(summary)
            cleaning_next_steps_state.set(_cleaning_next_steps(action))
            push_message("success", f"Cleaning applied → [{target_key}].")
        except Exception as exc:
            push_message("error", f"Cleaning apply failed: {exc}")

    # ── Feature Engineering operations ──────────────────────────────────

    @output
    @render.ui
    def feature_custom_expr_columns():
        df = feature_active_df()
        if df is None:
            return ui.tags.small()
        cols = ", ".join(df.columns.tolist())
        return ui.div(
            {"style": "margin-top:6px;"},
            ui.tags.small(
                {"style": "color:#555;"},
                ui.strong("Available columns: "),
                cols,
            ),
        )

    def compute_feature_result() -> tuple[pd.DataFrame, str, dict]:
        df = feature_active_df()
        if df is None:
            raise ValueError("Load a dataset first.")

        method = input.feature_method()
        col2 = input.feature_col2() or None

        transformed, meta = feature_engineering.apply_feature_engineering_to_df(
            df, method,
            input.feature_col1() or None,
            col2=col2,
            bins=int(input.feature_bins()),
            new_column=input.feature_new_column() or None,
            labels=bool(input.feature_labels()),
            prefix=input.feature_prefix() or None,
            drop_first=bool(input.feature_drop_first()),
            strategy=input.feature_fill_strategy(),
            fill_value=coerce_text_value(input.feature_fill_value()),
            expr=input.feature_custom_expr() if method == "custom_expr" else None,
        )
        parts = [f"{meta['feature_type']} → columns: {meta['output_columns']}"]
        if meta.get("formula"):
            parts.append(f"Formula: {meta['formula']}")
        if meta.get("mean") is not None:
            parts.append(f"mean={meta['mean']:.4f}, std={meta['std']:.4f}")
        if meta.get("min") is not None and meta.get("max") is not None:
            parts.append(f"min={meta['min']:.4f}, max={meta['max']:.4f}")
        if meta.get("fill_value_used") is not None:
            parts.append(f"fill_value={meta['fill_value_used']}")
        if meta.get("n_zero_denominator"):
            parts.append(f"zero-denominator rows: {meta['n_zero_denominator']}")
        if meta.get("rows_removed") is not None:
            parts.append(f"rows removed: {meta['rows_removed']}")
        return transformed, " | ".join(parts), meta

    @reactive.effect
    @reactive.event(input.preview_feature_btn)
    def _preview_feature() -> None:
        try:
            transformed, summary, meta = compute_feature_result()
            feature_preview_df.set(transformed.head(20))
            feature_preview_meta.set(summary)
            df = feature_active_df()
            input_col = (meta.get("input_columns") or [None])[0]
            output_cols = meta.get("output_columns", [])
            out_col = output_cols[0] if output_cols else None
            if (input_col and out_col and df is not None
                    and input_col in df.columns and out_col in transformed.columns):
                feature_comparison_fig.set(
                    build_comparison_figure(df[input_col], transformed[out_col],
                                            f"{input_col} → {out_col}")
                )
            else:
                feature_comparison_fig.set(None)
            if (input_col and out_col and df is not None
                    and input_col in df.columns and out_col in transformed.columns
                    and pd.api.types.is_numeric_dtype(df[input_col])
                    and pd.api.types.is_numeric_dtype(transformed[out_col])):
                b = df[input_col].dropna()
                a = transformed[out_col].dropna()
                push_message("info",
                    f"Before: mean={b.mean():.3f}, std={b.std():.3f} | "
                    f"After: mean={a.mean():.3f}, std={a.std():.3f}")
            push_message("info",
                "Feature preview updated. To undo, switch to a previous version in Load tab.")
        except Exception as exc:
            push_message("error", f"Feature preview failed: {exc}")

    @reactive.effect
    @reactive.event(input.apply_feature_btn)
    def _apply_feature() -> None:
        try:
            transformed, summary, meta = compute_feature_result()
            method = input.feature_method()
            col1 = input.feature_col1() or None
            col2 = input.feature_col2() or None
            new_col = input.feature_new_column() or None

            # Build descriptive prefix
            if method == "custom_expr":
                cols_for_key = [new_col] if new_col else None
            elif method in ("interaction", "ratio"):
                cols_for_key = [c for c in [col1, col2] if c]
            else:
                cols_for_key = [col1] if col1 else (
                    [new_col] if new_col else meta.get("output_columns")
                )
            desc_key = generate_descriptive_key(method, cols_for_key, None)
            src_key = _tab_picker_key("feature_df_picker")
            desc_label = f"{desc_key.replace('_', ' ')} (from {src_key})"

            target_key = apply_transformed_result(
                transformed,
                mode=input.feature_save_mode(),
                prefix=desc_key,
                label=desc_label,
                transform=summary,
                source_key=src_key,
                apply_kind="feature",
            )
            feature_preview_df.set(transformed.head(20))
            feature_preview_meta.set(summary)
            push_message("success", f"Feature engineering applied → [{target_key}].")
        except Exception as exc:
            push_message("error", f"Feature apply failed: {exc}")

    # ── Filter ──────────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.apply_filter_btn)
    def _apply_filter() -> None:
        df = eda_active_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        expr = input.filter_expr().strip()
        if not expr:
            push_message("warning", "Enter a pandas query expression first.")
            return
        try:
            filtered = EDA.apply_filter(df, expr)
            src_key = _tab_picker_key("eda_df_picker")
            desc_key = generate_descriptive_key("filter", expr=expr)
            desc_label = f"{src_key} | {desc_key}"
            target_key = apply_transformed_result(
                filtered,
                mode=input.filter_save_mode(),
                prefix=desc_key,
                label=desc_label,
                transform=f"filter: {expr}",
                source_key=src_key,
            )
            push_message("success",
                         f"Filter applied → [{target_key}]. "
                         f"Rows: {len(df):,} → {len(filtered):,}.")
        except Exception as exc:
            push_message(
                "error",
                f"Filter failed: {exc}. "
                "Check syntax — wrap each condition in parentheses before combining with "
                "& or |, e.g. (\"col_cat\" == \"sex\") & (\"col_num\" >= 5). "
                "Use == for equality, backticks for column names with spaces."
            )

    # ── EDA plot handlers ───────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.render_1d_btn)
    def _render_1d() -> None:
        df = eda_active_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        column = input.plot1d_column()
        if not column:
            push_message("warning", "Choose a column for the 1D plot.")
            return
        try:
            if pd.api.types.is_numeric_dtype(df[column]):
                payload = EDA.plot_numeric_1d(df, column=column,
                                               bins=int(input.plot1d_bins()),
                                               normalize=bool(input.plot1d_normalize()))
            else:
                payload = EDA.plot_categorical_1d(df, column=column,
                                                   normalize=bool(input.plot1d_normalize()))
            plot1d_payload.set(payload)
            if payload.get("status") == "warning":
                push_message("warning", payload.get("message", "1D plot rendered with warnings."))
            elif payload.get("status") == "error":
                push_message("error", payload.get("message", "1D plot failed."))
            else:
                push_message("success", "1D plot rendered.")
                col_data = df[column].dropna()
                if pd.api.types.is_numeric_dtype(col_data):
                    plot1d_stats_text.set(
                        f"n={len(col_data):,}  |  mean={col_data.mean():.3f}  |  "
                        f"median={col_data.median():.3f}  |  std={col_data.std():.3f}  |  "
                        f"skew={col_data.skew():.3f}  |  kurtosis={col_data.kurtosis():.3f}"
                    )
                else:
                    vc = col_data.value_counts()
                    plot1d_stats_text.set(
                        f"n={len(col_data):,}  |  unique={vc.shape[0]}  |  "
                        f"top='{vc.index[0]}' ({vc.iloc[0]:,})"
                    )
        except Exception as exc:
            push_message("error", f"1D plot failed: {exc}")

    @reactive.effect
    @reactive.event(input.render_2d_btn)
    def _render_2d() -> None:
        df = eda_active_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        try:
            kind = input.plot2d_kind()
            payload = EDA.plot_two_columns(
                df, x=input.plot2d_x(), y=input.plot2d_y(),
                hue=input.plot2d_hue() or None,
                kind=None if kind == "auto" else kind,
            )
            plot2d_payload.set(payload)
            if payload.get("status") == "warning":
                push_message("warning", payload.get("message", "2D plot rendered with warnings."))
            elif payload.get("status") == "error":
                push_message("error", payload.get("message", "2D plot failed."))
            else:
                push_message("success", "2D plot rendered.")
        except Exception as exc:
            push_message("error", f"2D plot failed: {exc}")

    @reactive.effect
    @reactive.event(input.render_regression_btn)
    def _render_regression() -> None:
        df = eda_active_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        try:
            payload = EDA.regression_analysis(
                df, x=input.regression_x(), y=input.regression_y(),
                order=int(input.regression_order()),
                logx=bool(input.regression_logx()),
                robust=bool(input.regression_robust()),
                lowess=bool(input.regression_lowess()),
            )
            regression_payload.set(payload)
            if payload.get("status") == "warning":
                push_message("warning", payload.get("message", "Regression rendered with warnings."))
            elif payload.get("status") == "error":
                push_message("error", payload.get("message", "Regression failed."))
            else:
                push_message("success", "Regression rendered.")
                from scipy.stats import pearsonr
                common = df[[input.regression_x(), input.regression_y()]].dropna()
                if len(common) > 2:
                    r, p = pearsonr(common.iloc[:, 0], common.iloc[:, 1])
                    push_message("info",
                        f"Pearson r = {r:.4f}, R\u00b2 = {r**2:.4f}, "
                        f"p-value = {p:.2e} (n={len(common)})")
        except Exception as exc:
            push_message("error", f"Regression failed: {exc}")

    @reactive.effect
    @reactive.event(input.render_multiline_btn)
    def _render_multiline() -> None:
        df = eda_active_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        try:
            payload = EDA.plot_multiline(
                df, column=input.multiline_value(),
                group_by=input.multiline_group() or None,
                normalize=bool(input.multiline_normalize()),
                nbins=int(input.multiline_bins()),
            )
            multiline_payload.set(payload)
            if payload.get("status") == "warning":
                push_message("warning", payload.get("message", "Multiline rendered with warnings."))
            elif payload.get("status") == "error":
                push_message("error", payload.get("message", "Multiline failed."))
            else:
                push_message("success", "Multiline plot rendered.")
        except Exception as exc:
            push_message("error", f"Multiline plot failed: {exc}")

    @reactive.effect
    @reactive.event(input.render_corr_btn)
    def _render_correlation() -> None:
        df = eda_active_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        try:
            payload = EDA.correlation_matrix(df, method=input.corr_method())
            corr_payload.set(payload)
            if payload.get("status") == "error":
                push_message("error", payload.get("message", "Correlation matrix failed."))
            else:
                push_message("success", "Correlation matrix rendered.")
        except Exception as exc:
            push_message("error", f"Correlation matrix failed: {exc}")

    # ── Download handlers ───────────────────────────────────────────────

    @render.download(filename="active_dataset.csv")
    def download_active():
        df = current_df()
        if df is not None:
            yield df.to_csv(index=False)

    @render.download(filename="cleaned_preview.csv")
    def download_cleaned():
        df = cleaning_preview_df.get()
        if not df.empty:
            yield df.to_csv(index=False)

    @render.download(filename="feature_preview.csv")
    def download_featured():
        df = feature_preview_df.get()
        if not df.empty:
            yield df.to_csv(index=False)

    # ── Render outputs ──────────────────────────────────────────────────

    @output
    @render.ui
    def workflow_step_tracker():
        stage = current_stage_state()
        try:
            current_tab = input.main_nav()
        except Exception:
            current_tab = "Load"

        # Linear pipeline only; EDA is optional and always available via Open EDA.
        completed = {
            "Load": bool(stage.get("loaded")),
            "Cleaning": bool(stage.get("cleaned")),
            "Feature Engineering": bool(stage.get("feature_engineered")),
        }
        steps = ["Load", "Cleaning", "Feature Engineering"]
        # While on EDA, highlight the last workflow tab so pills stay meaningful.
        highlight_tab = (
            current_tab if current_tab != "EDA" else last_non_eda_tab_state.get()
        )

        pills = []
        for idx, name in enumerate(steps, start=1):
            classes = ["step-pill"]
            prefix = f"{idx}. "
            if completed[name]:
                classes.append("done")
                prefix = f"{idx}. [done] "
            if highlight_tab == name:
                classes.append("current")
            pills.append(
                ui.tags.span({"class": " ".join(classes)}, f"{prefix}{name}")
            )

        return ui.div({"class": "step-tracker"}, *pills)

    @output
    @render.ui
    def workflow_top_right_control():
        return ui.div(
            {"class": "workflow-top-right"},
            ui.input_action_button("workflow_open_eda_btn", "Open EDA", class_="btn btn-outline-primary btn-sm"),
        )

    @output
    @render.ui
    def workflow_bottom_nav():
        try:
            current_tab = input.main_nav()
        except Exception:
            current_tab = "Load"

        back_label = "Back"
        next_label = "Next"
        if current_tab == "EDA":
            back_label = "Return"
            next_label = "Return"
        return ui.div(
            {"class": "workflow-footer-wrap"},
            ui.div(
                {"class": "workflow-bottom-nav"},
                ui.input_action_button("workflow_prev_btn", back_label, class_="btn btn-outline-dark btn-sm"),
                ui.input_action_button("workflow_next_btn", next_label, class_="btn btn-dark btn-sm"),
            ),
        )

    @output
    @render.ui
    def message_stack():
        messages = messages_state.get()
        if not messages:
            return ui.div()
        _level_map = {"info": "alert-info", "success": "alert-success",
                      "warning": "alert-warning", "error": "alert-danger"}
        return ui.div(
            {"class": "alert-stack", "style": "padding: 0 12px;"},
            *[ui.div(
                {"class": f"alert {_level_map.get(item['level'], 'alert-secondary')} py-2 mb-1",
                 "role": "alert"},
                item["text"],
            ) for item in messages],
        )

    @output
    @render.ui
    def cleaning_context_banner():
        stage = current_stage_state()
        if stage.get("cleaned"):
            return ui.div({"class": "alert alert-success py-2", "role": "status", "style": "margin-bottom: 12px;"},
                          "You are working on a cleaned dataset lineage. Continue refining or move to feature engineering.")
        if stage.get("loaded"):
            return ui.div({"class": "alert alert-info py-2", "role": "status", "style": "margin-bottom: 12px;"},
                          "You are cleaning raw data. Start with missing values or duplicates based on the Load overview section.")
        return ui.div()

    @output
    @render.ui
    def feature_unlock_banner():
        stage = current_stage_state()
        if not stage.get("loaded"):
            return ui.div({"class": "alert alert-warning py-2", "role": "status", "style": "margin-bottom: 12px;"},
                          "Load a dataset before applying feature transformations.")
        if stage.get("cleaned"):
            return ui.div()
        return ui.div(
            {"class": "alert alert-warning", "role": "status", "style": "margin-bottom: 12px;"},
            ui.strong("You have not cleaned your data yet. "),
            "Transforms may produce noisy results. ",
            ui.input_action_link("feature_unlock_go_cleaning", "Go to Cleaning ->"),
        )

    @reactive.effect
    @reactive.event(input.feature_unlock_go_cleaning)
    def _on_feature_unlock_go_cleaning():
        _select_main_tab("Cleaning")

    @output
    @render.ui
    def eda_context_banner():
        stage = current_stage_state()
        if not stage.get("loaded"):
            return ui.div({"class": "alert alert-warning py-2", "role": "status", "style": "margin-bottom: 12px;"},
                          "Load a dataset to begin exploration.")
        if stage.get("cleaned"):
            return ui.div({"class": "alert alert-success py-2", "role": "status", "style": "margin-bottom: 12px;"},
                          "You are exploring a cleaned version. Use these diagnostics to validate final quality.")
        return ui.div(
            {"class": "alert alert-info", "role": "status", "style": "margin-bottom: 12px;"},
            ui.strong("You are exploring raw data. "),
            "Anomalies may reflect issues to clean. ",
            ui.input_action_link("eda_context_go_cleaning", "Go to Cleaning ->"),
        )

    @reactive.effect
    @reactive.event(input.eda_context_go_cleaning)
    def _on_eda_context_go_cleaning():
        _select_main_tab("Cleaning")

    @output
    @render.ui
    def active_dataset_summary():
        df = current_df()
        record = current_record()
        if df is None or record is None:
            return ui.div({"class": "small-note"}, "Load a dataset to begin.")
        overview = current_overview(df)
        return ui.div(
            ui.p(ui.strong("Label: "), record["label"]),
            ui.p(ui.strong("Key: "), active_key_state.get()),
            ui.div(
                {"class": "metric-grid"},
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Rows"),
                       ui.div({"class": "value"}, str(overview["n_rows"]))),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Columns"),
                       ui.div({"class": "value"}, str(overview["n_cols"]))),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Missing"),
                       ui.div({"class": "value"}, str(overview["n_missing"]))),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Duplicates"),
                       ui.div({"class": "value"}, str(overview["n_duplicates"]))),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Source"),
                       ui.div({"class": "value"}, record["source_key"] or "-")),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Transform"),
                       ui.div({"class": "value"}, record["transform"] or "-")),
            ),
        )

    @output
    @render.data_frame
    def history_table():
        return render.DataGrid(format_history_table(datasets_state.get()))

    # ── Overview outputs ────────────────────────────────────────────────

    @output
    @render.ui
    def overview_ready_banner():
        message = overview_ready_banner_state.get().strip()
        if not message:
            return ui.div()
        return ui.div(
            {"class": "alert alert-info", "role": "status", "style": "margin-bottom: 12px;"},
            ui.strong("Dataset loaded. "),
            message,
        )

    @output
    @render.ui
    def overview_missing_content():
        df = current_df()
        if df is None:
            return ui.div({"class": "small-note"}, "Load a dataset to see missing value summary.")
        info = cleaning.get_column_info(df)
        has_missing = (info[info["Missing"] > 0]
                       [["Column", "Missing", "Missing %"]]
                       .sort_values("Missing", ascending=False)
                       .reset_index(drop=True))
        if has_missing.empty:
            return ui.div(
                {"class": "alert alert-success py-2"},
                ui.tags.strong("No missing values detected in the current dataset.")
            )
        rows_html = [
            ui.tags.tr(
                ui.tags.td(str(r["Column"])),
                ui.tags.td(str(int(r["Missing"]))),
                ui.tags.td(f"{r['Missing %']:.1f}%"),
            )
            for _, r in has_missing.iterrows()
        ]
        return ui.div(
            ui.tags.p({"class": "small-note"},
                      f"{len(has_missing)} of {len(df.columns)} columns have missing values."),
            ui.tags.table(
                {"class": "table table-sm table-bordered table-hover"},
                ui.tags.thead(ui.tags.tr(
                    ui.tags.th("Column"),
                    ui.tags.th("Missing Count"),
                    ui.tags.th("Missing %"),
                )),
                ui.tags.tbody(*rows_html),
            ),
        )

    @output
    @render.ui
    def overview_priority_recommendation():
        df = current_df()
        if df is None:
            return ui.div({"class": "small-note"}, "Load a dataset to get a recommended next step.")

        n_rows = len(df)
        n_missing = int(df.isna().sum().sum())
        missing_pct = (100.0 * n_missing / (n_rows * len(df.columns))) if n_rows and len(df.columns) else 0.0

        dup_df = cleaning.get_duplicates(df)
        n_duplicates = len(dup_df)
        duplicate_pct = (100.0 * n_duplicates / n_rows) if n_rows else 0.0

        numeric_df = df.select_dtypes(include="number")
        has_scale_signal = False
        if not numeric_df.empty:
            max_abs_means: list[float] = []
            for col in numeric_df.columns:
                series = numeric_df[col].dropna()
                if series.empty:
                    continue
                max_abs_means.append(abs(float(series.mean())))
            if len(max_abs_means) >= 2:
                positive_means = [v for v in max_abs_means if v > 0]
                if len(positive_means) >= 2:
                    has_scale_signal = (max(positive_means) / min(positive_means)) >= 100

        recommendation_title = "Suggested next step: validate in EDA"
        recommendation_body = (
            "No major data-quality flags were detected. Continue by exploring distributions "
            "and relationships before deciding on additional transformations."
        )
        cta_label = "Open EDA"
        cta_target = "EDA"

        if n_missing > 0:
            recommendation_title = "Suggested next step: address missing values first"
            recommendation_body = (
                f"Detected {n_missing:,} missing values ({missing_pct:.1f}% of all cells). "
                "Handle missingness before feature engineering to avoid propagating bias."
            )
            cta_label = "Go to Cleaning - Handle missing values"
            cta_target = "Cleaning"
        elif n_duplicates > 0:
            recommendation_title = "Suggested next step: remove duplicate rows"
            recommendation_body = (
                f"Detected {n_duplicates:,} duplicate rows ({duplicate_pct:.1f}% of rows). "
                "Removing duplicates can prevent over-weighting repeated observations."
            )
            cta_label = "Go to Cleaning - Remove duplicates"
            cta_target = "Cleaning"
        elif has_scale_signal:
            recommendation_title = "Suggested next step: review numeric scaling"
            recommendation_body = (
                "Numeric columns appear to be on very different scales. Consider scaling "
                "before distance-based operations and model fitting."
            )
            cta_label = "Go to Cleaning - Scale numeric columns"
            cta_target = "Cleaning"

        overview_recommendation_target.set(cta_target)
        return ui.div(
            {"class": "alert alert-primary", "role": "status", "style": "margin-bottom: 12px;"},
            ui.tags.div(ui.strong(recommendation_title)),
            ui.tags.p({"class": "mb-2 mt-1"}, recommendation_body),
            ui.input_action_link("overview_recommendation_cta", cta_label),
        )

    @reactive.effect
    @reactive.event(input.overview_recommendation_cta)
    def _on_overview_recommendation_cta_click():
        _select_main_tab(overview_recommendation_target.get())

    @output
    @render.ui
    def overview_duplicate_content():
        df = current_df()
        if df is None:
            return ui.div({"class": "small-note"}, "Load a dataset to see duplicate summary.")
        dup_df = cleaning.get_duplicates(df)
        n_dup = len(dup_df)
        n_total = len(df)
        if n_dup == 0:
            return ui.div(
                {"class": "alert alert-success py-2"},
                ui.tags.strong("No duplicate detected in current dataset.")
            )
        example = dup_df.head(3)
        example_rows = [
            ui.tags.tr(*[ui.tags.td(str(v)) for v in row])
            for row in example.values
        ]
        return ui.div(
            ui.tags.p(
                {"class": "small-note"},
                f"{n_dup:,} duplicate rows found out of {n_total:,} total "
                f"({100 * n_dup / n_total:.1f}%). Showing up to 3 example rows:",
            ),
            ui.tags.table(
                {"class": "table table-sm table-bordered table-hover"},
                ui.tags.thead(ui.tags.tr(
                    *[ui.tags.th(str(c)) for c in example.columns]
                )),
                ui.tags.tbody(*example_rows),
            ),
            ui.tags.p(
                {"class": "small-note mt-2"},
                "Use Cleaning → Remove duplicates to eliminate these rows.",
            ),
        )

    @output
    @render.ui
    def overview_scale_content():
        df = current_df()
        if df is None:
            return ui.div({"class": "small-note"}, "Load a dataset to see scale summary.")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return ui.div({"class": "small-note"}, "No numeric columns in this dataset.")
        rows_html = []
        for col in num_cols:
            s = df[col].dropna()
            if s.empty:
                rows_html.append(ui.tags.tr(
                    ui.tags.td(col), ui.tags.td("—"), ui.tags.td("—"), ui.tags.td("—"),
                ))
            else:
                rows_html.append(ui.tags.tr(
                    ui.tags.td(col),
                    ui.tags.td(f"{s.min():.4g}"),
                    ui.tags.td(f"{s.max():.4g}"),
                    ui.tags.td(f"{s.mean():.4g}"),
                ))
        return ui.div(
            ui.tags.p({"class": "small-note"}, f"{len(num_cols)} numeric columns."),
            ui.tags.table(
                {"class": "table table-sm table-bordered table-hover"},
                ui.tags.thead(ui.tags.tr(
                    ui.tags.th("Column"),
                    ui.tags.th("Min"),
                    ui.tags.th("Max"),
                    ui.tags.th("Mean"),
                )),
                ui.tags.tbody(*rows_html),
            ),
            ui.tags.p(
                {"class": "small-note mt-2"},
                "Large scale differences across columns may affect k-NN imputation and "
                "distance-based algorithms. Consider scaling before applying such methods.",
            ),
        )

    # ── EDA outputs ─────────────────────────────────────────────────────

    @output
    @render.data_frame
    def head_table():
        df = eda_active_df()
        if df is None:
            return render.DataGrid(pd.DataFrame())
        payload = EDA.show_head(df, n=int(input.head_rows()))
        return render.DataGrid(dataframe_from_payload(payload))

    @output
    @render.data_frame
    def describe_num_table():
        df = eda_active_df()
        if df is None:
            return render.DataGrid(pd.DataFrame())
        num_df = df.select_dtypes(include="number")
        if num_df.empty:
            return render.DataGrid(
                pd.DataFrame({"Note": ["No numeric columns in this dataset."]})
            )
        desc = num_df.describe().T.reset_index().rename(columns={"index": "column"})
        # Round for display
        for c in desc.columns:
            if c != "column":
                desc[c] = desc[c].apply(lambda v: round(v, 4) if pd.notnull(v) else v)
        return render.DataGrid(desc)

    @output
    @render.data_frame
    def describe_cat_table():
        df = eda_active_df()
        if df is None:
            return render.DataGrid(pd.DataFrame())
        cat_df = df.select_dtypes(exclude="number")
        if cat_df.empty:
            return render.DataGrid(
                pd.DataFrame({"Note": ["No categorical columns in this dataset."]})
            )
        desc = cat_df.describe(include="all").T.reset_index().rename(columns={"index": "column"})
        keep = [c for c in ["column", "count", "unique", "top", "freq"] if c in desc.columns]
        return render.DataGrid(desc[keep])

    @output
    @render.data_frame
    def column_types_table():
        df = eda_active_df()
        if df is None:
            return render.DataGrid(
                pd.DataFrame(columns=["column", "dtype", "is_numeric", "is_categorical"])
            )
        return render.DataGrid(current_column_types(df))

    # ── Cleaning outputs ─────────────────────────────────────────────────

    @output
    @render.ui
    def cleaning_result_summary():
        text = cleaning_preview_meta.get()
        return ui.div({"class": "small-note"}, text or "Preview a cleaning action to see a summary.")

    @output
    @render.ui
    def cleaning_whats_next():
        items = cleaning_next_steps_state.get()
        if not items:
            return ui.div()
        links = []
        for idx, item in enumerate(items[:3], start=1):
            links.append(
                ui.tags.li(
                    item["text"],
                    " ",
                    ui.input_action_link(f"cleaning_next_link_{idx}", f"Open {item['target']}"),
                )
            )
        return ui.div(
            {"class": "alert alert-primary mt-2 mb-0", "role": "status"},
            ui.tags.div(ui.strong("What's next?")),
            ui.tags.ul({"class": "mb-1 mt-2"}, *links),
        )

    @reactive.effect
    @reactive.event(input.cleaning_next_link_1)
    def _on_cleaning_next_link_1():
        items = cleaning_next_steps_state.get()
        if items:
            _select_main_tab(items[0]["target"])

    @reactive.effect
    @reactive.event(input.cleaning_next_link_2)
    def _on_cleaning_next_link_2():
        items = cleaning_next_steps_state.get()
        if len(items) >= 2:
            _select_main_tab(items[1]["target"])

    @reactive.effect
    @reactive.event(input.cleaning_next_link_3)
    def _on_cleaning_next_link_3():
        items = cleaning_next_steps_state.get()
        if len(items) >= 3:
            _select_main_tab(items[2]["target"])

    @output
    @render.data_frame
    def cleaning_preview_table():
        return render.DataGrid(cleaning_preview_df.get())

    # ── Feature Engineering outputs ──────────────────────────────────────

    @output
    @render.ui
    def feature_result_summary():
        text = feature_preview_meta.get()
        return ui.div({"class": "small-note"}, text or "Preview a feature step to see a summary.")

    @output
    @render.ui
    def feature_explanation():
        method = input.feature_method()
        explanations = {
            "log": "Applies log(1+x). Reduces right-skew and compresses large values.",
            "square": "Squares values (x²). Amplifies differences between large and small values.",
            "cube": "Cubes values (x³). Captures cubic relationships, preserves sign.",
            "interaction": "Multiplies two columns (x * y). Captures combined effects.",
            "ratio": "Divides col1 by col2. Useful for per-unit metrics (e.g., price per sqft).",
            "binning": "Groups continuous values into discrete bins using equal-width intervals.",
            "one_hot": "Creates binary 0/1 columns for each category. Required by most ML models.",
            "standardize": "Z-score normalization: (x - mean) / std. Centers data at 0.",
            "normalize": "Min-Max scaling to [0, 1]. Preserves shape, bounds values.",
            "fillna": "Replaces missing values with a computed or constant value.",
            "dropna": "Removes rows containing missing values in the selected column.",
            "custom_expr": (
                "Evaluates a custom algebraic expression to create a new column. "
                "Enter any pandas-eval expression referencing column names."
            ),
        }
        text = explanations.get(method, "")
        if not text:
            return ui.div()
        return ui.div({"class": "tip-box", "style": "margin-top: 8px;"},
                      ui.tags.small(text))

    @output
    @render.data_frame
    def feature_preview_table():
        return render.DataGrid(feature_preview_df.get())

    # ── Plot outputs ─────────────────────────────────────────────────────

    @output
    @render_plotly
    def plot_1d():
        payload = plot1d_payload.get()
        if payload is None:
            return empty_figure("Render a 1D plot to see output here.")
        fig = figure_from_payload(payload)
        if input.plot1d_logx():
            fig.update_xaxes(type="log")
        if input.plot1d_logy():
            fig.update_yaxes(type="log")
        return fig

    @output
    @render.ui
    def plot1d_stats():
        text = plot1d_stats_text.get()
        if not text:
            return ui.div()
        return ui.div({"class": "small-note", "style": "margin-top:6px;"}, text)

    @output
    @render_plotly
    def plot_2d():
        payload = plot2d_payload.get()
        if payload is None:
            return empty_figure("Render a 2D plot to see output here.")
        fig = figure_from_payload(payload)
        if input.plot2d_logx():
            fig.update_xaxes(type="log")
        if input.plot2d_logy():
            fig.update_yaxes(type="log")
        return fig

    @output
    @render_plotly
    def plot_regression():
        payload = regression_payload.get()
        if payload is None:
            return empty_figure("Render a regression plot to see output here.")
        fig = figure_from_payload(payload)
        if input.regression_logx():
            fig.update_xaxes(type="log")
        return fig

    @output
    @render_plotly
    def plot_multiline():
        payload = multiline_payload.get()
        if payload is None:
            return empty_figure("Render a multiline plot to see output here.")
        return figure_from_payload(payload)

    @output
    @render_plotly
    def plot_correlation():
        payload = corr_payload.get()
        if payload is None:
            return empty_figure("Render a correlation matrix to see output here.")
        return figure_from_payload(payload)

    @output
    @render_plotly
    def plot_clean_comparison():
        fig = clean_comparison_fig.get()
        return fig if fig else empty_figure("Preview a cleaning action to see comparison.")

    @output
    @render_plotly
    def plot_feature_comparison():
        fig = feature_comparison_fig.get()
        return fig if fig else empty_figure("Preview a feature to see comparison.")


app = App(app_ui, server)
