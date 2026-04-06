## Overview

An interactive, code-free data workbench built with **Shiny for Python**. Users can load, clean, transform, and explore tabular datasets entirely in the browser. The app is organized into six tabs — **Guide**, **Load**, **Overview**, **Cleaning**, **Feature Engineering**, and **EDA** — with a polished Lux Bootstrap theme, per-tab dataset pickers, user instruction cards, and a full dataset version history.

The tabs are **not** strictly sequential. Overview and EDA can be visited at any point to support cleaning and feature-engineering decisions. The workflow is flexible by design.

All computation runs locally through pure Python module imports. There is no Flask, no REST API, and no external backend — user data never leaves the machine.

---

## Repository Structure

| File | Purpose |
|------|---------|
| `app.py` | Shiny app entrypoint — UI layout and server logic (6 tabs, per-tab dataset pickers, reactive versioning) |
| `eda.py` | EDA backend — summary tables, pandas query filtering, 6 plot families, correlation matrix |
| `data_cleaning.py` | Cleaning backend — 9 operations including k-NN imputation, validation, and pipeline support |
| `feature_engineering.py` | Feature engineering backend — 12 transforms including custom algebraic expressions |
| `tests.py` | Integration smoke tests — 7 test cases covering all modules |
| `test_data/` | Built-in dataset: Sleep, Mobile and Stress (15,000 rows, 13 columns) |
| `requirements.txt` | All Python dependencies (13 packages) |
| `REPORT.md` | Final project report (markdown source) |
| `report.pdf` | Final project report (PDF) |

---

## Features

### 1. Data Loading

- Upload **CSV, Excel (.xlsx/.xls), JSON, and RDS** files with robust error handling
- **3 built-in datasets:** Sleep/Mobile/Stress (15,000 rows), Iris (150 rows), Tips (244 rows)
- **Conflict resolution:** if more than one source dataset is loaded, a modal dialog prompts the user to choose one; all other datasets are removed so subsequent work starts from a single clean source
- Full **dataset version history** — every load, clean, or transform creates a descriptively named version; switch to any previous version via the Load tab picker

### 2. Overview (new)

A quick decision-support summary shown between Load and Cleaning, designed to inform cleaning and feature-engineering choices before diving in:

- **Missing Value Overview** — table of columns with missing values, counts, and percentages
- **Duplicate Overview** — count of fully duplicate rows with example rows; "No duplicate detected" if none
- **Scale Review** — min, max, and mean for every numeric column in tabular form
- Reminder that EDA provides in-depth analysis and can be consulted at any time

### 3. Data Cleaning and Preprocessing

- **9 operations:** handle missing values, remove duplicates, scale numeric columns, encode categorical columns, detect and handle outliers, standardize text, coerce column types
- **k-NN imputation (new, default):** fills missing values using k nearest neighbors computed from other numeric columns. Automatically selects feature columns with ≥ 80 % valid values in the rows to be imputed. Issues a warning if fewer than 50 % of rows can be matched. Configurable `k` parameter (default 5)
- **Drop Rows/Cols fix:** now requires at least one column to be selected — prevents accidentally dropping rows across all columns
- **Single column selector** for outlier handling is hidden for all other actions, eliminating ambiguity
- **Per-tab dataset picker:** choose which saved version to clean from the sidebar, independently of the Load tab
- **User instruction card** at the top of the Cleaning tab with EDA/Overview cross-references
- **Default save mode** is "Apply to current version" (overwrite in place); switch to "Save as derived version" to branch a new named copy
- **Preview-then-apply workflow** with before/after comparison charts

### 4. Feature Engineering

- **12 transforms:** log (log1p), square, cube, interaction (col1 × col2), ratio (col1 / col2), binning, one-hot encoding, standardize (z-score), normalize (min-max), fill NA, drop NA, **custom algebraic expression (new)**
- **Custom New Column:** enter any pandas-eval expression (e.g., `(price - cost) / price`) to create a new column; errors for non-existent columns or invalid syntax are reported immediately
- **Per-tab dataset picker:** choose which saved version to transform
- **User instruction card** with EDA cross-references
- **Descriptive version names:** derived datasets are named to encode the operation — e.g., `log_Age_01`, `expr_margin_01`

### 5. Exploratory Data Analysis (EDA)

- **Per-tab dataset picker:** choose which version to analyze for each operation independently
- **User instruction card** reminding users that EDA is useful at every stage, not just after cleaning
- **Summary tables:**
  - Data preview (adjustable row count)
  - **Describe — Numeric:** count, mean, std, min, 25 %, 50 %, 75 %, max (numeric columns only)
  - **Describe — Categorical:** count, unique, top, freq (categorical columns only)
  - Column types
- **Free-text pandas query filtering** with improved error guidance: wrap each condition in parentheses before combining, e.g. `("col_cat" == "sex") & ("col_num" >= 5)`
- **1D plots:** histograms and categorical bar charts with log-scale toggles, normalization, and persistent statistics (mean, median, std, skewness, kurtosis)
- **2D plots:** scatter, line, bar, box, heatmap, 2D histogram with optional color grouping
- **Regression analysis:** polynomial (order 1–5), robust, and LOWESS with Pearson r, R², and p-value
- **Multiline grouped plots** for cross-category distribution comparison
- **Correlation matrix heatmap** (Pearson, Spearman, Kendall) with annotated values

### 6. Non-Linear Workflow Philosophy

The tabs are not a rigid pipeline. Users are encouraged to:

- Visit **Overview** and **EDA** before and during cleaning to understand column distributions and scales
- Jump to **EDA → 1D Plot** to visualize outliers before deciding on an outlier strategy in Cleaning
- Use **EDA → Correlation Matrix** before Feature Engineering to identify redundant columns
- Apply cleaning or feature engineering operations on any saved version using the per-tab pickers, not just the most recent one

### 7. Descriptive Dataset Version Names

Derived datasets are named to encode the operation performed, making it easy to track versions:

| Operation | Example key |
|-----------|-------------|
| k-NN impute column `Age` | `knn_Age_01` |
| Drop rows for column `Income` | `dropr_Income_01` |
| Scale with Min-Max | `scl_mm_ColA_01` |
| One-hot encode `Gender` | `enc_ohe_Gender_01` |
| Log transform `Price` | `log_Price_01` |
| Custom expression, new col `margin` | `expr_margin_01` |
| Filter `age >= 5` | `filt_age_geq_5_01` |
| Filter `(age >= 5) & (type == "race")` | `filt_age_geq_5_type_eq_race_01` |

### 8. UI/UX

- **Lux Bootstrap theme** (shinyswatch) with custom CSS — gradient metric cards, instruction boxes, tip boxes
- **User instruction cards** on Cleaning, Feature Engineering, and EDA tabs with cross-tab references
- **20+ tooltips** throughout
- **Sidebar layouts** in Cleaning and Feature Engineering (collapsible on mobile)
- **Full-screen expandable cards** for all plots and tables
- **Busy indicators** during computations

### 9. Export

- **CSV download buttons** on the Load, Cleaning, and Feature Engineering tabs

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | Shiny for Python 1.0+ |
| Theme | shinyswatch (Lux Bootstrap) |
| Plotting | Plotly (with ScatterGL for large datasets) |
| Data | pandas, numpy |
| ML/Stats | scikit-learn (scalers, encoders), statsmodels (robust/LOWESS), scipy (pearsonr) |
| File I/O | openpyxl (Excel), pyreadr (RDS), seaborn (Tips dataset) |
| Testing | unittest (7 integration tests) |

---