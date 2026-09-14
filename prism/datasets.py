"""Dataset loading and the cleaning pipelines that sit in front of the models.

Three of the five sources ship inside scikit-learn. The two atmospheric and
wine-quality files are bundled under ``data/`` so the app never needs a network
connection at runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_digits, load_wine

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CO2_FILE = DATA_DIR / "co2_daily_mlo.csv"
WINE_QUALITY_FILE = DATA_DIR / "winequality-red.csv"

FREQUENCIES = {
    "Weekly": ("W-SAT", 13, "weeks"),
    "Daily": ("D", 91, "days"),
    "Monthly": ("MS", 12, "months"),
}


# ==========================================================================
# atmospheric CO2
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_co2_raw() -> pd.DataFrame:
    """Daily in-situ CO2 concentration recorded at Mauna Loa Observatory."""
    raw = pd.read_csv(CO2_FILE)
    raw.columns = ["date", "co2_ppm"]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["co2_ppm"] = pd.to_numeric(raw["co2_ppm"], errors="coerce")
    return raw


def _inject_faults(df: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Deliberately corrupt a copy of the record.

    Real archives arrive with duplicated rows, sentinel values and blanks. The
    pipeline should catch all three, so the interface offers a switch that puts
    them back in to prove it does.
    """
    rng = np.random.default_rng(seed)
    work = df.copy()
    n = len(work)

    blanks = rng.choice(n, size=max(1, n // 90), replace=False)
    work.loc[blanks, "co2_ppm"] = np.nan

    sentinels = rng.choice(n, size=max(1, n // 400), replace=False)
    work.loc[sentinels, "co2_ppm"] = -99.99

    spikes = rng.choice(n, size=max(1, n // 900), replace=False)
    work.loc[spikes, "co2_ppm"] = work.loc[spikes, "co2_ppm"] * rng.uniform(1.6, 2.4, len(spikes))

    duplicated = work.iloc[rng.choice(n, size=max(1, n // 250), replace=False)].copy()
    work = pd.concat([work, duplicated], ignore_index=True)

    broken = rng.choice(len(work), size=max(1, n // 800), replace=False)
    work.loc[broken, "date"] = pd.NaT

    return work.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def clean_co2(
    frequency: str = "Weekly",
    interpolation: str = "time",
    iqr_multiplier: float = 1.5,
    drop_outliers: bool = False,
    inject_faults: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run the full cleaning pipeline and return the series plus an audit trail."""
    raw = load_co2_raw()
    source = _inject_faults(raw) if inject_faults else raw.copy()

    report: dict = {"rows_received": int(len(source))}

    # 1. type coercion -----------------------------------------------------
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["co2_ppm"] = pd.to_numeric(source["co2_ppm"], errors="coerce")

    # 2. rows without a usable timestamp cannot anchor a time series --------
    undated = int(source["date"].isna().sum())
    source = source.dropna(subset=["date"]).copy()
    report["rows_without_date"] = undated

    # 3. physically impossible readings become missing, not zero -----------
    impossible = int((source["co2_ppm"] <= 0).fillna(False).sum())
    source.loc[source["co2_ppm"] <= 0, "co2_ppm"] = np.nan
    report["non_positive_readings"] = impossible

    # 4. order and de-duplicate -------------------------------------------
    source = source.sort_values("date")
    duplicates = int(source["date"].duplicated().sum())
    source = source.drop_duplicates(subset="date", keep="first").reset_index(drop=True)
    report["duplicate_dates"] = duplicates

    report["missing_before_resample"] = int(source["co2_ppm"].isna().sum())
    report["rows_after_dedupe"] = int(len(source))

    # 5. place on a regular grid ------------------------------------------
    rule, window, unit = FREQUENCIES.get(frequency, FREQUENCIES["Weekly"])
    indexed = source.set_index("date")["co2_ppm"]
    series = indexed.asfreq("D") if rule == "D" else indexed.resample(rule).mean()

    report["grid_points"] = int(len(series))
    report["gaps_on_grid"] = int(series.isna().sum())
    report["longest_gap"] = _longest_run(series.isna().to_numpy())
    report["grid_unit"] = unit

    # 6. outlier rule, computed before any values are synthesised ----------
    observed = series.dropna()
    q1, q3 = observed.quantile(0.25), observed.quantile(0.75)
    iqr = q3 - q1
    lower = float(q1 - iqr_multiplier * iqr)
    upper = float(q3 + iqr_multiplier * iqr)
    outlier_mask = ((series < lower) | (series > upper)).fillna(False)

    report["iqr_lower"] = lower
    report["iqr_upper"] = upper
    report["outliers_flagged"] = int(outlier_mask.sum())
    report["outliers_removed"] = int(outlier_mask.sum()) if drop_outliers else 0

    if drop_outliers:
        series = series.mask(outlier_mask)

    # 7. fill the gaps -----------------------------------------------------
    filled_mask = series.isna()
    if interpolation == "none":
        series = series.dropna()
        filled_mask = filled_mask.reindex(series.index, fill_value=False)
    else:
        series = series.interpolate(method=interpolation, limit_area="inside")
        series = series.ffill().bfill()

    report["values_synthesised"] = int(filled_mask.sum())
    report["missing_after_fill"] = int(series.isna().sum())

    # 8. assemble the analysis table --------------------------------------
    clean = series.rename("co2_ppm").to_frame()
    clean["was_filled"] = filled_mask.reindex(clean.index, fill_value=False).to_numpy()
    clean["was_outlier"] = outlier_mask.reindex(clean.index, fill_value=False).to_numpy()
    clean = clean.reset_index().rename(columns={"index": "date"})
    clean.columns = ["date", "co2_ppm", "was_filled", "was_outlier"]

    # 9. engineered features ----------------------------------------------
    clean["year"] = clean["date"].dt.year
    clean["month"] = clean["date"].dt.month
    clean["month_name"] = clean["date"].dt.strftime("%b")
    clean["week_of_year"] = clean["date"].dt.isocalendar().week.astype(int)
    clean["day_of_year"] = clean["date"].dt.dayofyear
    clean["elapsed_days"] = (clean["date"] - clean["date"].min()).dt.days
    clean["rolling_mean"] = clean["co2_ppm"].rolling(window, min_periods=1, center=True).mean()
    std = clean["co2_ppm"].std(ddof=0)
    clean["anomaly_z"] = (clean["co2_ppm"] - clean["co2_ppm"].mean()) / (std if std else 1.0)

    annual = (
        clean.groupby("year", as_index=False)["co2_ppm"]
        .mean()
        .rename(columns={"co2_ppm": "annual_mean"})
    )
    annual["annual_change"] = annual["annual_mean"].diff()
    clean = clean.merge(annual, on="year", how="left")

    # 10. integrity gate ---------------------------------------------------
    assert clean["date"].notna().all(), "Cleaned series contains an invalid timestamp."
    assert clean["co2_ppm"].notna().all(), "Cleaned series still contains gaps."
    assert (clean["co2_ppm"] > 0).all(), "Cleaned series contains an impossible reading."
    assert not clean["date"].duplicated().any(), "Cleaned series repeats a timestamp."
    assert clean["date"].is_monotonic_increasing, "Cleaned series is out of order."

    report["rows_final"] = int(len(clean))
    report["span_start"] = str(clean["date"].min().date())
    report["span_end"] = str(clean["date"].max().date())
    report["co2_min"] = float(clean["co2_ppm"].min())
    report["co2_max"] = float(clean["co2_ppm"].max())
    report["co2_mean"] = float(clean["co2_ppm"].mean())
    report["integrity_checks_passed"] = 5

    return clean, report


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest consecutive True run."""
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return int(best)


@st.cache_data(show_spinner=False)
def decompose_co2(frequency: str = "Weekly") -> pd.DataFrame:
    """Split the series into trend, seasonal and residual components.

    The trend is a centred moving average over one full year; the seasonal term
    is the average detrended value for each position in the year; whatever is
    left over is the residual.
    """
    clean, _ = clean_co2(frequency=frequency)
    per_year = {"Weekly": 52, "Daily": 365, "Monthly": 12}.get(frequency, 52)
    window = per_year if per_year % 2 else per_year + 1

    work = clean[["date", "co2_ppm", "day_of_year", "month", "week_of_year", "year"]].copy()
    work["trend"] = work["co2_ppm"].rolling(window, center=True, min_periods=window // 2).mean()
    work["detrended"] = work["co2_ppm"] - work["trend"]

    cycle_key = {"Weekly": "week_of_year", "Daily": "day_of_year", "Monthly": "month"}[frequency]
    seasonal_map = work.groupby(cycle_key)["detrended"].transform("mean")
    work["seasonal"] = seasonal_map - seasonal_map.mean()
    work["residual"] = work["detrended"] - work["seasonal"]
    return work


# ==========================================================================
# wine cultivars
# ==========================================================================
CULTIVAR_LABELS = {0: "Cultivar A", 1: "Cultivar B", 2: "Cultivar C"}


@st.cache_data(show_spinner=False)
def load_cultivars() -> tuple[pd.DataFrame, pd.Series]:
    bundle = load_wine(as_frame=True)
    features = bundle.data.copy()
    features.columns = [c.replace("/", "_").replace(" ", "_").lower() for c in features.columns]
    truth = bundle.target.astype(int).map(CULTIVAR_LABELS).rename("cultivar")
    return features, truth


# ==========================================================================
# tumour biopsies
# ==========================================================================
RATIO_FEATURES = [
    "texture_to_radius",
    "perimeter_to_radius",
    "area_to_radius_squared",
    "worst_perimeter_to_radius",
    "worst_area_to_radius_squared",
]


def add_ratio_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-wise shape ratios.

    Nothing is fitted from the data here, so these can be created before the
    train/test split without leaking information across it.
    """
    out = frame.copy()
    eps = 1e-9
    out["texture_to_radius"] = out["mean texture"] / (out["mean radius"] + eps)
    out["perimeter_to_radius"] = out["mean perimeter"] / (out["mean radius"] + eps)
    out["area_to_radius_squared"] = out["mean area"] / (out["mean radius"] ** 2 + eps)
    out["worst_perimeter_to_radius"] = out["worst perimeter"] / (out["worst radius"] + eps)
    out["worst_area_to_radius_squared"] = out["worst area"] / (out["worst radius"] ** 2 + eps)
    return out


@st.cache_data(show_spinner=False)
def load_biopsies() -> tuple[pd.DataFrame, pd.Series]:
    """Nuclear measurements from breast tissue biopsies.

    The positive class is malignant, which inverts scikit-learn's original
    encoding. Recall on that class is the number that matters clinically.
    """
    bundle = load_breast_cancer(as_frame=True)
    features = bundle.data.copy()
    labels = (bundle.target.astype(int) == 0).astype(int).rename("malignant")
    return features, labels


# ==========================================================================
# handwritten digits
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_digit_images() -> tuple[np.ndarray, np.ndarray]:
    """1,797 handwritten digits as 8x8 grids with intensities from 0 to 16."""
    bundle = load_digits()
    return bundle.images.astype(np.float64), bundle.target.astype(np.int64)


# ==========================================================================
# red wine quality
# ==========================================================================
QUALITY_FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]

QUALITY_UNITS = {
    "fixed_acidity": "g/dm³ tartaric",
    "volatile_acidity": "g/dm³ acetic",
    "citric_acid": "g/dm³",
    "residual_sugar": "g/dm³",
    "chlorides": "g/dm³ NaCl",
    "free_sulfur_dioxide": "mg/dm³",
    "total_sulfur_dioxide": "mg/dm³",
    "density": "g/cm³",
    "ph": "pH",
    "sulphates": "g/dm³",
    "alcohol": "% vol",
}

QUALITY_PLAIN = {
    "fixed_acidity": "Fixed acidity",
    "volatile_acidity": "Volatile acidity",
    "citric_acid": "Citric acid",
    "residual_sugar": "Residual sugar",
    "chlorides": "Chlorides",
    "free_sulfur_dioxide": "Free sulphur dioxide",
    "total_sulfur_dioxide": "Total sulphur dioxide",
    "density": "Density",
    "ph": "pH",
    "sulphates": "Sulphates",
    "alcohol": "Alcohol",
}


@st.cache_data(show_spinner=False)
def load_wine_quality() -> tuple[pd.DataFrame, dict]:
    """Physicochemical readings and blind tasting scores for red wines."""
    raw = pd.read_csv(WINE_QUALITY_FILE, sep=";")
    if raw.shape[1] == 1:
        raw = pd.read_csv(WINE_QUALITY_FILE)

    raw.columns = (
        raw.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("\ufeff", "", regex=False)
    )

    report: dict = {"rows_received": int(len(raw))}

    work = raw.copy()
    for column in work.columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    before = len(work)
    work = work.drop_duplicates().reset_index(drop=True)
    report["duplicate_rows"] = int(before - len(work))

    report["missing_cells"] = int(work.isna().sum().sum())
    work = work.dropna().reset_index(drop=True)

    plausible = (
        work["fixed_acidity"].gt(0)
        & work["volatile_acidity"].gt(0)
        & work["citric_acid"].between(0, 1)
        & work["residual_sugar"].gt(0)
        & work["chlorides"].gt(0)
        & work["free_sulfur_dioxide"].ge(0)
        & work["total_sulfur_dioxide"].ge(0)
        & work["density"].between(0.98, 1.02)
        & work["ph"].between(2.0, 5.0)
        & work["sulphates"].gt(0)
        & work["alcohol"].gt(0)
        & work["quality"].between(0, 10)
    )
    report["out_of_range_rows"] = int((~plausible).sum())
    work = work.loc[plausible].reset_index(drop=True)

    if work.empty:
        raise ValueError("Cleaning removed every observation; check the source file.")

    assert work.isna().sum().sum() == 0
    assert not work.duplicated().any()
    assert work["quality"].between(0, 10).all()

    report["rows_final"] = int(len(work))
    report["quality_min"] = int(work["quality"].min())
    report["quality_max"] = int(work["quality"].max())
    return work, report
