"""Overview page: the masthead, the spectral band, and the index of analyses."""

from __future__ import annotations

import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .. import datasets, theme, ui


@st.cache_data(show_spinner=False)
def _spectral_traces() -> list[dict]:
    """One sparkline per analysis, each computed from that analysis's own data.

    The band is the first thing on the page, so it has to be honest: no shape
    here is drawn by hand.
    """
    traces = []

    # 1. the long climb -----------------------------------------------------
    co2, _ = datasets.clean_co2(frequency="Weekly")
    weekly = co2["co2_ppm"].to_numpy()
    traces.append(
        {
            "title": theme.MODULE_BY_KEY["refinery"]["title"],
            "caption": "Weekly CO\u2082, 1958\u20132025",
            "scale": f"{weekly.min():.0f}\u2013{weekly.max():.0f} ppm",
            "colour": theme.accent("refinery"),
            "y": _thin(weekly, 260),
        }
    )

    # 2. the seasonal breath underneath it ----------------------------------
    parts = datasets.decompose_co2("Weekly")
    seasonal = parts["seasonal"].to_numpy()[-260:]
    traces.append(
        {
            "title": theme.MODULE_BY_KEY["atmosphere"]["title"],
            "caption": "Seasonal cycle, detrended",
            "scale": f"\u00b1{np.abs(seasonal).max():.1f} ppm",
            "colour": theme.accent("atmosphere"),
            "y": seasonal,
        }
    )

    # 3. wine samples ordered along their main axis of variation ------------
    cultivars, _ = datasets.load_cultivars()
    scaled = StandardScaler().fit_transform(cultivars)
    pc1 = PCA(n_components=1, random_state=42).fit_transform(scaled).ravel()
    density, _ = np.histogram(pc1, bins=46)
    traces.append(
        {
            "title": theme.MODULE_BY_KEY["cultivars"]["title"],
            "caption": "Density along PC1",
            "scale": "36% variance",
            "colour": theme.accent("cultivars"),
            "y": _smooth(density.astype(float), 3),
        }
    )

    # 4. how cleanly the classifier separates the two classes ---------------
    features, labels = datasets.load_biopsies()
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=4000, solver="liblinear", random_state=42)
    model.fit(scaler.transform(x_train), y_train)
    scores = model.predict_proba(scaler.transform(x_test))[:, 1]
    fpr, tpr, _ = roc_curve(y_test, scores)
    traces.append(
        {
            "title": theme.MODULE_BY_KEY["diagnostics"]["title"],
            "caption": "Sorted malignancy scores",
            "scale": f"AUC {auc(fpr, tpr):.3f}",
            "colour": theme.accent("diagnostics"),
            "y": np.sort(scores),
        }
    )

    # 5. where ink actually falls on the writing grid -----------------------
    images, _ = datasets.load_digit_images()
    traces.append(
        {
            "title": theme.MODULE_BY_KEY["digits"]["title"],
            "caption": "Mean ink per pixel",
            "scale": "1,797 samples",
            "colour": theme.accent("digits"),
            "y": images.mean(axis=0).ravel(),
        }
    )

    # 6. the spread of alcohol across the cellar ----------------------------
    wines, _ = datasets.load_wine_quality()
    counts, _ = np.histogram(wines["alcohol"], bins=60)
    traces.append(
        {
            "title": theme.MODULE_BY_KEY["quality"]["title"],
            "caption": "Alcohol distribution",
            "scale": f"{wines['alcohol'].min():.1f}\u2013{wines['alcohol'].max():.1f}%",
            "colour": theme.accent("quality"),
            "y": counts.astype(float),
        }
    )

    return traces


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Light moving average so a coarse histogram reads as a curve."""
    if window < 2 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _thin(values: np.ndarray, target: int) -> np.ndarray:
    """Reduce a long series to roughly `target` points by block averaging."""
    if len(values) <= target:
        return values
    block = len(values) // target
    trimmed = values[: block * target]
    return trimmed.reshape(target, block).mean(axis=1)


def render() -> None:
    ui.masthead(
        "Six analyses across five public datasets, from the first pass over a raw "
        "archive to a neural network that trains while you watch. Every chart here "
        "is live: change an input and the result recomputes."
    )

    ui.spectral_stack(_spectral_traces())
    st.markdown(
        f'<p style="font-size:.78rem;color:{theme.TYPE_FAINT};margin:-.2rem 0 1.9rem 0;'
        f'line-height:1.5;">Each line above is drawn from the dataset behind that '
        f"analysis, not sketched for decoration.</p>",
        unsafe_allow_html=True,
    )

    ui.readouts(
        [
            {"label": "Analyses", "value": "6"},
            {"label": "Public datasets", "value": "5"},
            {"label": "Observations", "value": "22,447"},
            {"label": "Years of record", "value": "67"},
            {"label": "Model families", "value": "8"},
        ]
    )

    ui.section(
        "Where to start",
        "The analyses build on one another but each stands alone, so any order works.",
    )

    pages = st.session_state.get("_nav_pages", {})
    for module in theme.MODULES:
        left, right = st.columns([1, 0.17], vertical_alignment="center")
        with left:
            ui.index_row(module)
        with right:
            if st.button("Open", key=f"go_{module['key']}", width="stretch"):
                target = pages.get(module["key"])
                if target is not None:
                    st.switch_page(target)

    ui.spacer(1.2)
    ui.section("How this is put together")

    left, right = st.columns(2, gap="large")
    with left:
        ui.panel(
            "Everything recomputes",
            "No chart is a saved image. Each page re-runs its pipeline against the "
            "controls you set, and results are cached so repeated views stay quick.",
        )
        ui.panel(
            "The neural network has no framework",
            "Convolution, pooling, dropout, softmax cross-entropy and the Adam "
            "optimiser are written directly in NumPy. The backward pass is checked "
            "against numerical gradients in the test suite.",
        )
    with right:
        ui.panel(
            "Data stays local",
            "The atmospheric and wine-quality files are bundled in the repository; "
            "the other three ship inside scikit-learn. Nothing is fetched at runtime.",
        )
        ui.panel(
            "Splits come before decisions",
            "Test data is held out before any model is fitted, scaling is learned "
            "on training folds only, and cross-validation is reported alongside "
            "single-split results so the spread is visible.",
        )

    ui.spacer(0.6)
    ui.caveat(
        "<b>Data sources.</b> Atmospheric CO\u2082 from the Mauna Loa Observatory "
        "in-situ record. Wine cultivars, tumour biopsies and handwritten digits "
        "from the UCI Machine Learning Repository, distributed with scikit-learn. "
        "Red wine quality from Cortez et al., UCI Machine Learning Repository. "
        "All are public research datasets, used here for analysis and teaching."
    )
