"""Model fitting, cached so that moving a slider never retrains needlessly.

Every function takes only hashable arguments and loads its own data, which
keeps Streamlit's cache keys small and predictable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import datasets

SEED = 42


# ==========================================================================
# cultivar segmentation
# ==========================================================================
@st.cache_data(show_spinner=False)
def cluster_cultivars(
    features: tuple[str, ...],
    k: int,
    algorithm: str = "K-means",
    linkage_method: str = "ward",
    standardise: bool = True,
) -> dict:
    """Fit one clustering and report how well separated the result is."""
    frame, truth = load_cultivar_frame()
    matrix = frame[list(features)].to_numpy(dtype=float)

    if standardise:
        matrix = StandardScaler().fit_transform(matrix)

    if algorithm == "K-means":
        model = KMeans(n_clusters=k, n_init=20, random_state=SEED)
        labels = model.fit_predict(matrix)
        inertia = float(model.inertia_)
        centres = model.cluster_centers_
    else:
        tree = linkage(matrix, method=linkage_method)
        labels = fcluster(tree, t=k, criterion="maxclust") - 1
        inertia = float(
            sum(
                ((matrix[labels == c] - matrix[labels == c].mean(axis=0)) ** 2).sum()
                for c in np.unique(labels)
            )
        )
        centres = np.vstack([matrix[labels == c].mean(axis=0) for c in np.unique(labels)])

    distinct = len(np.unique(labels))
    scores = {
        "silhouette": float(silhouette_score(matrix, labels)) if distinct > 1 else float("nan"),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)) if distinct > 1 else float("nan"),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)) if distinct > 1 else float("nan"),
        "inertia": inertia,
        "adjusted_rand": float(adjusted_rand_score(truth.to_numpy(), labels)),
    }

    # A projection cannot have more axes than the space it came from.
    n_components = min(3, matrix.shape[1])
    projector = PCA(n_components=n_components, random_state=SEED)
    coords = projector.fit_transform(matrix)

    profile = pd.DataFrame(matrix, columns=list(features))
    profile["cluster"] = labels
    profile = profile.groupby("cluster")[list(features)].mean()

    per_sample = (
        silhouette_samples(matrix, labels) if distinct > 1 else np.zeros(len(labels))
    )

    return {
        "labels": labels,
        "scores": scores,
        "coords": coords,
        "explained": projector.explained_variance_ratio_,
        "loadings": pd.DataFrame(
            projector.components_[:2].T, index=list(features), columns=["PC1", "PC2"]
        ),
        "profile": profile,
        "silhouette_samples": per_sample,
        "truth": truth.to_numpy(),
        "centres": centres,
        "matrix": matrix,
    }


@st.cache_data(show_spinner=False)
def load_cultivar_frame() -> tuple[pd.DataFrame, pd.Series]:
    return datasets.load_cultivars()


@st.cache_data(show_spinner=False)
def cultivar_sweep(
    features: tuple[str, ...],
    k_values: tuple[int, ...],
    algorithm: str = "K-means",
    linkage_method: str = "ward",
    standardise: bool = True,
) -> pd.DataFrame:
    """Score a range of cluster counts so the elbow can be read off a chart."""
    rows = []
    for k in k_values:
        result = cluster_cultivars(features, k, algorithm, linkage_method, standardise)
        rows.append(
            {
                "k": k,
                "inertia": result["scores"]["inertia"],
                "silhouette": result["scores"]["silhouette"],
                "davies_bouldin": result["scores"]["davies_bouldin"],
                "adjusted_rand": result["scores"]["adjusted_rand"],
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cultivar_linkage(features: tuple[str, ...], method: str = "ward", standardise: bool = True):
    frame, _ = load_cultivar_frame()
    matrix = frame[list(features)].to_numpy(dtype=float)
    if standardise:
        matrix = StandardScaler().fit_transform(matrix)
    return linkage(matrix, method=method)


# ==========================================================================
# tumour classification
# ==========================================================================
CLASSIFIERS = ("Logistic regression", "Random forest")


def _build_classifier(name: str):
    if name == "Logistic regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=SEED,
                    ),
                ),
            ]
        )
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=1,
        min_samples_leaf=1,
    )


@st.cache_resource(show_spinner=False)
def fit_classifier(name: str, test_fraction: float, use_ratios: bool) -> dict:
    """Train a tumour classifier and keep everything needed to score it."""
    features, labels = datasets.load_biopsies()
    if use_ratios:
        features = datasets.add_ratio_features(features)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_fraction,
        random_state=SEED,
        stratify=labels,
    )

    model = _build_classifier(name)
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    train_probabilities = model.predict_proba(x_train)[:, 1]

    return {
        "model": model,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "probabilities": probabilities,
        "train_probabilities": train_probabilities,
        "columns": list(features.columns),
    }


@st.cache_data(show_spinner=False)
def classifier_cross_validation(name: str, use_ratios: bool, folds: int = 5) -> pd.DataFrame:
    features, labels = datasets.load_biopsies()
    if use_ratios:
        features = datasets.add_ratio_features(features)

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores = cross_validate(
        _build_classifier(name), features, labels, cv=splitter, scoring=metrics, n_jobs=1
    )
    rows = []
    for metric in metrics:
        for fold, value in enumerate(scores[f"test_{metric}"], start=1):
            rows.append({"metric": metric, "fold": fold, "score": float(value)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def classifier_importance(name: str, test_fraction: float, use_ratios: bool, repeats: int = 10) -> pd.DataFrame:
    bundle = fit_classifier(name, test_fraction, use_ratios)
    result = permutation_importance(
        bundle["model"],
        bundle["x_test"],
        bundle["y_test"],
        scoring="roc_auc",
        n_repeats=repeats,
        random_state=SEED,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": bundle["columns"],
                "importance": result.importances_mean,
                "spread": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """Confusion counts and the metrics derived from them at one cut point."""
    predicted = (probabilities >= threshold).astype(int)
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    tn = int(((predicted == 0) & (y_true == 0)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


# ==========================================================================
# wine quality regression
# ==========================================================================
REGRESSORS = ("Baseline mean", "Ridge", "Random forest", "Gradient boosting")


def _build_regressor(name: str):
    if name == "Baseline mean":
        return DummyRegressor(strategy="mean")
    if name == "Ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    if name == "Random forest":
        # 140 trees scores within 0.001 RMSE of 220 here and fits in half the time.
        return RandomForestRegressor(
            n_estimators=140, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        )
    return GradientBoostingRegressor(
        n_estimators=140, learning_rate=0.08, max_depth=3, random_state=SEED
    )


@st.cache_resource(show_spinner="Training the regression models…")
def fit_quality_models(test_fraction: float = 0.2) -> dict:
    """Train every regressor on the same split so the comparison is fair."""
    frame, _ = datasets.load_wine_quality()
    x = frame[datasets.QUALITY_FEATURES]
    y = frame["quality"].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_fraction, random_state=SEED
    )

    fitted = {}
    rows = []
    for name in REGRESSORS:
        model = _build_regressor(name)
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        residual = y_test.to_numpy() - predicted
        rows.append(
            {
                "model": name,
                "mae": float(np.abs(residual).mean()),
                "rmse": float(np.sqrt((residual**2).mean())),
                "r2": float(1 - (residual**2).sum() / ((y_test - y_test.mean()) ** 2).sum()),
                "within_half": float((np.abs(residual) <= 0.5).mean()),
            }
        )
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return {
        "fitted": fitted,
        "results": results,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "best": results.iloc[0]["model"],
    }


@st.cache_data(show_spinner="Cross-validating four models across five folds…")
def quality_cross_validation(folds: int = 5) -> pd.DataFrame:
    frame, _ = datasets.load_wine_quality()
    x = frame[datasets.QUALITY_FEATURES]
    y = frame["quality"].astype(float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=SEED)

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    rows = []
    for name in REGRESSORS:
        scores = cross_validate(
            _build_regressor(name), x, y, cv=splitter, scoring=scoring, n_jobs=1
        )
        rmse = -scores["test_rmse"]
        mae = -scores["test_mae"]
        r2 = scores["test_r2"]
        rows.append(
            {
                "model": name,
                "rmse_mean": float(rmse.mean()),
                "rmse_std": float(rmse.std(ddof=1)),
                "mae_mean": float(mae.mean()),
                "r2_mean": float(r2.mean()),
                "r2_std": float(r2.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)


@st.cache_data(show_spinner="Measuring which measurements matter…")
def quality_importance(model_name: str, test_fraction: float = 0.2, repeats: int = 6) -> pd.DataFrame:
    bundle = fit_quality_models(test_fraction)
    result = permutation_importance(
        bundle["fitted"][model_name],
        bundle["x_test"],
        bundle["y_test"],
        scoring="neg_root_mean_squared_error",
        n_repeats=repeats,
        random_state=SEED,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": datasets.QUALITY_FEATURES,
                "importance": result.importances_mean,
                "spread": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def segment_wines(k: int) -> dict:
    """Cluster wines on chemistry alone, with the tasting score held back."""
    frame, _ = datasets.load_wine_quality()
    matrix = StandardScaler().fit_transform(frame[datasets.QUALITY_FEATURES])

    model = KMeans(n_clusters=k, n_init=20, random_state=SEED)
    labels = model.fit_predict(matrix)

    projector = PCA(n_components=2, random_state=SEED)
    coords = projector.fit_transform(matrix)

    profile = pd.DataFrame(matrix, columns=datasets.QUALITY_FEATURES)
    profile["cluster"] = labels
    profile = profile.groupby("cluster")[datasets.QUALITY_FEATURES].mean()

    summary = (
        frame.assign(cluster=labels)
        .groupby("cluster")
        .agg(
            wines=("quality", "size"),
            mean_quality=("quality", "mean"),
            mean_alcohol=("alcohol", "mean"),
            mean_volatile_acidity=("volatile_acidity", "mean"),
        )
        .reset_index()
    )

    return {
        "labels": labels,
        "coords": coords,
        "explained": projector.explained_variance_ratio_,
        "profile": profile,
        "summary": summary,
        "silhouette": float(silhouette_score(matrix, labels)),
        "quality": frame["quality"].to_numpy(),
    }


@st.cache_data(show_spinner=False)
def segment_sweep(k_values: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for k in k_values:
        result = segment_wines(k)
        rows.append({"k": k, "silhouette": result["silhouette"]})
    return pd.DataFrame(rows)


def median_contributions(model, sample: pd.Series, medians: pd.Series) -> pd.DataFrame:
    """How much each reading moves the prediction away from a median wine.

    Each feature is reset to its dataset median one at a time; the resulting
    change in the prediction is that feature's contribution for this wine.
    """
    base_frame = pd.DataFrame([sample])
    base = float(model.predict(base_frame)[0])

    rows = []
    for feature in sample.index:
        swapped = sample.copy()
        swapped[feature] = medians[feature]
        without = float(model.predict(pd.DataFrame([swapped]))[0])
        rows.append({"feature": feature, "contribution": base - without})

    table = pd.DataFrame(rows)
    table["magnitude"] = table["contribution"].abs()
    return table.sort_values("magnitude", ascending=False).reset_index(drop=True)
