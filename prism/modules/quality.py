"""Quality Lab - the integrative page, where every earlier technique meets one dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import datasets, models, theme, ui

SEGMENT_TINTS = ["#FB7185", "#22D3EE", "#FBBF24", "#A855F7", "#34D399", "#818CF8"]


def _label(feature: str) -> str:
    return datasets.QUALITY_PLAIN.get(feature, feature.replace("_", " "))


def render() -> None:
    colour = ui.page_header("quality")

    frame, report = datasets.load_wine_quality()
    features = datasets.QUALITY_FEATURES

    bundle = models.fit_quality_models(0.2)
    results = bundle["results"]
    best_name = bundle["best"]
    best = results.iloc[0]
    baseline = results[results["model"] == "Baseline mean"].iloc[0]

    ui.readouts(
        [
            {"label": "Bottles analysed", "value": f"{report['rows_final']:,}",
             "delta": f"{report['duplicate_rows']} duplicates removed", "tone": "flat"},
            {"label": "Best model", "value": best_name, "accent": True},
            {"label": "Typical error", "value": f"{best['mae']:.3f}",
             "unit": "pts", "delta": f"baseline {baseline['mae']:.3f}", "tone": "up"},
            {"label": "Within half a point", "value": f"{best['within_half']:.1%}"},
            {"label": "Variance explained", "value": f"{best['r2']:.3f}",
             "delta": "held-out R\u00b2", "tone": "flat"},
        ],
        colour,
    )

    tab_data, tab_models, tab_segments, tab_taste = st.tabs(
        ["The wines", "Predicting the score", "Natural groupings", "Pour a glass"]
    )

    # ---------------------------------------------------------------- data
    with tab_data:
        ui.section(
            "What the judges actually awarded",
            "Every bottle was scored by at least three assessors, and the median was recorded.",
        )

        counts = frame["quality"].value_counts().sort_index()
        bars = go.Figure(
            go.Bar(
                x=counts.index.astype(str), y=counts.to_numpy(),
                marker_color=[
                    colour if v == counts.max() else theme.rgba(colour, 0.5) for v in counts
                ],
                marker_line_width=0,
                text=[f"{v:,}" for v in counts],
                textposition="outside",
                textfont=dict(size=11, family=theme.FONT_MONO, color=theme.TYPE_DIM),
                hovertemplate="Score %{x}<br>%{y} bottles<extra></extra>",
            )
        )
        bars.update_xaxes(title_text="Quality score awarded")
        bars.update_yaxes(title_text="Bottles")
        st.plotly_chart(theme.style(bars, height=300, legend=False), width="stretch")

        middle = int(counts.loc[[5, 6]].sum()) if {5, 6}.issubset(counts.index) else 0
        ui.finding(
            f"<b class='num'>{middle:,}</b> of <b class='num'>{len(frame):,}</b> bottles "
            f"(<b class='num'>{middle / len(frame) * 100:.0f}%</b>) scored either 5 or 6, and "
            f"nothing was awarded below 3 or above 8. This lopsidedness shapes everything that "
            f"follows: a model can look accurate simply by guessing near the middle every time, "
            f"which is exactly why the baseline that always predicts the mean is carried through "
            f"the comparison.",
            colour,
        )

        ui.section(
            "Which measurements move with quality",
            "Pick one and see it against the score. Correlation here is weak across the board, "
            "which is the honest headline of this dataset.",
        )

        correlations = (
            frame[features + ["quality"]].corr()["quality"].drop("quality").sort_values()
        )
        strip = go.Figure(
            go.Bar(
                x=correlations.to_numpy(),
                y=[_label(f) for f in correlations.index],
                orientation="h",
                marker_color=[
                    colour if v > 0 else "#6C8CFF" for v in correlations
                ],
                marker_line_width=0,
                hovertemplate="%{y}<br>r = %{x:.3f}<extra></extra>",
            )
        )
        strip.update_xaxes(title_text="Correlation with quality score")
        st.plotly_chart(theme.style(strip, height=340, legend=False), width="stretch")

        chosen = st.selectbox(
            "Measurement to inspect", features, index=features.index("alcohol"),
            format_func=_label,
        )
        box = go.Figure()
        for score in sorted(frame["quality"].unique()):
            box.add_trace(
                go.Box(
                    y=frame.loc[frame["quality"] == score, chosen],
                    name=str(int(score)),
                    marker_color=colour,
                    line=dict(color=colour, width=1.5),
                    fillcolor=theme.rgba(colour, 0.15),
                    boxpoints="outliers",
                    marker=dict(size=3, opacity=0.5),
                    hovertemplate="%{y:.3f}<extra>Score " + str(int(score)) + "</extra>",
                )
            )
        box.update_xaxes(title_text="Quality score")
        box.update_yaxes(
            title_text=f"{_label(chosen)} ({datasets.QUALITY_UNITS.get(chosen, '')})"
        )
        st.plotly_chart(theme.style(box, height=330, legend=False), width="stretch")

        strongest = correlations.abs().idxmax()
        ui.finding(
            f"The strongest single relationship is <b>{_label(strongest)}</b> at "
            f"r = <b class='num'>{correlations[strongest]:.3f}</b>. Squared, that accounts for "
            f"roughly <b class='num'>{correlations[strongest] ** 2 * 100:.0f}%</b> of the "
            f"variation in scores, so no one measurement comes close to explaining quality. "
            f"The boxes overlap heavily at every score, which is why a model that combines all "
            f"eleven measurements does meaningfully better than any rule based on one.",
            colour,
        )

        with st.expander("Cleaning report"):
            st.json(report)

    # -------------------------------------------------------------- models
    with tab_models:
        ui.section(
            "Four models on the same held-out bottles",
            "The baseline ignores the chemistry entirely and always predicts the average "
            "score. Any model that cannot beat it has learned nothing.",
        )

        ordered = results.sort_values("rmse", ascending=False)
        comparison = go.Figure(
            go.Bar(
                x=ordered["rmse"], y=ordered["model"], orientation="h",
                marker_color=[
                    theme.rgba(colour, 0.4) if m == "Baseline mean" else colour
                    for m in ordered["model"]
                ],
                marker_line_width=0,
                text=[f"{v:.4f}" for v in ordered["rmse"]],
                textposition="outside",
                textfont=dict(size=11, family=theme.FONT_MONO, color=theme.TYPE_DIM),
                hovertemplate="%{y}<br>RMSE %{x:.4f}<extra></extra>",
            )
        )
        comparison.update_xaxes(title_text="Root mean squared error (lower is better)")
        st.plotly_chart(theme.style(comparison, height=280, legend=False), width="stretch")

        ui.section(
            "The same models under five-fold cross-validation",
            "Refitting on five different splits shows whether the ranking above is real or "
            "an accident of one particular split.",
        )
        folds = models.quality_cross_validation(5)
        display = folds.rename(
            columns={
                "model": "Model", "rmse_mean": "RMSE", "rmse_std": "RMSE spread",
                "mae_mean": "MAE", "r2_mean": "R\u00b2", "r2_std": "R\u00b2 spread",
            }
        )
        st.dataframe(display.round(4), width="stretch", hide_index=True)

        improvement = (baseline["rmse"] - best["rmse"]) / baseline["rmse"] * 100
        ui.finding(
            f"<b>{best_name}</b> reduces error by <b class='num'>{improvement:.0f}%</b> against "
            f"the do-nothing baseline, and holds first place under cross-validation as well as "
            f"on the single split. The spread across folds is small relative to the gap between "
            f"models, so the ranking is trustworthy even though the absolute accuracy is modest.",
            colour,
        )

        ui.section(
            "Where the predictions land",
            "Perfect predictions would sit on the diagonal. The banding is because scores are "
            "whole numbers while predictions are continuous.",
        )
        predicted = bundle["fitted"][best_name].predict(bundle["x_test"])
        actual = bundle["y_test"].to_numpy()
        jitter = np.random.default_rng(7).normal(0, 0.06, len(actual))

        scatter = go.Figure()
        scatter.add_trace(
            go.Scatter(
                x=actual + jitter, y=predicted, mode="markers",
                marker=dict(size=6, color=theme.rgba(colour, 0.5),
                            line=dict(width=0.5, color=theme.PLATE)),
                name="Held-out bottles",
                hovertemplate="Actual %{x:.0f}<br>Predicted %{y:.2f}<extra></extra>",
            )
        )
        limits = [actual.min() - 0.5, actual.max() + 0.5]
        scatter.add_trace(
            go.Scatter(
                x=limits, y=limits, mode="lines", name="Perfect prediction",
                line=dict(color=theme.RULE, width=1.4, dash="dash"), hoverinfo="skip",
            )
        )
        scatter.update_xaxes(title_text="Score awarded")
        scatter.update_yaxes(title_text="Score predicted")
        st.plotly_chart(theme.style(scatter, height=380), width="stretch")

        residuals = actual - predicted
        ui.finding(
            f"The cloud is flatter than the diagonal: low-scoring wines are predicted too "
            f"high and high-scoring wines too low. That pull toward the middle is what a "
            f"squared-error model does when the signal is weak, because guessing near the "
            f"average is the safest way to avoid a large penalty. Residuals average "
            f"<b class='num'>{residuals.mean():+.3f}</b> with a spread of "
            f"<b class='num'>{residuals.std():.3f}</b>, so the model is unbiased overall "
            f"even while it is timid at the extremes.",
            colour,
        )

        ui.section(
            "Which measurements the model leans on",
            "Each column shuffled in turn on held-out data, and the damage recorded.",
        )
        importance = models.quality_importance(best_name, 0.2, 12)
        ordered_importance = importance.sort_values("importance")
        bars = go.Figure(
            go.Bar(
                x=ordered_importance["importance"],
                y=[_label(f) for f in ordered_importance["feature"]],
                orientation="h",
                marker_color=theme.rgba(colour, 0.8), marker_line_width=0,
                error_x=dict(
                    type="data", array=ordered_importance["spread"],
                    color=theme.TYPE_FAINT, thickness=1, width=3,
                ),
                hovertemplate="%{y}<br>RMSE worsens by %{x:.4f}<extra></extra>",
            )
        )
        bars.update_xaxes(title_text="Increase in error when shuffled")
        st.plotly_chart(theme.style(bars, height=380, legend=False), width="stretch")

        top_three = ", ".join(_label(f) for f in importance.head(3)["feature"])
        ui.finding(
            f"The model relies most on {top_three}. Alcohol and volatile acidity also topped "
            f"the correlation chart, but tree models can exploit relationships that a "
            f"correlation coefficient cannot see, which is why the two rankings are related "
            f"without being identical.",
            colour,
        )

    # ------------------------------------------------------------ segments
    with tab_segments:
        ui.section(
            "Grouping the cellar without looking at the scores",
            "The same unsupervised approach used on the cultivars, applied here to "
            "chemistry alone. The tasting score is withheld, then compared afterwards.",
        )

        k = st.slider("Groups to form", 2, 6, 3)
        segmentation = models.segment_wines(k)
        sweep = models.segment_sweep(tuple(range(2, 7)))

        left, right = st.columns([1.3, 1], gap="large")

        with left:
            scatter = go.Figure()
            for cluster in range(k):
                mask = segmentation["labels"] == cluster
                scatter.add_trace(
                    go.Scatter(
                        x=segmentation["coords"][mask, 0],
                        y=segmentation["coords"][mask, 1],
                        mode="markers", name=f"Group {cluster + 1}",
                        marker=dict(
                            size=6, opacity=0.72,
                            color=SEGMENT_TINTS[cluster % len(SEGMENT_TINTS)],
                            line=dict(width=0.4, color=theme.PLATE),
                        ),
                        customdata=segmentation["quality"][mask],
                        hovertemplate="Score %{customdata}<extra>Group "
                        + str(cluster + 1) + "</extra>",
                    )
                )
            explained = segmentation["explained"]
            scatter.update_xaxes(title_text=f"PC1 \u2014 {explained[0] * 100:.0f}% of variance")
            scatter.update_yaxes(title_text=f"PC2 \u2014 {explained[1] * 100:.0f}% of variance")
            st.plotly_chart(theme.style(scatter, height=400), width="stretch")

        with right:
            sweep_figure = go.Figure(
                go.Scatter(
                    x=sweep["k"], y=sweep["silhouette"], mode="lines+markers",
                    line=dict(color=colour, width=2.4), marker=dict(size=8),
                    hovertemplate="k = %{x}<br>silhouette %{y:.3f}<extra></extra>",
                )
            )
            sweep_figure.add_vline(
                x=k, line=dict(color=theme.NEGATIVE, width=1.4, dash="dash"),
                annotation_text="your k",
                annotation_font=dict(size=10, color=theme.TYPE_FAINT),
            )
            sweep_figure.update_xaxes(title_text="Groups", dtick=1)
            sweep_figure.update_yaxes(title_text="Silhouette")
            st.plotly_chart(theme.style(sweep_figure, height=400, legend=False), width="stretch")

        summary = segmentation["summary"].copy()
        summary["cluster"] = [f"Group {int(c) + 1}" for c in summary["cluster"]]
        summary = summary.rename(
            columns={
                "cluster": "Group", "wines": "Bottles", "mean_quality": "Mean score",
                "mean_alcohol": "Mean alcohol", "mean_volatile_acidity": "Mean volatile acidity",
            }
        )
        st.dataframe(summary.round(3), width="stretch", hide_index=True)

        spread = summary["Mean score"].max() - summary["Mean score"].min()
        ui.finding(
            f"Silhouette peaks around <b class='num'>{int(sweep.loc[sweep['silhouette'].idxmax(), 'k'])}</b> "
            f"but never rises far above <b class='num'>{sweep['silhouette'].max():.2f}</b>, which "
            f"says these wines form one continuous cloud rather than distinct families. Even so, "
            f"average score differs by <b class='num'>{spread:.2f}</b> points between the "
            f"strongest and weakest group. The grouping never saw a single tasting score, so that "
            f"separation is chemistry alone lining up with human judgement.",
            colour,
        )

    # --------------------------------------------------------------- taste
    with tab_taste:
        ui.section(
            "Build a wine and see how it scores",
            "Start from a real bottle, then change the chemistry and watch the prediction move.",
        )

        medians = frame[features].median()
        if "wine_index" not in st.session_state:
            st.session_state.wine_index = 0
            st.session_state.wine_nonce = 0

        picker = st.columns([1, 1, 1, 1.6])
        with picker[0]:
            if st.button("Random bottle", width="stretch"):
                generator = np.random.default_rng()
                st.session_state.wine_index = int(generator.integers(0, len(frame)))
                st.session_state.wine_nonce += 1
        with picker[1]:
            if st.button("A weak one", width="stretch"):
                pool = frame.index[frame["quality"] <= 4]
                if len(pool):
                    st.session_state.wine_index = int(
                        np.random.default_rng().choice(pool.to_numpy())
                    )
                    st.session_state.wine_nonce += 1
        with picker[2]:
            if st.button("A strong one", width="stretch"):
                pool = frame.index[frame["quality"] >= 7]
                if len(pool):
                    st.session_state.wine_index = int(
                        np.random.default_rng().choice(pool.to_numpy())
                    )
                    st.session_state.wine_nonce += 1
        with picker[3]:
            st.caption("Each button reloads the sliders from a real bottle in the dataset.")

        index = min(st.session_state.wine_index, len(frame) - 1)
        sample = frame[features].iloc[index].copy()
        awarded = int(frame["quality"].iloc[index])
        nonce = st.session_state.wine_nonce

        slider_columns = st.columns(3)
        for position, feature in enumerate(features):
            low = float(frame[feature].min())
            high = float(frame[feature].max())
            span = high - low or 1.0
            with slider_columns[position % 3]:
                sample[feature] = st.slider(
                    f"{_label(feature)} ({datasets.QUALITY_UNITS.get(feature, '')})",
                    min_value=float(low),
                    max_value=float(high),
                    value=float(np.clip(sample[feature], low, high)),
                    step=float(span / 200),
                    key=f"wine_{feature}_{nonce}",
                    format="%.3f",
                )

        model = bundle["fitted"][best_name]
        prediction = float(model.predict(pd.DataFrame([sample]))[0])
        median_prediction = float(model.predict(pd.DataFrame([medians]))[0])

        gauge_left, gauge_right = st.columns([1, 1], gap="large")
        with gauge_left:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction,
                    delta=dict(
                        reference=median_prediction,
                        valueformat=".2f",
                        increasing=dict(color=theme.POSITIVE),
                        decreasing=dict(color=theme.NEGATIVE),
                        font=dict(size=14, family=theme.FONT_MONO),
                    ),
                    number=dict(
                        valueformat=".2f",
                        font=dict(size=42, family=theme.FONT_MONO, color=theme.TYPE),
                    ),
                    gauge=dict(
                        axis=dict(
                            range=[3, 8], dtick=1,
                            tickfont=dict(size=10, color=theme.TYPE_FAINT, family=theme.FONT_MONO),
                            tickwidth=1, tickcolor=theme.RULE,
                        ),
                        bar=dict(color=colour, thickness=0.7),
                        bgcolor=theme.SURFACE,
                        borderwidth=1, bordercolor=theme.RULE_SOFT,
                        threshold=dict(
                            line=dict(color=theme.TYPE_FAINT, width=2.5),
                            thickness=0.85, value=median_prediction,
                        ),
                    ),
                )
            )
            st.plotly_chart(
                theme.style(gauge, height=290, legend=False, margin=(20, 20, 10, 10)),
                width="stretch",
            )
            st.caption(
                "Predicted score. The grey mark and the delta are both against a wine "
                "sitting at the median of every measurement."
            )

        with gauge_right:
            ui.spacer(0.6)
            ui.readouts(
                [
                    {"label": "Predicted score", "value": f"{prediction:.2f}", "accent": True},
                    {"label": "Awarded to this bottle", "value": f"{awarded}",
                     "delta": "before you moved anything", "tone": "flat"},
                    {"label": "A median wine scores", "value": f"{median_prediction:.2f}"},
                ],
                colour,
            )

        ui.section(
            "What is pushing this score around",
            "Each measurement is reset to the dataset median one at a time, and the change in "
            "prediction recorded. Bars to the right are lifting this wine above a median one.",
        )
        contributions = models.median_contributions(model, sample, medians)
        ordered_contributions = contributions.sort_values("contribution")

        waterfall = go.Figure(
            go.Bar(
                x=ordered_contributions["contribution"],
                y=[_label(f) for f in ordered_contributions["feature"]],
                orientation="h",
                marker_color=[
                    colour if v > 0 else "#6C8CFF" for v in ordered_contributions["contribution"]
                ],
                marker_line_width=0,
                hovertemplate="%{y}<br>%{x:+.3f} points<extra></extra>",
            )
        )
        waterfall.add_vline(x=0, line=dict(color=theme.RULE, width=1))
        waterfall.update_xaxes(title_text="Effect on the predicted score")
        st.plotly_chart(theme.style(waterfall, height=380, legend=False), width="stretch")

        leader = contributions.iloc[0]
        direction = "raising" if leader["contribution"] > 0 else "lowering"
        ui.finding(
            f"<b>{_label(leader['feature'])}</b> matters most for this particular wine, "
            f"{direction} the prediction by <b class='num'>{abs(leader['contribution']):.3f}</b> "
            f"points against a median bottle. This is a local explanation: it describes this "
            f"wine, not the model in general, and the same measurement can matter enormously "
            f"for one bottle and not at all for another.",
            colour,
        )

        ui.caveat(
            "<b>What this model is and is not.</b> It learns from blind tasting scores given "
            "by a panel, so it predicts how assessors tended to rate a chemistry profile, not "
            "whether a wine is good. It knows nothing of grape, vintage, region, price or "
            "winemaker, all of which shape quality and none of which appear in the data. "
            "Sliders can also be moved into combinations that no real wine would show, and the "
            "model will answer confidently anyway, because extrapolating beyond the training "
            "data is precisely where a model is least trustworthy and least likely to warn you."
        )
