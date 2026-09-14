"""Cultivar Segmentation - recovering wine groups from chemistry alone."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram

from .. import datasets, models, theme, ui

CLUSTER_TINTS = ["#C084FC", "#22D3EE", "#FBBF24", "#34D399", "#FB7185", "#818CF8", "#F0ABFC", "#A3E635"]

PLAIN_NAMES = {
    "od280_od315_of_diluted_wines": "od280 / od315",
    "nonflavanoid_phenols": "nonflavanoid phenols",
    "alcalinity_of_ash": "alcalinity of ash",
    "color_intensity": "colour intensity",
    "total_phenols": "total phenols",
    "malic_acid": "malic acid",
}


COOL = "#6C8CFF"


def _pretty(name: str) -> str:
    return PLAIN_NAMES.get(name, name.replace("_", " "))


def _heat_styles(frame: pd.DataFrame) -> pd.DataFrame:
    """Shade each cell by how far the group sits from the overall mean.

    Written by hand rather than with Styler.background_gradient, which reaches
    into matplotlib for its colour maps. That would have been the only
    dependency in the whole project pulled in for a single table, and its
    palette does not match the rest of the app in any case.
    """

    def css(value) -> str:
        if pd.isna(value):
            return ""
        value = float(value)
        strength = float(np.clip(abs(value) / 2.0, 0.0, 1.0))
        base = theme.SPECTRUM["cultivars"] if value >= 0 else COOL
        text = theme.TYPE if strength > 0.5 else theme.TYPE_DIM
        return f"background-color: {theme.fade(base, strength * 0.75)}; color: {text};"

    return pd.DataFrame(
        [[css(value) for value in row] for row in frame.to_numpy()],
        index=frame.index,
        columns=frame.columns,
    )


def render() -> None:
    colour = ui.page_header("cultivars")

    frame, truth = datasets.load_cultivars()
    all_features = list(frame.columns)

    # ------------------------------------------------------------- controls
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1.1])
        with c1:
            algorithm = st.selectbox("Algorithm", ["K-means", "Hierarchical"], index=0)
        with c2:
            k = st.slider("Groups to find (k)", 2, 8, 3)
        with c3:
            if algorithm == "Hierarchical":
                linkage_method = st.selectbox(
                    "Linkage", ["ward", "complete", "average", "single"], index=0
                )
            else:
                linkage_method = "ward"
            standardise = st.checkbox(
                "Standardise features",
                value=True,
                help="Off, the result is decided almost entirely by proline, which is "
                "measured in hundreds while every other column sits near 1.",
            )

        chosen = st.multiselect(
            "Chemical measures used",
            all_features,
            default=all_features,
            format_func=_pretty,
        )

    if len(chosen) < 2:
        ui.finding("Pick at least two measures so the samples can be placed in a space.")
        return

    features = tuple(chosen)
    result = models.cluster_cultivars(features, k, algorithm, linkage_method, standardise)
    scores = result["scores"]
    labels = result["labels"]

    # ------------------------------------------------------------- readouts
    sweep = models.cultivar_sweep(features, tuple(range(2, 9)), algorithm, linkage_method, standardise)
    best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])

    ui.readouts(
        [
            {"label": "Silhouette", "value": f"{scores['silhouette']:.3f}", "accent": True,
             "delta": f"best at k = {best_k}", "tone": "flat"},
            {"label": "Davies\u2013Bouldin", "value": f"{scores['davies_bouldin']:.3f}",
             "delta": "lower is tighter", "tone": "flat"},
            {"label": "Calinski\u2013Harabasz", "value": f"{scores['calinski_harabasz']:.0f}",
             "delta": "higher is better", "tone": "flat"},
            {"label": "Agreement with truth", "value": f"{scores['adjusted_rand']:.3f}",
             "delta": "adjusted Rand index", "tone": "flat"},
            {"label": "Measures used", "value": f"{len(chosen)} / {len(all_features)}"},
        ],
        colour,
    )

    tab_map, tab_choose, tab_profile, tab_truth = st.tabs(
        ["The grouping", "Choosing k", "What defines each group", "Checking the answer"]
    )

    # ------------------------------------------------------------------ map
    with tab_map:
        left, right = st.columns([1, 0.42])
        with left:
            ui.section(
                "Samples placed by their chemistry",
                "Thirteen measurements compressed onto the axes that carry the most variation.",
            )
        explained = result["explained"]
        coords = result["coords"]
        can_show_3d = coords.shape[1] >= 3

        with right:
            ui.spacer(0.9)
            view_3d = (
                st.toggle("Show the third component", value=False)
                if can_show_3d
                else False
            )

        if view_3d:
            figure = go.Figure()
            for cluster in range(k):
                mask = labels == cluster
                figure.add_trace(
                    go.Scatter3d(
                        x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
                        mode="markers",
                        name=f"Group {cluster + 1}",
                        marker=dict(size=4, color=CLUSTER_TINTS[cluster % len(CLUSTER_TINTS)],
                                    line=dict(width=0)),
                        text=[truth.iloc[i] for i in np.flatnonzero(mask)],
                        hovertemplate="%{text}<br>Group %{fullData.name}<extra></extra>",
                    )
                )
            def scene_axis(label: str) -> dict:
                return dict(
                    title=dict(text=label, font=dict(size=10, color=theme.TYPE_FAINT)),
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor=theme.RULE_SOFT,
                    zerolinecolor=theme.RULE,
                    showbackground=True,
                    tickfont=dict(size=9, color=theme.TYPE_FAINT, family=theme.FONT_MONO),
                )

            figure.update_layout(
                scene=dict(
                    xaxis=scene_axis(f"PC1 ({explained[0] * 100:.0f}%)"),
                    yaxis=scene_axis(f"PC2 ({explained[1] * 100:.0f}%)"),
                    zaxis=scene_axis(f"PC3 ({explained[2] * 100:.0f}%)"),
                )
            )
            st.plotly_chart(
                theme.style(figure, height=520, margin=(0, 0, 26, 0)), width="stretch"
            )
        else:
            figure = go.Figure()
            for cluster in range(k):
                mask = labels == cluster
                figure.add_trace(
                    go.Scatter(
                        x=coords[mask, 0], y=coords[mask, 1],
                        mode="markers",
                        name=f"Group {cluster + 1}",
                        marker=dict(
                            size=9,
                            color=CLUSTER_TINTS[cluster % len(CLUSTER_TINTS)],
                            opacity=0.85,
                            line=dict(width=0.6, color=theme.PLATE),
                        ),
                        text=[truth.iloc[i] for i in np.flatnonzero(mask)],
                        hovertemplate="%{text}<extra>Group " + str(cluster + 1) + "</extra>",
                    )
                )
            figure.update_xaxes(title_text=f"PC1 \u2014 {explained[0] * 100:.1f}% of variance")
            figure.update_yaxes(title_text=f"PC2 \u2014 {explained[1] * 100:.1f}% of variance")
            st.plotly_chart(theme.style(figure, height=470), width="stretch")

        ui.finding(
            f"The first two components carry "
            f"<b class='num'>{(explained[0] + explained[1]) * 100:.0f}%</b> of the total "
            f"variation, which is why the groups look cleanly separated on a flat chart. "
            f"Remember that the axes are combinations of all "
            f"<b class='num'>{len(chosen)}</b> measures, not any single one, and that the "
            f"clustering was done in the full space, not on these two axes.",
            colour,
        )

        if algorithm == "Hierarchical":
            ui.section(
                "How the tree was cut",
                f"Ward linkage merges the two groups whose union adds least to within-group "
                f"spread. The dashed line shows where cutting gives {k} groups.",
            )
            tree = models.cultivar_linkage(features, linkage_method, standardise)
            plot = dendrogram(tree, no_plot=True, no_labels=True)
            merge_heights = np.sort(tree[:, 2])
            cut = (
                (merge_heights[-k] + merge_heights[-k + 1]) / 2
                if k > 1 and k <= len(merge_heights)
                else merge_heights[-1]
            )

            figure = go.Figure()
            for xs, ys in zip(plot["icoord"], plot["dcoord"]):
                figure.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines",
                        line=dict(
                            color=colour if max(ys) < cut else theme.rgba(theme.TYPE_FAINT, 0.55),
                            width=1.1,
                        ),
                        hoverinfo="skip", showlegend=False,
                    )
                )
            figure.add_hline(
                y=cut, line=dict(color=theme.NEGATIVE, width=1.4, dash="dash"),
                annotation_text=f"cut for k = {k}",
                annotation_font=dict(size=10, color=theme.TYPE_FAINT),
            )
            figure.update_xaxes(showticklabels=False, title_text="Individual wines")
            figure.update_yaxes(title_text="Merge distance")
            st.plotly_chart(theme.style(figure, height=330, legend=False), width="stretch")

    # --------------------------------------------------------------- choose
    with tab_choose:
        ui.section(
            "Two ways of asking how many groups there are",
            "Inertia always falls as k rises, so it can only be read for a bend. Silhouette "
            "has a genuine maximum and is the more useful of the two.",
        )

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=sweep["k"], y=sweep["inertia"], mode="lines+markers", name="Inertia",
                line=dict(color=theme.rgba(theme.TYPE_FAINT, 0.9), width=2),
                marker=dict(size=7),
                hovertemplate="k = %{x}<br>inertia %{y:.0f}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=sweep["k"], y=sweep["silhouette"], mode="lines+markers", name="Silhouette",
                line=dict(color=colour, width=2.4), marker=dict(size=8), yaxis="y2",
                hovertemplate="k = %{x}<br>silhouette %{y:.3f}<extra></extra>",
            )
        )
        figure.add_vline(
            x=k, line=dict(color=theme.NEGATIVE, width=1.4, dash="dash"),
            annotation_text="your k", annotation_font=dict(size=10, color=theme.TYPE_FAINT),
        )
        figure.update_layout(
            yaxis=dict(title="Inertia"),
            yaxis2=dict(
                title="Silhouette", overlaying="y", side="right",
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(size=10, color=theme.TYPE_FAINT, family=theme.FONT_MONO),
                title_font=dict(size=11, color=theme.TYPE_FAINT),
            ),
            xaxis=dict(title="Number of groups (k)", dtick=1),
        )
        st.plotly_chart(theme.style(figure, height=390), width="stretch")

        ui.section(
            "How comfortably each sample sits in its group",
            "One bar per wine. Wide, even blocks mean well-formed groups; bars dipping below "
            "zero mark wines that sit closer to a neighbouring group than their own.",
        )

        per_sample = result["silhouette_samples"]
        figure = go.Figure()
        offset = 0
        ticks, tick_labels = [], []
        for cluster in range(k):
            values = np.sort(per_sample[labels == cluster])
            if len(values) == 0:
                continue
            positions = np.arange(offset, offset + len(values))
            figure.add_trace(
                go.Bar(
                    x=values, y=positions, orientation="h",
                    marker_color=CLUSTER_TINTS[cluster % len(CLUSTER_TINTS)],
                    marker_line_width=0, name=f"Group {cluster + 1}",
                    hovertemplate="silhouette %{x:.3f}<extra>Group "
                    + str(cluster + 1) + "</extra>",
                )
            )
            ticks.append(offset + len(values) / 2)
            tick_labels.append(f"Group {cluster + 1}")
            offset += len(values) + 6

        figure.add_vline(
            x=float(np.mean(per_sample)),
            line=dict(color=theme.NEGATIVE, width=1.4, dash="dash"),
            annotation_text="mean", annotation_font=dict(size=10, color=theme.TYPE_FAINT),
        )
        figure.update_layout(
            bargap=0.0,
            yaxis=dict(tickmode="array", tickvals=ticks, ticktext=tick_labels, showgrid=False),
        )
        figure.update_xaxes(title_text="Silhouette value")
        st.plotly_chart(theme.style(figure, height=430, legend=False), width="stretch")

        negative = int((per_sample < 0).sum())
        ui.finding(
            f"At k = <b class='num'>{k}</b> the mean silhouette is "
            f"<b class='num'>{scores['silhouette']:.3f}</b> and "
            f"<b class='num'>{negative}</b> of <b class='num'>{len(per_sample)}</b> wines have "
            f"a negative score. Silhouette peaks at k = <b class='num'>{best_k}</b> for this "
            f"choice of measures. A score near 0.3 is normal for real chemistry: these are "
            f"overlapping populations, not islands.",
            colour,
        )

    # -------------------------------------------------------------- profile
    with tab_profile:
        ui.section(
            "Group fingerprints",
            "Each group's average, in standard deviations from the overall mean. Zero is "
            "typical; the further out a spoke reaches, the more that measure defines the group.",
        )

        profile = result["profile"]
        radar_features = (
            profile.abs().mean(axis=0).sort_values(ascending=False).head(8).index.tolist()
        )

        figure = go.Figure()
        for cluster in profile.index:
            values = profile.loc[cluster, radar_features].tolist()
            figure.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=[_pretty(f) for f in radar_features] + [_pretty(radar_features[0])],
                    fill="toself",
                    name=f"Group {int(cluster) + 1}",
                    line=dict(color=CLUSTER_TINTS[int(cluster) % len(CLUSTER_TINTS)], width=2),
                    fillcolor=theme.rgba(CLUSTER_TINTS[int(cluster) % len(CLUSTER_TINTS)], 0.12),
                    hovertemplate="%{theta}<br>%{r:+.2f}\u03c3<extra>Group "
                    + str(int(cluster) + 1) + "</extra>",
                )
            )
        figure.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    gridcolor=theme.RULE_SOFT, linecolor=theme.RULE_SOFT,
                    tickfont=dict(size=9, color=theme.TYPE_FAINT, family=theme.FONT_MONO),
                ),
                angularaxis=dict(
                    gridcolor=theme.RULE_SOFT, linecolor=theme.RULE_SOFT,
                    tickfont=dict(size=10, color=theme.TYPE_DIM),
                ),
            )
        )
        st.plotly_chart(
            theme.style(figure, height=520, margin=(30, 30, 30, 30)), width="stretch"
        )

        ui.section("The same thing as a table", "All measures, all groups, in standard deviations.")
        display = profile.copy()
        display.columns = [_pretty(c) for c in display.columns]
        display.index = [f"Group {int(i) + 1}" for i in display.index]
        st.dataframe(
            display.round(2).style.apply(_heat_styles, axis=None),
            width="stretch",
            height=min(60 + 36 * len(display), 360),
        )

        headline = []
        for cluster in profile.index:
            row = profile.loc[cluster]
            strongest = row.abs().idxmax()
            direction = "high" if row[strongest] > 0 else "low"
            headline.append(
                f"Group {int(cluster) + 1} is defined most by {direction} "
                f"{_pretty(strongest)} ({row[strongest]:+.2f}\u03c3)"
            )
        ui.finding(
            "Reading the strongest spoke for each group: " + "; ".join(headline) + ". "
            "These are descriptions, not causes. The algorithm had no idea what any column "
            "meant when it drew the boundaries.",
            colour,
        )

    # ---------------------------------------------------------------- truth
    with tab_truth:
        ui.section(
            "Marking the unsupervised work",
            "These wines carry known cultivar labels that were withheld from the algorithm. "
            "Holding them back and comparing afterwards is the only honest way to judge "
            "clustering on real data.",
        )

        crosstab = pd.crosstab(
            pd.Series([f"Group {c + 1}" for c in labels], name="Found"),
            pd.Series(result["truth"], name="Actual cultivar"),
        )

        heat = go.Figure(
            go.Heatmap(
                z=crosstab.to_numpy(),
                x=list(crosstab.columns),
                y=list(crosstab.index),
                colorscale=theme.sequential(colour),
                text=crosstab.to_numpy(),
                texttemplate="%{text}",
                textfont=dict(size=13, family=theme.FONT_MONO, color=theme.TYPE),
                hovertemplate="%{y} \u00d7 %{x}<br>%{z} wines<extra></extra>",
                showscale=False,
            )
        )
        st.plotly_chart(
            theme.style(heat, height=90 + 62 * len(crosstab), legend=False), width="stretch"
        )

        pure = int(crosstab.max(axis=1).sum())
        total = int(crosstab.to_numpy().sum())
        ui.finding(
            f"If each found group is assigned to the cultivar it contains most of, "
            f"<b class='num'>{pure}</b> of <b class='num'>{total}</b> wines "
            f"(<b class='num'>{pure / total * 100:.1f}%</b>) land in the right place, and the "
            f"adjusted Rand index is <b class='num'>{scores['adjusted_rand']:.3f}</b>. "
            f"That index corrects for agreement that would happen by chance, so unlike raw "
            f"accuracy it cannot be inflated by simply asking for more groups.",
            colour,
        )

        ui.section(
            "Why standardising matters",
            "The same clustering, run with and without putting every measure on a common scale.",
        )
        comparison = []
        for flag, label in [(True, "Standardised"), (False, "Raw units")]:
            outcome = models.cluster_cultivars(features, k, algorithm, linkage_method, flag)
            comparison.append(
                {
                    "Scaling": label,
                    "Silhouette": round(outcome["scores"]["silhouette"], 3),
                    "Agreement with truth": round(outcome["scores"]["adjusted_rand"], 3),
                }
            )
        st.dataframe(pd.DataFrame(comparison), width="stretch", hide_index=True)

        ui.caveat(
            "<b>Scale decides the answer.</b> Proline is measured in the hundreds while hue "
            "sits near one. Without standardising, the distance between any two wines is "
            "almost entirely the difference in their proline, and every other measurement is "
            "effectively ignored. Any distance-based method inherits whatever units its "
            "inputs happen to arrive in."
        )
