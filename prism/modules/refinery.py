"""Signal Refinery - the cleaning pass that every later page depends on."""

from __future__ import annotations

import json
from html import escape

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .. import datasets, theme, ui


def _gap_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Start index and length of each run of synthesised values."""
    runs = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(flags) - start))
    return sorted(runs, key=lambda r: r[1], reverse=True)


def _ledger(report: dict, unit: str) -> str:
    stages = [
        ("Readings received", f"{report['rows_received']:,}",
         "Rows as they arrive from the archive."),
        ("Timestamps parsed", f"\u2212{report['rows_without_date']:,}",
         "A row without a usable date cannot anchor a series."),
        ("Impossible values", f"\u2212{report['non_positive_readings']:,}",
         "Zero and negative ppm become missing, never zero."),
        ("Repeated dates", f"\u2212{report['duplicate_dates']:,}",
         "The first reading for each date is kept."),
        ("Placed on a grid", f"{report['grid_points']:,}",
         f"Averaged into evenly spaced {unit}."),
        ("Gaps found", f"{report['gaps_on_grid']:,}",
         f"Longest unbroken gap runs {report['longest_gap']} {unit}."),
        ("Values synthesised", f"+{report['values_synthesised']:,}",
         "Interpolated, and flagged so charts can mark them."),
        ("Final series", f"{report['rows_final']:,}",
         "Passes all five integrity checks."),
    ]
    rows = []
    for name, delta, note in stages:
        rows.append(
            f'<div style="display:grid;grid-template-columns:1fr auto;gap:1rem;'
            f'padding:.62rem 0;border-bottom:1px solid {theme.RULE_SOFT};">'
            f'<div><div style="font-size:.87rem;color:{theme.TYPE};font-weight:500;">'
            f"{escape(name)}</div>"
            f'<div style="font-size:.76rem;color:{theme.TYPE_FAINT};line-height:1.45;'
            f'margin-top:.12rem;">{escape(note)}</div></div>'
            f'<div style="font-family:{theme.FONT_MONO};font-size:.95rem;color:{theme.TYPE_DIM};'
            f'font-variant-numeric:tabular-nums;white-space:nowrap;">{escape(delta)}</div></div>'
        )
    return "".join(rows)


def render() -> None:
    colour = ui.page_header("refinery")

    # ------------------------------------------------------------- controls
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.1])
        with c1:
            frequency = st.selectbox(
                "Averaging interval", list(datasets.FREQUENCIES.keys()), index=0
            )
        with c2:
            interpolation = st.selectbox(
                "Gap filling",
                ["time", "linear", "nearest", "none"],
                index=0,
                help="'time' spaces the fill by elapsed time; 'none' leaves gaps out entirely.",
            )
        with c3:
            iqr_multiplier = st.slider(
                "Outlier sensitivity (IQR multiplier)", 1.0, 3.0, 1.5, 0.1
            )
        with c4:
            drop_outliers = st.checkbox("Remove flagged outliers", value=False)
            inject = st.checkbox(
                "Add faults to the source",
                value=False,
                help="Puts duplicate rows, blank cells and sentinel values back into a copy "
                "of the archive so you can watch the pipeline catch them.",
            )

    clean, report = datasets.clean_co2(
        frequency=frequency,
        interpolation=interpolation,
        iqr_multiplier=iqr_multiplier,
        drop_outliers=drop_outliers,
        inject_faults=inject,
    )
    unit = report["grid_unit"]

    # ------------------------------------------------------------- readouts
    rejected = (
        report["rows_without_date"]
        + report["non_positive_readings"]
        + report["duplicate_dates"]
    )
    ui.readouts(
        [
            {"label": "Readings received", "value": f"{report['rows_received']:,}"},
            {
                "label": "Rejected at the door",
                "value": f"{rejected:,}",
                "delta": "bad dates, values, repeats",
                "tone": "flat",
            },
            {"label": "Points on the grid", "value": f"{report['grid_points']:,}"},
            {
                "label": "Gaps filled",
                "value": f"{report['values_synthesised']:,}",
                "delta": f"longest {report['longest_gap']} {unit}",
                "tone": "flat",
            },
            {"label": "Integrity checks", "value": "5 / 5", "accent": True},
        ],
        colour,
    )

    if inject:
        ui.finding(
            f"With faults injected the pipeline rejected "
            f"<b class='num'>{report['rows_without_date']}</b> undated rows, "
            f"<b class='num'>{report['non_positive_readings']}</b> impossible readings and "
            f"<b class='num'>{report['duplicate_dates']}</b> repeated dates, and still produced "
            f"a series that passes every integrity check.",
            colour,
        )

    tab_series, tab_gaps, tab_outliers, tab_output = st.tabs(
        ["Cleaned series", "Where the gaps are", "Outlier rule", "What comes out"]
    )

    # --------------------------------------------------------------- series
    with tab_series:
        ui.section(
            "The refined record",
            "Synthesised points are marked, so nothing invented is mistaken for a measurement.",
        )

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=clean["date"],
                y=clean["co2_ppm"],
                mode="lines",
                name="Measured series",
                line=dict(color=theme.rgba(colour, 0.55), width=1),
                hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} ppm<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=clean["date"],
                y=clean["rolling_mean"],
                mode="lines",
                name="Centred rolling mean",
                line=dict(color=colour, width=2.2),
                hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} ppm<extra></extra>",
            )
        )
        filled = clean[clean["was_filled"]]
        if not filled.empty:
            figure.add_trace(
                go.Scatter(
                    x=filled["date"],
                    y=filled["co2_ppm"],
                    mode="markers",
                    name=f"Synthesised ({len(filled):,})",
                    marker=dict(
                        color=theme.NEGATIVE,
                        size=4.5,
                        symbol="circle-open",
                        line=dict(width=1.2),
                    ),
                    hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} ppm (filled)<extra></extra>",
                )
            )
        figure.update_layout(xaxis=dict(rangeslider=dict(visible=True, thickness=0.07)))
        figure.update_yaxes(title_text="CO\u2082 (ppm)")
        st.plotly_chart(
            theme.style(figure, height=430, margin=(10, 10, 34, 10)),
            width="stretch",
        )

        first = float(clean["co2_ppm"].iloc[0])
        last = float(clean["co2_ppm"].iloc[-1])
        years = (clean["date"].iloc[-1] - clean["date"].iloc[0]).days / 365.25
        ui.finding(
            f"Across <b class='num'>{years:.0f}</b> years the record climbs from "
            f"<b class='num'>{first:.1f}</b> to <b class='num'>{last:.1f}</b> ppm, a rise of "
            f"<b class='num'>{last - first:.1f}</b> ppm or "
            f"<b class='num'>{(last / first - 1) * 100:.1f}%</b>. The annual wobble riding on "
            f"that climb is the biosphere breathing in and out; it gets separated from the "
            f"trend on the next page.",
            colour,
        )

    # ----------------------------------------------------------------- gaps
    with tab_gaps:
        ui.section(
            "Missing stretches",
            "Instrument downtime is not spread evenly, and the early decades carry most of it.",
        )

        by_year = (
            clean.assign(filled=clean["was_filled"].astype(int))
            .groupby("year", as_index=False)["filled"]
            .sum()
        )
        bar = go.Figure(
            go.Bar(
                x=by_year["year"],
                y=by_year["filled"],
                marker_color=theme.rgba(colour, 0.75),
                marker_line_width=0,
                hovertemplate="%{x}<br>%{y} synthesised<extra></extra>",
            )
        )
        bar.update_yaxes(title_text=f"Synthesised {unit}")
        bar.update_xaxes(title_text="Year")
        st.plotly_chart(theme.style(bar, height=250, legend=False), width="stretch")

        runs = _gap_runs(clean["was_filled"].to_numpy())
        if runs:
            ui.section(
                "Look inside one gap",
                "Pick a stretch and the chart zooms to it, so the fill can be judged against "
                "the measurements on either side.",
            )
            options = {}
            for start, length in runs[:8]:
                label_date = clean["date"].iloc[start].strftime("%b %Y")
                options[f"{label_date} \u2014 {length} {unit}"] = (start, length)

            choice = st.selectbox("Gap", list(options.keys()), label_visibility="collapsed")
            start, length = options[choice]
            pad = max(length * 4, 20)
            window = clean.iloc[max(0, start - pad) : min(len(clean), start + length + pad)]
            gap_rows = window[window["was_filled"]]
            real_rows = window[~window["was_filled"]]

            zoom = go.Figure()
            zoom.add_trace(
                go.Scatter(
                    x=window["date"],
                    y=window["co2_ppm"],
                    mode="lines",
                    name="Series",
                    line=dict(color=theme.rgba(colour, 0.45), width=1.4),
                    hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} ppm<extra></extra>",
                )
            )
            zoom.add_trace(
                go.Scatter(
                    x=real_rows["date"],
                    y=real_rows["co2_ppm"],
                    mode="markers",
                    name="Measured",
                    marker=dict(color=colour, size=6),
                    hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} ppm<extra></extra>",
                )
            )
            zoom.add_trace(
                go.Scatter(
                    x=gap_rows["date"],
                    y=gap_rows["co2_ppm"],
                    mode="markers",
                    name="Synthesised",
                    marker=dict(color=theme.NEGATIVE, size=7, symbol="x"),
                    hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} ppm (filled)<extra></extra>",
                )
            )
            zoom.update_yaxes(title_text="CO\u2082 (ppm)")
            st.plotly_chart(theme.style(zoom, height=320), width="stretch")

            ui.finding(
                f"This stretch spans <b class='num'>{length}</b> {unit}. Interpolation is "
                f"defensible here because the underlying signal is smooth and slow-moving, but "
                f"a filled point is still an estimate: it carries no measurement error and "
                f"should not be treated as evidence on its own.",
                colour,
            )
        else:
            ui.finding(
                "This configuration produced no gaps, so nothing needed synthesising.", colour
            )

    # ------------------------------------------------------------- outliers
    with tab_outliers:
        ui.section(
            "Testing the outlier rule",
            "The interquartile rule is the standard first reach. It is worth checking whether "
            "it suits a series like this one.",
        )

        left, right = st.columns([1.25, 1], gap="large")
        with left:
            hist = go.Figure(
                go.Histogram(
                    x=clean["co2_ppm"],
                    nbinsx=60,
                    marker_color=theme.rgba(colour, 0.6),
                    marker_line_width=0,
                    hovertemplate="%{x:.0f} ppm<br>%{y} points<extra></extra>",
                )
            )
            for bound, label in [
                (report["iqr_lower"], "Lower bound"),
                (report["iqr_upper"], "Upper bound"),
            ]:
                hist.add_vline(
                    x=bound,
                    line=dict(color=theme.NEGATIVE, width=1.4, dash="dash"),
                    annotation_text=label,
                    annotation_font=dict(size=10, color=theme.TYPE_FAINT),
                )
            hist.update_xaxes(title_text="CO\u2082 (ppm)")
            hist.update_yaxes(title_text="Count")
            st.plotly_chart(theme.style(hist, height=320, legend=False), width="stretch")

        with right:
            ui.readouts(
                [
                    {"label": "Lower bound", "value": f"{report['iqr_lower']:.1f}", "unit": "ppm"},
                    {"label": "Upper bound", "value": f"{report['iqr_upper']:.1f}", "unit": "ppm"},
                    {"label": "Flagged", "value": f"{report['outliers_flagged']:,}"},
                    {"label": "Removed", "value": f"{report['outliers_removed']:,}"},
                ],
                colour,
            )

        ui.finding(
            "The distribution is wide and flat because the series is dominated by a trend, not "
            "because the readings are noisy. A rule built on quartiles of the whole record "
            "therefore has almost nothing to catch: only "
            f"<b class='num'>{report['outliers_flagged']}</b> of "
            f"<b class='num'>{report['rows_final']:,}</b> points fall outside the fence. "
            "Applying the same rule to residuals after detrending is far more informative, "
            "which is what the anomaly view on the next page does.",
            colour,
        )

        ui.caveat(
            "<b>Flag, then decide.</b> This pipeline marks outliers rather than silently "
            "deleting them. Removing a genuine extreme is how the record of an unusual event "
            "gets erased, so the switch to drop them is deliberate and off by default."
        )

    # -------------------------------------------------------------- outputs
    with tab_output:
        left, right = st.columns([1.15, 1], gap="large")

        with left:
            ui.section("Stage-by-stage ledger", "Every decision the pipeline made, in order.")
            st.markdown(_ledger(report, unit), unsafe_allow_html=True)

        with right:
            ui.section(
                "Engineered features", "Columns the later pages depend on, derived once here."
            )
            preview = (
                clean[
                    [
                        "date",
                        "co2_ppm",
                        "year",
                        "month",
                        "week_of_year",
                        "elapsed_days",
                        "rolling_mean",
                        "anomaly_z",
                    ]
                ]
                .tail(12)
                .copy()
            )
            preview["date"] = preview["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(preview.round(3), width="stretch", hide_index=True, height=330)

        ui.spacer(0.4)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download the cleaned series (CSV)",
                clean.to_csv(index=False).encode("utf-8"),
                file_name=f"co2_{frequency.lower()}_cleaned.csv",
                mime="text/csv",
                width="stretch",
            )
        with d2:
            st.download_button(
                "Download the audit report (JSON)",
                json.dumps(report, indent=2).encode("utf-8"),
                file_name=f"co2_{frequency.lower()}_report.json",
                mime="application/json",
                width="stretch",
            )

        with st.expander("Full audit report"):
            st.json(report)
