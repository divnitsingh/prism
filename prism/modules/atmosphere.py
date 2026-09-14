"""Atmospheric Trends - separating the seasonal breath from the long climb."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .. import datasets, theme, ui

MILESTONES = [320, 340, 360, 380, 400, 420]


@st.cache_data(show_spinner=False)
def _milestone_table() -> pd.DataFrame:
    """The first date the smoothed record passed each round number."""
    parts = datasets.decompose_co2("Weekly")
    smoothed = parts.dropna(subset=["trend"])
    rows = []
    for level in MILESTONES:
        above = smoothed[smoothed["trend"] >= level]
        if not above.empty:
            crossing = above.iloc[0]
            rows.append({"level": level, "date": crossing["date"], "year": int(crossing["year"])})
    table = pd.DataFrame(rows)
    if len(table) > 1:
        table["years_since_previous"] = (
            table["date"].diff().dt.days / 365.25
        ).round(1)
    return table


def render() -> None:
    colour = ui.page_header("atmosphere")

    clean, _ = datasets.clean_co2(frequency="Weekly")
    parts = datasets.decompose_co2("Weekly")

    year_min, year_max = int(clean["year"].min()), int(clean["year"].max())

    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            span = st.slider(
                "Years shown", year_min, year_max, (year_min, year_max), step=1
            )
        with c2:
            window = st.slider(
                "Smoothing window (weeks)", 4, 104, 52, step=4,
                help="52 weeks removes one full seasonal cycle.",
            )

    view = clean[clean["year"].between(*span)].copy()
    view["smoothed"] = view["co2_ppm"].rolling(window, center=True, min_periods=1).mean()
    parts_view = parts[parts["year"].between(*span)]

    if view.empty:
        ui.finding("No observations fall inside that range. Widen the slider to continue.")
        return

    # ------------------------------------------------------------- readouts
    annual = view.groupby("year", as_index=False)["co2_ppm"].mean()
    latest = float(view["co2_ppm"].iloc[-1])
    rise = latest - float(view["co2_ppm"].iloc[0])

    recent_growth = annual["co2_ppm"].diff().tail(10).mean()
    early_growth = annual["co2_ppm"].diff().head(11).mean()
    seasonal_amplitude = float(parts_view["seasonal"].max() - parts_view["seasonal"].min())

    ui.readouts(
        [
            {"label": "Latest reading", "value": f"{latest:.1f}", "unit": "ppm", "accent": True},
            {"label": "Rise over the window", "value": f"+{rise:.1f}", "unit": "ppm"},
            {
                "label": "Growth, last decade",
                "value": f"{recent_growth:.2f}",
                "unit": "ppm/yr",
                "delta": f"first decade {early_growth:.2f} ppm/yr",
                "tone": "flat",
            },
            {"label": "Seasonal swing", "value": f"{seasonal_amplitude:.1f}", "unit": "ppm"},
            {"label": "Seasons observed", "value": f"{span[1] - span[0] + 1}"},
        ],
        colour,
    )

    tab_climb, tab_cycle, tab_spiral, tab_anomaly = st.tabs(
        ["The climb", "Seasonal cycle", "Year on year", "Anomalies"]
    )

    # ---------------------------------------------------------------- climb
    with tab_climb:
        ui.section(
            "Trend and milestones",
            "Round numbers carry no physical meaning, but the shrinking interval between "
            "them is the story.",
        )

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=view["date"], y=view["co2_ppm"], mode="lines", name="Weekly reading",
                line=dict(color=theme.rgba(colour, 0.35), width=0.9),
                hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} ppm<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=view["date"], y=view["smoothed"], mode="lines",
                name=f"{window}-week mean",
                line=dict(color=colour, width=2.4),
                hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} ppm<extra></extra>",
            )
        )

        milestones = _milestone_table()
        inside = milestones[milestones["year"].between(*span)]
        for _, row in inside.iterrows():
            figure.add_hline(
                y=row["level"],
                line=dict(color=theme.RULE, width=1, dash="dot"),
                annotation_text=f"{row['level']} ppm \u2014 {row['year']}",
                annotation_position="top left",
                annotation_font=dict(size=10, color=theme.TYPE_FAINT),
            )
        figure.update_yaxes(title_text="CO\u2082 (ppm)")
        st.plotly_chart(theme.style(figure, height=420), width="stretch")

        if len(inside) >= 2:
            gaps = inside.dropna(subset=["years_since_previous"])
            if len(gaps) >= 2:
                first_gap = gaps.iloc[0]
                last_gap = gaps.iloc[-1]
                ui.finding(
                    f"It took <b class='num'>{first_gap['years_since_previous']:.0f}</b> years to "
                    f"climb the 20 ppm to <b class='num'>{first_gap['level']}</b>, but only "
                    f"<b class='num'>{last_gap['years_since_previous']:.0f}</b> years to climb the "
                    f"same 20 ppm to <b class='num'>{last_gap['level']}</b>. Equal steps arriving "
                    f"faster is what acceleration looks like on a chart.",
                    colour,
                )

        ui.section("Year-to-year change", "How much each year added over the one before it.")
        change = annual.copy()
        change["delta"] = change["co2_ppm"].diff()
        change = change.dropna(subset=["delta"])

        bars = go.Figure(
            go.Bar(
                x=change["year"], y=change["delta"],
                marker_color=theme.rgba(colour, 0.75), marker_line_width=0,
                name="Annual change",
                hovertemplate="%{x}<br>+%{y:.2f} ppm<extra></extra>",
            )
        )
        if len(change) > 2:
            fit = np.polyfit(change["year"], change["delta"], 1)
            bars.add_trace(
                go.Scatter(
                    x=change["year"], y=np.polyval(fit, change["year"]), mode="lines",
                    name="Linear fit",
                    line=dict(color=theme.NEGATIVE, width=2, dash="dash"),
                    hovertemplate="%{x}<br>%{y:.2f} ppm/yr<extra></extra>",
                )
            )
        bars.update_yaxes(title_text="Change (ppm/yr)")
        bars.update_xaxes(title_text="Year")
        st.plotly_chart(theme.style(bars, height=280), width="stretch")

        if len(change) > 2:
            ui.finding(
                f"The yearly increment is itself growing by about "
                f"<b class='num'>{fit[0] * 10:.2f}</b> ppm per year with every decade that "
                f"passes: early years added around <b class='num'>{early_growth:.2f}</b> ppm, "
                f"recent ones closer to <b class='num'>{recent_growth:.2f}</b> ppm. The record "
                f"is not merely rising, it is rising faster. Individual years scatter widely "
                f"around that line because El Ni\u00f1o conditions release extra carbon from "
                f"the tropics.",
                colour,
            )

    # ---------------------------------------------------------------- cycle
    with tab_cycle:
        ui.section(
            "Pulling the series apart",
            "A centred one-year average gives the trend; averaging what is left by week of "
            "year gives the season; the remainder is residual.",
        )

        figure = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.055,
            subplot_titles=("Trend", "Seasonal component", "Residual"),
        )
        figure.add_trace(
            go.Scatter(
                x=parts_view["date"], y=parts_view["trend"], mode="lines",
                line=dict(color=colour, width=2), name="Trend",
                hovertemplate="%{x|%b %Y}<br>%{y:.2f} ppm<extra></extra>",
            ), row=1, col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=parts_view["date"], y=parts_view["seasonal"], mode="lines",
                line=dict(color=theme.rgba(colour, 0.8), width=1), name="Seasonal",
                hovertemplate="%{x|%b %Y}<br>%{y:+.2f} ppm<extra></extra>",
            ), row=2, col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=parts_view["date"], y=parts_view["residual"], mode="lines",
                line=dict(color=theme.TYPE_FAINT, width=0.8), name="Residual",
                hovertemplate="%{x|%b %Y}<br>%{y:+.2f} ppm<extra></extra>",
            ), row=3, col=1,
        )
        for annotation in figure.layout.annotations:
            annotation.font.update(size=11, color=theme.TYPE_DIM, family=theme.FONT_DISPLAY)
        st.plotly_chart(
            theme.style(figure, height=520, legend=False, margin=(10, 10, 28, 10)),
            width="stretch",
        )

        ui.section(
            "The average year",
            "Averaging the seasonal component by calendar month shows the same cycle "
            "stripped of any particular year.",
        )
        monthly = (
            parts.assign(month_name=parts["date"].dt.strftime("%b"))
            .groupby("month", as_index=False)
            .agg(seasonal=("seasonal", "mean"), label=("month_name", "first"))
            .sort_values("month")
        )
        shape = go.Figure(
            go.Bar(
                x=monthly["label"], y=monthly["seasonal"],
                marker_color=[
                    colour if v > 0 else theme.rgba(colour, 0.35) for v in monthly["seasonal"]
                ],
                marker_line_width=0,
                hovertemplate="%{x}<br>%{y:+.2f} ppm<extra></extra>",
            )
        )
        shape.update_yaxes(title_text="Departure from trend (ppm)")
        st.plotly_chart(theme.style(shape, height=290, legend=False), width="stretch")

        peak = monthly.loc[monthly["seasonal"].idxmax()]
        trough = monthly.loc[monthly["seasonal"].idxmin()]
        residual_sd = float(parts_view["residual"].std())
        ui.finding(
            f"The cycle peaks in <b>{peak['label']}</b> at "
            f"<b class='num'>{peak['seasonal']:+.2f}</b> ppm and bottoms out in "
            f"<b>{trough['label']}</b> at <b class='num'>{trough['seasonal']:+.2f}</b> ppm. "
            f"Northern-hemisphere forests take up carbon through the growing season and "
            f"release it again as leaf litter decays over winter. Because most of the "
            f"world's land sits north of the equator, that hemisphere sets the rhythm for "
            f"the whole planet. What remains after removing trend and season has a standard "
            f"deviation of just <b class='num'>{residual_sd:.2f}</b> ppm, so the two "
            f"components together explain almost everything in the record.",
            colour,
        )

    # --------------------------------------------------------------- spiral
    with tab_spiral:
        ui.section(
            "Every year drawn on the same axis",
            "Each ring is one calendar year, running clockwise from January. Later years "
            "sit further out, so the widening gap between rings is the rise itself.",
        )

        spiral_source = parts_view.dropna(subset=["co2_ppm"])
        spiral = go.Figure()
        spiral.add_trace(
            go.Scatterpolar(
                r=spiral_source["co2_ppm"],
                theta=spiral_source["day_of_year"] / 365.25 * 360.0,
                mode="markers",
                marker=dict(
                    size=2.6,
                    color=spiral_source["year"],
                    colorscale=[[0, theme.fade(colour, 0.25)], [1, colour]],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Year", font=dict(size=10, color=theme.TYPE_FAINT)),
                        tickfont=dict(size=9, color=theme.TYPE_FAINT, family=theme.FONT_MONO),
                        thickness=10,
                        len=0.75,
                        outlinewidth=0,
                    ),
                ),
                customdata=np.stack(
                    [spiral_source["year"], spiral_source["co2_ppm"]], axis=-1
                ),
                hovertemplate="%{customdata[0]}<br>%{customdata[1]:.1f} ppm<extra></extra>",
                name="",
            )
        )
        spiral.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    gridcolor=theme.RULE_SOFT, linecolor=theme.RULE_SOFT,
                    tickfont=dict(size=9, color=theme.TYPE_FAINT, family=theme.FONT_MONO),
                    angle=90,
                ),
                angularaxis=dict(
                    gridcolor=theme.RULE_SOFT, linecolor=theme.RULE_SOFT,
                    direction="clockwise", rotation=90,
                    tickmode="array",
                    tickvals=[i * 30 for i in range(12)],
                    ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                    tickfont=dict(size=10, color=theme.TYPE_FAINT),
                ),
            )
        )
        st.plotly_chart(
            theme.style(spiral, height=540, legend=False, margin=(20, 20, 20, 20)),
            width="stretch",
        )

        ui.finding(
            "Read the rings inward to outward and the spacing tells the story twice over: "
            "the rings move outward because concentration is rising, and they spread further "
            "apart with time because the rise is speeding up. The lobe shape of each ring is "
            "the seasonal cycle, and it stays roughly the same width throughout, which is why "
            "the season can be modelled as an additive term rather than a growing one.",
            colour,
        )

    # ------------------------------------------------------------ anomalies
    with tab_anomaly:
        ui.section(
            "What the trend does not explain",
            "Residuals scored in standard deviations. Unlike the raw interquartile rule on "
            "the previous page, this one has something to find.",
        )

        residual = parts_view.dropna(subset=["residual"]).copy()
        residual_sd = float(residual["residual"].std())
        residual["z"] = residual["residual"] / (residual_sd if residual_sd else 1.0)
        extreme = residual[residual["z"].abs() >= 2]

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=residual["date"], y=residual["z"], mode="lines",
                line=dict(color=theme.rgba(colour, 0.6), width=0.9), name="Residual z-score",
                hovertemplate="%{x|%b %Y}<br>%{y:+.2f}\u03c3<extra></extra>",
            )
        )
        if not extreme.empty:
            figure.add_trace(
                go.Scatter(
                    x=extreme["date"], y=extreme["z"], mode="markers",
                    marker=dict(color=theme.NEGATIVE, size=5),
                    name=f"Beyond 2\u03c3 ({len(extreme):,})",
                    hovertemplate="%{x|%b %Y}<br>%{y:+.2f}\u03c3<extra></extra>",
                )
            )
        for level in (2, -2):
            figure.add_hline(y=level, line=dict(color=theme.RULE, width=1, dash="dash"))
        figure.update_yaxes(title_text="Standard deviations")
        st.plotly_chart(theme.style(figure, height=330), width="stretch")

        share = len(extreme) / max(len(residual), 1) * 100
        ui.finding(
            f"<b class='num'>{len(extreme):,}</b> of <b class='num'>{len(residual):,}</b> points "
            f"(<b class='num'>{share:.1f}%</b>) sit beyond two standard deviations. For a normal "
            f"distribution the expected share is about 4.6%, so anything close to that figure "
            f"means the residuals are behaving, and the trend-plus-season model is doing its job.",
            colour,
        )

        ui.section(
            "How the engineered features relate",
            "Strong correlations here are mostly a warning: several of these columns are "
            "different views of the same passage of time.",
        )
        numeric = view[
            ["co2_ppm", "rolling_mean", "annual_mean", "elapsed_days",
             "month", "week_of_year", "day_of_year"]
        ]
        matrix = numeric.corr(numeric_only=True)
        labels = [c.replace("_", " ") for c in matrix.columns]

        heat = go.Figure(
            go.Heatmap(
                z=matrix.to_numpy(), x=labels, y=labels,
                colorscale=theme.diverging(), zmid=0, zmin=-1, zmax=1,
                text=matrix.round(2).to_numpy(),
                texttemplate="%{text}",
                textfont=dict(size=10, family=theme.FONT_MONO),
                hovertemplate="%{y} \u00d7 %{x}<br>r = %{z:.3f}<extra></extra>",
                colorbar=dict(
                    thickness=10, len=0.8, outlinewidth=0,
                    tickfont=dict(size=9, color=theme.TYPE_FAINT, family=theme.FONT_MONO),
                ),
            )
        )
        st.plotly_chart(
            theme.style(heat, height=430, legend=False, margin=(10, 10, 10, 10)),
            width="stretch",
        )

        ui.finding(
            "Elapsed days correlates with concentration at almost exactly 1.0, which sounds "
            "like a triumph and is actually a caution: any model given that column will simply "
            "learn the calendar. The seasonal columns show near-zero linear correlation with "
            "concentration despite driving a clear cycle, because a sine wave has no linear "
            "relationship with anything. Correlation coefficients describe straight lines only.",
            colour,
        )
