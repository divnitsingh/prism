"""Visual identity for Prism.

The interface borrows its logic from a spectrograph: the instrument itself is
neutral graphite, and colour is reserved for the data. Each analysis owns one
spectral line, and that line is the only place its colour appears.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------
PLATE = "#0F1420"          # page field
SURFACE = "#161D2C"        # panels
SURFACE_HI = "#1C2436"     # raised panels
RULE = "#28324A"           # hairlines
RULE_SOFT = "#1F2738"
TYPE = "#E3E9F6"           # primary text
TYPE_DIM = "#8B98B6"       # secondary text
TYPE_FAINT = "#5D6B88"     # tertiary text

SPECTRUM = {
    "refinery": "#818CF8",     # indigo
    "atmosphere": "#22D3EE",   # cyan
    "cultivars": "#C084FC",    # violet
    "diagnostics": "#34D399",  # emerald
    "digits": "#FBBF24",       # amber
    "quality": "#FB7185",      # rose
}

POSITIVE = "#34D399"
NEGATIVE = "#FB7185"
NEUTRAL = "#8B98B6"

FONT_DISPLAY = "'Space Grotesk', 'Segoe UI', system-ui, sans-serif"
FONT_BODY = "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif"
FONT_MONO = "'IBM Plex Mono', 'Cascadia Mono', Consolas, monospace"


# --------------------------------------------------------------------------
# module registry
# --------------------------------------------------------------------------
MODULES: list[dict] = [
    {
        "key": "refinery",
        "title": "Signal Refinery",
        "blurb": "Turn 67 years of raw atmospheric readings into an analysis-ready series, "
                 "and watch every correction the pipeline makes.",
        "source": "Mauna Loa CO\u2082",
        "shape": "18,304 daily readings \u00b7 1958\u20132025",
        "methods": ["Validation", "Gap interpolation", "Outlier rules", "Feature engineering"],
        "icon": "\u2697",
    },
    {
        "key": "atmosphere",
        "title": "Atmospheric Trends",
        "blurb": "Separate the seasonal breath of the biosphere from the long climb, "
                 "and measure how fast that climb is accelerating.",
        "source": "Mauna Loa CO\u2082",
        "shape": "3,515 weekly points \u00b7 67 seasons",
        "methods": ["Seasonal decomposition", "Climatology", "Anomaly scoring", "Growth rates"],
        "icon": "\u25d1",
    },
    {
        "key": "cultivars",
        "title": "Cultivar Segmentation",
        "blurb": "Recover wine groupings from chemistry alone, with no labels supplied, "
                 "then check the answer against the truth.",
        "source": "Wine cultivars",
        "shape": "178 samples \u00b7 13 chemical measures",
        "methods": ["K-means", "Ward linkage", "PCA", "Silhouette analysis"],
        "icon": "\u25c8",
    },
    {
        "key": "diagnostics",
        "title": "Diagnostic Intelligence",
        "blurb": "Classify tumour biopsies, then move the decision threshold yourself "
                 "and see what it costs in missed cases.",
        "source": "Breast cancer diagnostic",
        "shape": "569 biopsies \u00b7 30 nuclear measures",
        "methods": ["Logistic regression", "Random forest", "Threshold tuning", "ROC \u0026 PR curves"],
        "icon": "\u25ce",
    },
    {
        "key": "digits",
        "title": "Digit Recognition",
        "blurb": "A convolutional network written from scratch in NumPy, trained live "
                 "while you watch the loss curve move.",
        "source": "Handwritten digits",
        "shape": "1,797 images \u00b7 8\u00d78 grayscale",
        "methods": ["Convolution", "Max pooling", "Dropout", "Adam"],
        "icon": "\u25a6",
    },
    {
        "key": "quality",
        "title": "Quality Lab",
        "blurb": "Predict how a red wine will score, find which chemistry moves the "
                 "needle, and pour a virtual glass of your own.",
        "source": "Red wine quality",
        "shape": "1,599 bottles \u00b7 11 physicochemical measures",
        "methods": ["Gradient boosting", "Cross-validation", "Permutation importance", "Segmentation"],
        "icon": "\u25d5",
    },
]

MODULE_BY_KEY = {m["key"]: m for m in MODULES}


def accent(key: str) -> str:
    return SPECTRUM.get(key, SPECTRUM["refinery"])


# --------------------------------------------------------------------------
# global stylesheet
# --------------------------------------------------------------------------
def _stylesheet() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
  --plate: {PLATE};
  --surface: {SURFACE};
  --surface-hi: {SURFACE_HI};
  --rule: {RULE};
  --rule-soft: {RULE_SOFT};
  --type: {TYPE};
  --type-dim: {TYPE_DIM};
  --type-faint: {TYPE_FAINT};
}}

.stApp {{
  background: var(--plate);
  font-family: {FONT_BODY};
  color: var(--type);
}}

.block-container {{ padding-top: 2.6rem; padding-bottom: 4rem; max-width: 1180px; }}

h1, h2, h3, h4 {{ font-family: {FONT_DISPLAY}; color: var(--type); letter-spacing: -0.015em; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{
  background: #0B0F19;
  border-right: 1px solid var(--rule-soft);
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}

.sidebar-mark {{
  font-family: {FONT_DISPLAY};
  font-weight: 700;
  font-size: 1.45rem;
  letter-spacing: 0.16em;
  color: var(--type);
  margin: 0 0 .15rem 0;
}}
.sidebar-sub {{
  font-family: {FONT_MONO};
  font-size: .66rem;
  color: var(--type-faint);
  letter-spacing: .07em;
  margin-bottom: 1.1rem;
}}
.sidebar-note {{
  font-size: .74rem;
  color: var(--type-faint);
  line-height: 1.55;
  border-top: 1px solid var(--rule-soft);
  padding-top: .85rem;
  margin-top: .6rem;
}}
.sidebar-note code {{
  font-family: {FONT_MONO};
  font-size: .7rem;
  color: var(--type-dim);
  background: transparent;
}}

/* ---------- masthead ---------- */
.mast {{ margin: 0 0 .4rem 0; }}
.mast-word {{
  font-family: {FONT_DISPLAY};
  font-weight: 700;
  font-size: clamp(3.1rem, 8vw, 5.4rem);
  line-height: .94;
  letter-spacing: 0.015em;
  margin: 0;
  color: var(--type);
}}
.mast-lede {{
  font-size: 1.06rem;
  line-height: 1.6;
  color: var(--type-dim);
  max-width: 60ch;
  margin: 1.05rem 0 0 0;
  font-weight: 300;
}}
.mast-lede strong {{ color: var(--type); font-weight: 500; }}

/* ---------- page header ---------- */
.page-head {{ margin: 0 0 1.5rem 0; }}
.page-title {{
  font-family: {FONT_DISPLAY};
  font-weight: 600;
  font-size: 2.35rem;
  line-height: 1.1;
  margin: .5rem 0 .55rem 0;
  color: var(--type);
}}
.page-blurb {{
  font-size: .99rem;
  line-height: 1.62;
  color: var(--type-dim);
  max-width: 68ch;
  font-weight: 300;
  margin: 0;
}}
.page-rule {{ height: 3px; width: 76px; border-radius: 2px; }}

.source-strip {{
  display: flex;
  flex-wrap: wrap;
  gap: 1.6rem;
  margin-top: 1.25rem;
  padding-top: .9rem;
  border-top: 1px solid var(--rule-soft);
  font-family: {FONT_MONO};
  font-size: .73rem;
  color: var(--type-faint);
  letter-spacing: .02em;
}}
.source-strip b {{ color: var(--type-dim); font-weight: 500; }}

/* ---------- readouts ---------- */
.readout-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
  gap: 1px;
  background: var(--rule-soft);
  border: 1px solid var(--rule-soft);
  border-radius: 6px;
  overflow: hidden;
  margin: .3rem 0 1.3rem 0;
}}
.readout {{ background: var(--surface); padding: .85rem 1rem .95rem 1rem; }}
.readout-label {{
  font-size: .71rem;
  color: var(--type-faint);
  margin-bottom: .34rem;
  line-height: 1.3;
  font-weight: 400;
}}
.readout-value {{
  font-family: {FONT_MONO};
  font-size: 1.42rem;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
  line-height: 1.1;
  color: var(--type);
}}
.readout-unit {{ font-size: .82rem; color: var(--type-dim); margin-left: .2rem; }}
.readout-delta {{
  font-family: {FONT_MONO};
  font-size: .72rem;
  margin-top: .3rem;
  font-variant-numeric: tabular-nums;
}}

/* ---------- index of analyses ---------- */
.index-row {{
  display: grid;
  grid-template-columns: 4px 1fr;
  gap: 1.15rem;
  padding: 1.25rem 0 1.15rem 0;
  border-bottom: 1px solid var(--rule-soft);
}}
.index-bar {{ border-radius: 2px; }}
.index-title {{
  font-family: {FONT_DISPLAY};
  font-size: 1.26rem;
  font-weight: 600;
  margin: 0 0 .3rem 0;
  color: var(--type);
}}
.index-blurb {{
  font-size: .9rem;
  line-height: 1.58;
  color: var(--type-dim);
  margin: 0 0 .7rem 0;
  max-width: 66ch;
  font-weight: 300;
}}
.index-meta {{
  font-family: {FONT_MONO};
  font-size: .7rem;
  color: var(--type-faint);
  letter-spacing: .02em;
}}
.tagset {{ display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .6rem; }}
.tag {{
  font-size: .69rem;
  padding: .2rem .5rem;
  border: 1px solid var(--rule);
  border-radius: 3px;
  color: var(--type-dim);
  background: var(--surface);
}}

/* ---------- panels & notes ---------- */
.panel {{
  background: var(--surface);
  border: 1px solid var(--rule-soft);
  border-radius: 7px;
  padding: 1.05rem 1.15rem;
  margin-bottom: .9rem;
}}
.panel h4 {{ margin: 0 0 .5rem 0; font-size: .98rem; font-weight: 600; }}
.panel p {{ margin: 0; font-size: .87rem; line-height: 1.6; color: var(--type-dim); font-weight: 300; }}

.finding {{
  border-left: 3px solid var(--rule);
  padding: .1rem 0 .1rem 1rem;
  margin: .2rem 0 1.15rem 0;
  font-size: .9rem;
  line-height: 1.65;
  color: var(--type-dim);
  font-weight: 300;
}}
.finding b {{ color: var(--type); font-weight: 500; }}
.finding .num {{ font-family: {FONT_MONO}; font-variant-numeric: tabular-nums; }}

.caveat {{
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: .8rem 1rem;
  font-size: .82rem;
  line-height: 1.6;
  color: var(--type-dim);
  background: var(--surface);
  font-weight: 300;
}}
.caveat b {{ color: var(--type); font-weight: 500; }}

.section-label {{
  font-family: {FONT_DISPLAY};
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--type);
  margin: 1.5rem 0 .2rem 0;
}}
.section-hint {{
  font-size: .85rem;
  color: var(--type-faint);
  margin: 0 0 .8rem 0;
  line-height: 1.55;
  font-weight: 300;
  max-width: 70ch;
}}

/* ---------- pixel canvas ---------- */
.canvas-wrap {{ display: inline-block; border: 1px solid var(--rule); border-radius: 6px; padding: 6px; background: #0A0E17; }}

/* ---------- streamlit widget polish ---------- */
.stButton > button {{
  font-family: {FONT_BODY};
  font-size: .84rem;
  font-weight: 500;
  border-radius: 5px;
  border: 1px solid var(--rule);
  background: var(--surface-hi);
  color: var(--type);
  transition: border-color .14s ease, background .14s ease;
}}
.stButton > button:hover {{ border-color: var(--type-faint); background: #212a3d; color: var(--type); }}
.stButton > button:focus-visible {{ outline: 2px solid #818CF8; outline-offset: 2px; }}

[data-testid="stTabs"] button {{ font-family: {FONT_BODY}; font-size: .88rem; }}
[data-testid="stMetricValue"] {{ font-family: {FONT_MONO}; font-variant-numeric: tabular-nums; }}
[data-testid="stExpander"] details {{ border: 1px solid var(--rule-soft); border-radius: 6px; background: var(--surface); }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--rule-soft); border-radius: 6px; }}

.stSlider label, .stSelectbox label, .stMultiSelect label,
.stRadio label, .stCheckbox label, .stNumberInput label {{
  font-size: .82rem !important;
  color: var(--type-dim) !important;
  font-weight: 400 !important;
}}

hr {{ border-color: var(--rule-soft); }}
a {{ color: #A5B4FC; }}

@media (prefers-reduced-motion: reduce) {{
  * {{ animation: none !important; transition: none !important; }}
}}
</style>
"""


def inject() -> None:
    """Apply the stylesheet once per rerun."""
    st.markdown(_stylesheet(), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# plotly styling
# --------------------------------------------------------------------------
def style(
    fig: go.Figure,
    height: int = 380,
    accent_color: str | None = None,
    legend: bool = True,
    margin: tuple[int, int, int, int] = (10, 10, 30, 10),
) -> go.Figure:
    """Apply the house chart style. Order is (l, r, t, b)."""
    left, right, top, bottom = margin
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=left, r=right, t=top, b=bottom),
        font=dict(family=FONT_BODY, size=12, color=TYPE_DIM),
        hoverlabel=dict(
            bgcolor=SURFACE_HI,
            bordercolor=RULE,
            font=dict(family=FONT_MONO, size=11, color=TYPE),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        )
        if legend
        else dict(),
        showlegend=legend,
    )
    fig.update_xaxes(
        gridcolor=RULE_SOFT,
        zerolinecolor=RULE,
        linecolor=RULE,
        tickfont=dict(family=FONT_MONO, size=10, color=TYPE_FAINT),
        title_font=dict(size=11, color=TYPE_FAINT),
    )
    fig.update_yaxes(
        gridcolor=RULE_SOFT,
        zerolinecolor=RULE,
        linecolor=RULE,
        tickfont=dict(family=FONT_MONO, size=10, color=TYPE_FAINT),
        title_font=dict(size=11, color=TYPE_FAINT),
    )
    # A title block carrying only a font and no text makes plotly.js render the
    # literal word "undefined" where the heading would sit, so the display font
    # is applied only to figures that actually set a title.
    if fig.layout.title.text:
        fig.update_layout(title_font=dict(family=FONT_DISPLAY, size=14, color=TYPE))

    if accent_color:
        fig.update_layout(colorway=[accent_color, TYPE_DIM, "#A5B4FC", "#F472B6", "#5EEAD4"])
    return fig


def sequential(color: str) -> list[list]:
    """A single-hue colour scale that fades into the page field."""
    return [[0.0, PLATE], [0.45, _mix(PLATE, color, 0.45)], [1.0, color]]


def diverging() -> list[list]:
    return [
        [0.0, "#4C5FD7"],
        [0.25, "#5B6EE0"],
        [0.5, SURFACE],
        [0.75, "#E8657F"],
        [1.0, "#FB7185"],
    ]


def _mix(a: str, b: str, t: float) -> str:
    ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    return "#%02x%02x%02x" % (
        round(ar + (br - ar) * t),
        round(ag + (bg - ag) * t),
        round(ab + (bb - ab) * t),
    )


def fade(color: str, t: float) -> str:
    """Blend a spectral colour toward the page field."""
    return _mix(PLATE, color, t)


def rgba(color: str, alpha: float) -> str:
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"
