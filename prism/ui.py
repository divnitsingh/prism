"""Reusable interface pieces.

Everything here renders plain HTML so the look stays stable across Streamlit
releases rather than depending on internal class names.
"""

from __future__ import annotations

from html import escape

import numpy as np
import streamlit as st

from . import theme


# --------------------------------------------------------------------------
# headers
# --------------------------------------------------------------------------
def masthead(lede: str) -> None:
    st.markdown(
        f"""
<div class="mast">
  <h1 class="mast-word">Prism</h1>
  <p class="mast-lede">{lede}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def page_header(key: str) -> str:
    """Render a module header and return its spectral colour."""
    module = theme.MODULE_BY_KEY[key]
    colour = theme.accent(key)
    st.markdown(
        f"""
<div class="page-head">
  <div class="page-rule" style="background:{colour};"></div>
  <h2 class="page-title">{escape(module["title"])}</h2>
  <p class="page-blurb">{escape(module["blurb"])}</p>
  <div class="source-strip">
    <span><b>Source</b>&nbsp;&nbsp;{escape(module["source"])}</span>
    <span><b>Shape</b>&nbsp;&nbsp;{escape(module["shape"])}</span>
    <span><b>Methods</b>&nbsp;&nbsp;{escape(" / ".join(module["methods"]))}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    return colour


def section(label: str, hint: str = "") -> None:
    hint_html = f'<p class="section-hint">{escape(hint)}</p>' if hint else ""
    st.markdown(
        f'<div class="section-label">{escape(label)}</div>{hint_html}',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# readouts
# --------------------------------------------------------------------------
def readouts(items: list[dict], colour: str | None = None) -> None:
    """A strip of instrument-style numeric readouts.

    Each item accepts: label, value, unit, delta, tone ('up' | 'down' | 'flat').
    """
    cells = []
    for item in items:
        unit = item.get("unit")
        unit_html = f'<span class="readout-unit">{escape(str(unit))}</span>' if unit else ""

        delta = item.get("delta")
        delta_html = ""
        if delta:
            tone = item.get("tone", "flat")
            tone_colour = {
                "up": theme.POSITIVE,
                "down": theme.NEGATIVE,
                "flat": theme.TYPE_FAINT,
            }.get(tone, theme.TYPE_FAINT)
            delta_html = (
                f'<div class="readout-delta" style="color:{tone_colour};">'
                f"{escape(str(delta))}</div>"
            )

        value_colour = f"color:{colour};" if item.get("accent") and colour else ""
        cells.append(
            f'<div class="readout">'
            f'<div class="readout-label">{escape(str(item["label"]))}</div>'
            f'<div class="readout-value" style="{value_colour}">'
            f'{escape(str(item["value"]))}{unit_html}</div>'
            f"{delta_html}</div>"
        )
    st.markdown(f'<div class="readout-row">{"".join(cells)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# callouts
# --------------------------------------------------------------------------
def finding(html: str, colour: str | None = None) -> None:
    """A short interpretation of the chart above it. Accepts inline markup."""
    border = f"border-left-color:{colour};" if colour else ""
    st.markdown(f'<div class="finding" style="{border}">{html}</div>', unsafe_allow_html=True)


def caveat(html: str) -> None:
    st.markdown(f'<div class="caveat">{html}</div>', unsafe_allow_html=True)


def panel(title: str, body: str) -> None:
    st.markdown(
        f'<div class="panel"><h4>{escape(title)}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# home page index
# --------------------------------------------------------------------------
def index_row(module: dict) -> None:
    colour = theme.accent(module["key"])
    tags = "".join(f'<span class="tag">{escape(m)}</span>' for m in module["methods"])
    st.markdown(
        f"""
<div class="index-row">
  <div class="index-bar" style="background:{colour};"></div>
  <div>
    <div class="index-title">{escape(module["title"])}</div>
    <p class="index-blurb">{escape(module["blurb"])}</p>
    <div class="index-meta">{escape(module["source"])} &nbsp;&mdash;&nbsp; {escape(module["shape"])}</div>
    <div class="tagset">{tags}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# spectral hero
# --------------------------------------------------------------------------
def spectral_stack(traces: list[dict], height_per: int = 46) -> None:
    """Stack one real sparkline per analysis, each in its own spectral colour.

    Every trace is computed from the actual dataset behind that analysis, so the
    band is a genuine readout rather than decoration.
    """
    width = 1000
    label_w = 210
    plot_w = width - label_w - 112
    gap = 10
    total_h = len(traces) * (height_per + gap)

    parts = [
        f'<svg viewBox="0 0 {width} {total_h}" width="100%" '
        f'style="display:block;margin:1.9rem 0 .6rem 0;" '
        f'role="img" aria-label="One sparkline per analysis, each drawn from its own dataset.">'
    ]

    for i, tr in enumerate(traces):
        y = np.asarray(tr["y"], dtype=float)
        colour = tr["colour"]
        top = i * (height_per + gap)
        pad = 7
        inner = height_per - 2 * pad

        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        span = hi - lo
        norm = np.full_like(y, 0.5) if span <= 0 else (y - lo) / span

        xs = np.linspace(label_w, label_w + plot_w, len(y))
        ys = top + pad + (1.0 - norm) * inner
        points = " ".join(f"{x:.1f},{v:.1f}" for x, v in zip(xs, ys))

        baseline = top + height_per - pad
        area = f"{label_w},{baseline:.1f} {points} {label_w + plot_w},{baseline:.1f}"

        parts.append(
            f'<polygon points="{area}" fill="{theme.rgba(colour, 0.10)}" />'
            f'<polyline points="{points}" fill="none" stroke="{colour}" '
            f'stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round" />'
            f'<text x="0" y="{top + height_per / 2 - 2:.1f}" fill="{theme.TYPE_DIM}" '
            f'font-family="{theme.FONT_DISPLAY}" font-size="13" font-weight="500">'
            f'{escape(tr["title"])}</text>'
            f'<text x="0" y="{top + height_per / 2 + 12:.1f}" fill="{theme.TYPE_FAINT}" '
            f'font-family="{theme.FONT_MONO}" font-size="9.5">{escape(tr["caption"])}</text>'
            f'<text x="{label_w + plot_w + 12}" y="{top + height_per / 2 + 4:.1f}" '
            f'fill="{theme.TYPE_FAINT}" font-family="{theme.FONT_MONO}" font-size="9.5">'
            f'{escape(tr["scale"])}</text>'
        )

    parts.append("</svg>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def pixel_grid_svg(image: np.ndarray, colour: str, size: int = 172, vmax: float = 16.0) -> str:
    """Render a small greyscale/tinted image as crisp SVG squares."""
    img = np.asarray(image, dtype=float)
    n_rows, n_cols = img.shape
    cell = size / max(n_rows, n_cols)
    rects = []
    for r in range(n_rows):
        for c in range(n_cols):
            v = float(np.clip(img[r, c] / vmax, 0.0, 1.0))
            fill = theme.fade(colour, v) if v > 0.01 else "#0A0E17"
            rects.append(
                f'<rect x="{c * cell:.2f}" y="{r * cell:.2f}" width="{cell:.2f}" '
                f'height="{cell:.2f}" fill="{fill}" />'
            )
    return (
        f'<svg viewBox="0 0 {n_cols * cell:.1f} {n_rows * cell:.1f}" width="{size}" '
        f'height="{size * n_rows / n_cols:.0f}" shape-rendering="crispEdges">'
        f'{"".join(rects)}</svg>'
    )


def spacer(rem: float = 1.0) -> None:
    st.markdown(f'<div style="height:{rem}rem;"></div>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {theme.RULE_SOFT};margin:1.6rem 0;">',
        unsafe_allow_html=True,
    )
