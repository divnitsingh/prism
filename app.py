"""Prism - an interactive data science atlas.

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Prism",
    page_icon="\u25e7",
    layout="wide",
    initial_sidebar_state="expanded",
)

from prism import theme, ui  # noqa: E402
from prism.modules import (  # noqa: E402
    atmosphere,
    cultivars,
    diagnostics,
    digits,
    home,
    quality,
    refinery,
)

PAGE_SPECS = [
    ("home", home.render, "Overview", ":material/grid_view:", "overview"),
    ("refinery", refinery.render, "Signal Refinery", ":material/science:", "signal-refinery"),
    ("atmosphere", atmosphere.render, "Atmospheric Trends", ":material/show_chart:", "atmospheric-trends"),
    ("cultivars", cultivars.render, "Cultivar Segmentation", ":material/scatter_plot:", "cultivar-segmentation"),
    ("diagnostics", diagnostics.render, "Diagnostic Intelligence", ":material/biotech:", "diagnostic-intelligence"),
    ("digits", digits.render, "Digit Recognition", ":material/grid_on:", "digit-recognition"),
    ("quality", quality.render, "Quality Lab", ":material/wine_bar:", "quality-lab"),
]

# Streamlit always injects its own navigation at the very top of the sidebar,
# whatever order the calls are made in. Hiding it and laying the links out by
# hand is what lets the wordmark sit above them.
NAV_STYLES = f"""
<style>
[data-testid="stSidebarNav"] {{ display: none; }}

[data-testid="stSidebar"] [data-testid="stPageLink"] a {{
  border-radius: 6px;
  padding: .42rem .6rem;
  margin: 0 0 .1rem 0;
  color: {theme.TYPE_DIM};
  font-size: .93rem;
  font-weight: 400;
  transition: background .14s ease, color .14s ease;
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{
  background: {theme.SURFACE};
  color: {theme.TYPE};
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"],
[data-testid="stSidebar"] [data-testid="stPageLink"] a.active {{
  background: {theme.SURFACE_HI};
  color: {theme.TYPE};
  font-weight: 500;
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] a span,
[data-testid="stSidebar"] [data-testid="stPageLink"] a p {{
  font-size: .93rem;
  line-height: 1.4;
}}

.sidebar-rule {{
  border-top: 1px solid {theme.RULE_SOFT};
  margin: .9rem 0 .85rem 0;
}}
</style>
"""


def main() -> None:
    theme.inject()

    pages: dict = {}
    for key, renderer, title, icon, url_path in PAGE_SPECS:
        pages[key] = st.Page(
            renderer,
            title=title,
            icon=icon,
            url_path=url_path,
            default=(key == "home"),
        )

    # Stored before the page runs so the overview can link to its siblings.
    st.session_state["_nav_pages"] = pages

    navigation = st.navigation(list(pages.values()), position="hidden")

    with st.sidebar:
        st.markdown(NAV_STYLES, unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-mark">PRISM</div>'
            '<div class="sidebar-sub">Data Science Atlas</div>'
            '<div class="sidebar-rule"></div>',
            unsafe_allow_html=True,
        )

        for page in pages.values():
            st.page_link(page, width="stretch")

        st.markdown(
            """
<div class="sidebar-note">
  Built with Streamlit, scikit-learn and Plotly. The convolutional network is
  written from scratch in NumPy.<br><br>
</div>
""",
            unsafe_allow_html=True,
        )

    navigation.run()


if __name__ == "__main__":
    main()