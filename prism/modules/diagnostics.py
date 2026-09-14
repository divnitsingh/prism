"""Diagnostic Intelligence - classification, and the cost of where you draw the line."""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from .. import models, theme, ui

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
}


@st.cache_data(show_spinner=False)
def _sweep(name: str, test_fraction: float, use_ratios: bool) -> pd.DataFrame:
    """Score every threshold from 0.02 to 0.98 in one pass."""
    bundle = models.fit_classifier(name, test_fraction, use_ratios)
    rows = []
    for threshold in np.linspace(0.02, 0.98, 97):
        scored = models.threshold_metrics(
            bundle["y_test"], bundle["probabilities"], float(threshold)
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": scored["precision"],
                "recall": scored["recall"],
                "f1": scored["f1"],
                "missed": scored["fn"],
                "false_alarms": scored["fp"],
            }
        )
    return pd.DataFrame(rows)


# The four cells of the confusion matrix, in reading order. Naming each outcome
# beneath the chart rather than inside it keeps the cells legible: at this size
# a two-line caption per cell simply collides with its neighbours.
OUTCOMES = [
    ("tn", "Correctly cleared", "benign tissue, called benign"),
    ("fp", "False alarm", "benign tissue, called malignant"),
    ("fn", "Missed", "malignant tissue, called benign"),
    ("tp", "Correctly caught", "malignant tissue, called malignant"),
]


def _outcome_fills(colour: str) -> dict[str, str]:
    """Colour carries the meaning: correct outcomes share the module hue, and
    the two kinds of mistake are kept apart because they are not equally bad."""
    return {
        "tn": theme.fade(colour, 0.32),
        "tp": theme.fade(colour, 0.62),
        "fp": theme.fade("#6C8CFF", 0.52),
        "fn": theme.fade(theme.NEGATIVE, 0.58),
    }


def _confusion_figure(scored: dict, colour: str) -> go.Figure:
    fills = _outcome_fills(colour)
    counts = np.array([[scored["tn"], scored["fp"]], [scored["fn"], scored["tp"]]])

    # z holds a category index per cell, so the scale below is a flat lookup
    # rather than a gradient over the counts.
    categories = np.array([[0, 1], [2, 3]])
    order = [fills["tn"], fills["fp"], fills["fn"], fills["tp"]]
    scale = []
    for position, tint in enumerate(order):
        scale.append([position / 4, tint])
        scale.append([(position + 1) / 4, tint])

    figure = go.Figure(
        go.Heatmap(
            z=categories,
            x=["Benign", "Malignant"],
            y=["Benign", "Malignant"],
            colorscale=scale,
            zmin=-0.5,
            zmax=3.5,
            xgap=4,
            ygap=4,
            text=counts,
            texttemplate="%{text}",
            textfont=dict(size=26, family=theme.FONT_MONO, color=theme.TYPE),
            customdata=np.array(
                [[OUTCOMES[0][1], OUTCOMES[1][1]], [OUTCOMES[2][1], OUTCOMES[3][1]]]
            ),
            hovertemplate="%{customdata}<br>%{text} cases<extra></extra>",
            showscale=False,
        )
    )
    figure.update_xaxes(title_text="What the model called it", tickangle=0, showgrid=False)
    figure.update_yaxes(
        title_text="Recorded diagnosis", tickangle=0, showgrid=False, autorange="reversed"
    )
    return figure


def _outcome_key(scored: dict, colour: str) -> str:
    """A small legend under the matrix, so the cells can stay uncluttered."""
    fills = _outcome_fills(colour)
    cells = []
    for key, name, meaning in OUTCOMES:
        cells.append(
            f'<div style="display:flex;align-items:flex-start;gap:.55rem;'
            f'padding:.6rem .2rem;border-top:1px solid {theme.RULE_SOFT};">'
            f'<span style="width:12px;height:12px;border-radius:3px;flex:none;'
            f'margin-top:.25rem;background:{fills[key]};"></span>'
            f"<div>"
            f'<div style="font-size:.83rem;color:{theme.TYPE};font-weight:500;">'
            f"{escape(name)}"
            f'<span style="font-family:{theme.FONT_MONO};color:{theme.TYPE_DIM};'
            f'margin-left:.4rem;">{scored[key]}</span></div>'
            f'<div style="font-size:.73rem;color:{theme.TYPE_FAINT};line-height:1.42;">'
            f"{escape(meaning)}</div></div></div>"
        )
    return (
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));'
        'gap:0 1.3rem;margin:.1rem 0 .6rem 0;">' + "".join(cells) + "</div>"
    )


def render() -> None:
    colour = ui.page_header("diagnostics")

    # ------------------------------------------------------------- controls
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.1, 1, 1])
        with c1:
            name = st.selectbox("Model", list(models.CLASSIFIERS), index=0)
        with c2:
            test_fraction = st.slider("Share held back for testing", 0.15, 0.40, 0.25, 0.05)
        with c3:
            use_ratios = st.checkbox(
                "Add engineered shape ratios",
                value=True,
                help="Row-wise ratios such as perimeter to radius. Nothing is fitted from "
                "the data to build them, so they can be made before the split.",
            )

    bundle = models.fit_classifier(name, test_fraction, use_ratios)
    y_test = bundle["y_test"]
    probabilities = bundle["probabilities"]

    auc = roc_auc_score(y_test, probabilities)
    average_precision = average_precision_score(y_test, probabilities)
    sweep = _sweep(name, test_fraction, use_ratios)

    best_f1_row = sweep.loc[sweep["f1"].idxmax()]
    catch_all = sweep[sweep["missed"] == 0]
    catch_all_threshold = float(catch_all["threshold"].max()) if not catch_all.empty else None

    # ------------------------------------------------------------ threshold
    ui.section(
        "Where to draw the line",
        "The model outputs a probability, not a verdict. Turning that probability into a "
        "decision means choosing a cut-off, and that choice is a judgement about which "
        "kind of mistake is worse.",
    )

    with st.container(border=True):
        threshold = st.slider(
            "Call a case malignant when the probability is at least",
            0.02, 0.98, 0.50, 0.01,
            key="diagnostic_threshold",
        )
        presets = st.columns(3)
        with presets[0]:
            st.caption(f"Best balance sits at {best_f1_row['threshold']:.2f}")
        with presets[1]:
            if catch_all_threshold is not None:
                st.caption(f"Nothing is missed up to {catch_all_threshold:.2f}")
            else:
                st.caption("No cut-off catches every case")
        with presets[2]:
            st.caption("Convention is 0.50")

    scored = models.threshold_metrics(y_test, probabilities, threshold)

    ui.readouts(
        [
            {"label": "Missed malignancies", "value": f"{scored['fn']}",
             "delta": "the costly error", "tone": "down" if scored["fn"] else "up"},
            {"label": "False alarms", "value": f"{scored['fp']}",
             "delta": "benign called malignant", "tone": "flat"},
            {"label": "Recall", "value": f"{scored['recall']:.3f}",
             "delta": "share of cancers caught", "tone": "flat"},
            {"label": "Precision", "value": f"{scored['precision']:.3f}",
             "delta": "share of alarms that were real", "tone": "flat"},
            {"label": "ROC-AUC", "value": f"{auc:.3f}", "accent": True,
             "delta": "independent of threshold", "tone": "flat"},
        ],
        colour,
    )

    tab_line, tab_curves, tab_evidence, tab_case = st.tabs(
        ["The trade-off", "Curves", "What the model uses", "Score a case"]
    )

    # ----------------------------------------------------------- trade-off
    with tab_line:
        left, right = st.columns([1, 1.15], gap="large")

        with left:
            ui.section("At the current cut-off", "")
            st.plotly_chart(
                theme.style(
                    _confusion_figure(scored, colour),
                    height=300,
                    legend=False,
                    margin=(10, 10, 10, 10),
                ),
                width="stretch",
            )
            st.markdown(_outcome_key(scored, colour), unsafe_allow_html=True)

        with right:
            ui.section("As the cut-off moves", "")
            figure = go.Figure()
            for column, label, line_colour in [
                ("recall", "Recall", colour),
                ("precision", "Precision", "#A5B4FC"),
                ("f1", "F1", theme.TYPE_DIM),
            ]:
                figure.add_trace(
                    go.Scatter(
                        x=sweep["threshold"], y=sweep[column], mode="lines", name=label,
                        line=dict(color=line_colour, width=2),
                        hovertemplate=label + " %{y:.3f} at %{x:.2f}<extra></extra>",
                    )
                )
            figure.add_vline(
                x=threshold, line=dict(color=theme.NEGATIVE, width=1.6, dash="dash"),
                annotation_text="your cut-off",
                annotation_font=dict(size=10, color=theme.TYPE_FAINT),
            )
            figure.update_xaxes(title_text="Decision threshold")
            figure.update_yaxes(title_text="Score", range=[0, 1.03])
            st.plotly_chart(theme.style(figure, height=320), width="stretch")

        ui.section(
            "How the two classes score",
            "The further apart these two piles sit, the less the exact cut-off matters.",
        )
        figure = go.Figure()
        for label, class_name, tint in [
            (0, "Benign", theme.rgba("#A5B4FC", 0.55)),
            (1, "Malignant", theme.rgba(colour, 0.75)),
        ]:
            figure.add_trace(
                go.Histogram(
                    x=probabilities[y_test == label],
                    name=class_name, nbinsx=40, marker_color=tint, marker_line_width=0,
                    hovertemplate=class_name + "<br>%{y} cases near %{x:.2f}<extra></extra>",
                )
            )
        figure.add_vline(
            x=threshold, line=dict(color=theme.NEGATIVE, width=1.6, dash="dash"),
            annotation_text="cut-off", annotation_font=dict(size=10, color=theme.TYPE_FAINT),
        )
        figure.update_layout(barmode="overlay")
        figure.update_traces(opacity=0.78)
        figure.update_xaxes(title_text="Predicted probability of malignancy")
        figure.update_yaxes(title_text="Cases")
        st.plotly_chart(theme.style(figure, height=300), width="stretch")

        if catch_all_threshold is not None:
            catch_row = sweep.loc[(sweep["threshold"] - catch_all_threshold).abs().idxmin()]
            ui.finding(
                f"Moving the cut-off down to <b class='num'>{catch_all_threshold:.2f}</b> catches "
                f"every malignancy in this test set, at the price of "
                f"<b class='num'>{int(catch_row['false_alarms'])}</b> false alarms. At the "
                f"conventional 0.50 the model misses cases but raises fewer alarms. Neither "
                f"setting is correct in the abstract: a false alarm costs a follow-up biopsy, "
                f"a miss costs far more, and only someone who knows those costs can choose.",
                colour,
            )
        else:
            ui.finding(
                "No cut-off in this range catches every malignancy, so some cases are scored "
                "below even the lowest threshold tested.",
                colour,
            )

    # -------------------------------------------------------------- curves
    with tab_curves:
        fpr, tpr, roc_thresholds = roc_curve(y_test, probabilities)
        precision_values, recall_values, pr_thresholds = precision_recall_curve(
            y_test, probabilities
        )

        left, right = st.columns(2, gap="large")

        with left:
            ui.section("ROC", "Catching cancers against raising false alarms.")
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"AUC {auc:.4f}",
                    line=dict(color=colour, width=2.4),
                    fill="tozeroy", fillcolor=theme.rgba(colour, 0.08),
                    hovertemplate="False alarm rate %{x:.3f}<br>Caught %{y:.3f}<extra></extra>",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="Coin flip",
                    line=dict(color=theme.RULE, width=1.2, dash="dash"),
                    hoverinfo="skip",
                )
            )
            index = int(np.argmin(np.abs(roc_thresholds - threshold)))
            figure.add_trace(
                go.Scatter(
                    x=[fpr[index]], y=[tpr[index]], mode="markers", name="Your cut-off",
                    marker=dict(color=theme.NEGATIVE, size=11, symbol="circle",
                                line=dict(color=theme.PLATE, width=1.5)),
                    hovertemplate="Your cut-off<br>%{y:.3f} caught<extra></extra>",
                )
            )
            figure.update_xaxes(title_text="False alarm rate", range=[-0.02, 1.02])
            figure.update_yaxes(title_text="Share of cancers caught", range=[-0.02, 1.02])
            st.plotly_chart(theme.style(figure, height=380), width="stretch")

        with right:
            ui.section(
                "Precision against recall",
                "The curve when one class is rarer than the other.",
            )
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=recall_values, y=precision_values, mode="lines",
                    name=f"Average precision {average_precision:.4f}",
                    line=dict(color=colour, width=2.4),
                    fill="tozeroy", fillcolor=theme.rgba(colour, 0.08),
                    hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>",
                )
            )
            base_rate = float(np.mean(y_test))
            figure.add_hline(
                y=base_rate, line=dict(color=theme.RULE, width=1.2, dash="dash"),
                annotation_text=f"always-malignant baseline {base_rate:.2f}",
                annotation_font=dict(size=10, color=theme.TYPE_FAINT),
            )
            figure.add_trace(
                go.Scatter(
                    x=[scored["recall"]], y=[scored["precision"]], mode="markers",
                    name="Your cut-off",
                    marker=dict(color=theme.NEGATIVE, size=11,
                                line=dict(color=theme.PLATE, width=1.5)),
                    hovertemplate="Your cut-off<extra></extra>",
                )
            )
            figure.update_xaxes(title_text="Recall", range=[-0.02, 1.02])
            figure.update_yaxes(title_text="Precision", range=[-0.02, 1.02])
            st.plotly_chart(theme.style(figure, height=380), width="stretch")

        ui.section(
            "Five-fold cross-validation",
            "One split can flatter a model. Refitting across five different splits shows how "
            "much the score depends on which rows happened to land in the test set.",
        )
        folds = models.classifier_cross_validation(name, use_ratios, 5)
        figure = go.Figure()
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            values = folds[folds["metric"] == metric]["score"]
            figure.add_trace(
                go.Box(
                    y=values, name=METRIC_LABELS[metric],
                    marker_color=colour, line=dict(color=colour, width=1.6),
                    fillcolor=theme.rgba(colour, 0.16),
                    boxpoints="all", jitter=0.5, pointpos=0,
                    marker=dict(size=5, opacity=0.8),
                    hovertemplate="%{y:.4f}<extra></extra>",
                )
            )
        figure.update_yaxes(title_text="Score across folds")
        st.plotly_chart(theme.style(figure, height=330, legend=False), width="stretch")

        summary = folds.groupby("metric")["score"].agg(["mean", "std"])
        spread = summary.loc["recall"]
        ui.finding(
            f"Recall averages <b class='num'>{spread['mean']:.3f}</b> with a standard deviation "
            f"of <b class='num'>{spread['std']:.3f}</b> across folds. Quoting a single test "
            f"score without that spread would overstate how precisely the performance is known, "
            f"and on a test set of this size a couple of rows moving between folds is enough to "
            f"shift the third decimal place.",
            colour,
        )

    # ------------------------------------------------------------ evidence
    with tab_evidence:
        ui.section(
            "Which measurements carry the decision",
            "Each column is shuffled in turn and the drop in ROC-AUC recorded. Measured on "
            "held-out data, so this reflects what actually generalises.",
        )

        importance = models.classifier_importance(name, test_fraction, use_ratios, 10)
        top = importance.head(14).sort_values("importance")

        figure = go.Figure(
            go.Bar(
                x=top["importance"], y=[f.replace("_", " ") for f in top["feature"]],
                orientation="h",
                marker_color=theme.rgba(colour, 0.8), marker_line_width=0,
                error_x=dict(
                    type="data", array=top["spread"], color=theme.TYPE_FAINT,
                    thickness=1, width=3,
                ),
                hovertemplate="%{y}<br>AUC falls %{x:.4f} when shuffled<extra></extra>",
            )
        )
        figure.update_xaxes(title_text="Drop in ROC-AUC when shuffled")
        st.plotly_chart(theme.style(figure, height=460, legend=False), width="stretch")

        leader = importance.iloc[0]
        ui.finding(
            f"Shuffling <b>{leader['feature'].replace('_', ' ')}</b> costs the most, lowering "
            f"ROC-AUC by <b class='num'>{leader['importance']:.4f}</b>. Low scores further down "
            f"do not prove a measurement is useless: these columns are heavily correlated with "
            f"one another, so when one is destroyed the model simply leans on its neighbour and "
            f"barely suffers. Permutation importance answers what this model relies on, not "
            f"what matters biologically.",
            colour,
        )

        with st.expander("Compare against the other model"):
            other = [m for m in models.CLASSIFIERS if m != name][0]
            other_bundle = models.fit_classifier(other, test_fraction, use_ratios)
            other_scored = models.threshold_metrics(
                other_bundle["y_test"], other_bundle["probabilities"], threshold
            )
            comparison = pd.DataFrame(
                [
                    {
                        "Model": name,
                        "ROC-AUC": round(auc, 4),
                        "Recall": round(scored["recall"], 4),
                        "Precision": round(scored["precision"], 4),
                        "Missed": scored["fn"],
                        "False alarms": scored["fp"],
                    },
                    {
                        "Model": other,
                        "ROC-AUC": round(
                            roc_auc_score(other_bundle["y_test"], other_bundle["probabilities"]), 4
                        ),
                        "Recall": round(other_scored["recall"], 4),
                        "Precision": round(other_scored["precision"], 4),
                        "Missed": other_scored["fn"],
                        "False alarms": other_scored["fp"],
                    },
                ]
            )
            st.dataframe(comparison, width="stretch", hide_index=True)
            st.caption(f"Both evaluated on the same held-out cases at a cut-off of {threshold:.2f}.")

    # ---------------------------------------------------------------- case
    with tab_case:
        ui.section(
            "Score an individual case",
            "A biopsy is loaded from the held-out set. Adjust the measurements that matter "
            "most and watch the score respond.",
        )

        importance = models.classifier_importance(name, test_fraction, use_ratios, 10)
        adjustable = importance.head(6)["feature"].tolist()

        x_test = bundle["x_test"]
        if "case_index" not in st.session_state:
            st.session_state.case_index = 0
            st.session_state.case_nonce = 0

        controls = st.columns([1, 1, 2])
        with controls[0]:
            if st.button("Load another biopsy", width="stretch"):
                generator = np.random.default_rng()
                st.session_state.case_index = int(generator.integers(0, len(x_test)))
                st.session_state.case_nonce += 1
        with controls[1]:
            if st.button("Reset measurements", width="stretch"):
                st.session_state.case_nonce += 1

        index = min(st.session_state.case_index, len(x_test) - 1)
        case = x_test.iloc[index].copy()
        actual = int(y_test[index])
        nonce = st.session_state.case_nonce

        slider_columns = st.columns(3)
        for position, feature in enumerate(adjustable):
            low = float(x_test[feature].min())
            high = float(x_test[feature].max())
            span = high - low
            with slider_columns[position % 3]:
                case[feature] = st.slider(
                    feature.replace("_", " "),
                    min_value=float(low - 0.05 * span),
                    max_value=float(high + 0.05 * span),
                    value=float(case[feature]),
                    step=float(span / 200) if span else 0.01,
                    key=f"case_{feature}_{nonce}",
                )

        probability = float(bundle["model"].predict_proba(case.to_frame().T)[0, 1])
        verdict = "malignant" if probability >= threshold else "benign"

        gauge_left, gauge_right = st.columns([1, 1], gap="large")
        with gauge_left:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number=dict(
                        suffix="%", font=dict(size=40, family=theme.FONT_MONO, color=theme.TYPE)
                    ),
                    gauge=dict(
                        axis=dict(
                            range=[0, 100],
                            tickfont=dict(size=10, color=theme.TYPE_FAINT, family=theme.FONT_MONO),
                            tickwidth=1, tickcolor=theme.RULE,
                        ),
                        bar=dict(color=colour, thickness=0.7),
                        bgcolor=theme.SURFACE,
                        borderwidth=1, bordercolor=theme.RULE_SOFT,
                        steps=[
                            dict(range=[0, threshold * 100], color=theme.rgba("#A5B4FC", 0.10)),
                            dict(range=[threshold * 100, 100], color=theme.rgba(colour, 0.10)),
                        ],
                        threshold=dict(
                            line=dict(color=theme.NEGATIVE, width=3),
                            thickness=0.85, value=threshold * 100,
                        ),
                    ),
                )
            )
            st.plotly_chart(
                theme.style(gauge, height=280, legend=False, margin=(20, 20, 10, 10)),
                width="stretch",
            )
            st.caption("Probability of malignancy. The red mark is your cut-off.")

        with gauge_right:
            ui.spacer(0.8)
            ui.readouts(
                [
                    {"label": "Model says", "value": verdict.title(), "accent": True},
                    {"label": "Recorded diagnosis",
                     "value": "Malignant" if actual else "Benign"},
                    {"label": "Agreement",
                     "value": "Match" if (probability >= threshold) == bool(actual) else "Mismatch",
                     "tone": "up" if (probability >= threshold) == bool(actual) else "down",
                     "delta": "against the held-out label"},
                ],
                colour,
            )
            st.caption(
                "Moving a slider produces a case that may not resemble any real biopsy. "
                "The measurements in this dataset are strongly correlated, so changing one "
                "in isolation creates a combination the model has never seen."
            )

        ui.spacer(0.5)
        ui.caveat(
            "<b>This is a teaching exercise, not a diagnostic tool.</b> The model is trained "
            "on a public research dataset of 569 biopsies collected decades ago at a single "
            "institution. It has no regulatory approval, has never been validated on current "
            "clinical practice, and must not inform any real decision about any real person. "
            "Its value here is in showing how a threshold converts a probability into a "
            "decision, and what that conversion costs."
        )
