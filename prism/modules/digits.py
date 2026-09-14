"""Digit Recognition - a convolutional network with no framework underneath it."""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

from .. import datasets, theme, ui
from ..convnet import ConvNet, TrainConfig, train

GRID = 8


# --------------------------------------------------------------------------
# data preparation
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _splits() -> dict:
    """Three-way split: the model never sees the test set until it is scored."""
    images, labels = datasets.load_digit_images()
    scaled = (images / 16.0)[:, None, :, :]

    x_train, x_rest, y_train, y_rest, i_train, i_rest = train_test_split(
        scaled, labels, np.arange(len(labels)),
        test_size=0.30, random_state=42, stratify=labels,
    )
    x_val, x_test, y_val, y_test, i_val, i_test = train_test_split(
        x_rest, y_rest, i_rest, test_size=0.50, random_state=42, stratify=y_rest
    )
    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "raw_test": images[i_test],
    }


# --------------------------------------------------------------------------
# small renderers
# --------------------------------------------------------------------------
def _signed_tile(matrix: np.ndarray, size: int = 62) -> str:
    """A kernel drawn as squares: warm for positive weights, cool for negative."""
    peak = float(np.abs(matrix).max()) or 1.0
    rows, cols = matrix.shape
    cell = size / cols
    squares = []
    for r in range(rows):
        for c in range(cols):
            value = float(matrix[r, c]) / peak
            base = theme.SPECTRUM["digits"] if value >= 0 else "#6C8CFF"
            squares.append(
                f'<rect x="{c * cell:.2f}" y="{r * cell:.2f}" width="{cell:.2f}" '
                f'height="{cell:.2f}" fill="{theme.fade(base, abs(value))}" />'
            )
    return (
        f'<svg viewBox="0 0 {cols * cell:.1f} {rows * cell:.1f}" width="{size}" '
        f'height="{size * rows / cols:.0f}" shape-rendering="crispEdges" '
        f'style="border:1px solid {theme.RULE_SOFT};border-radius:3px;">'
        f'{"".join(squares)}</svg>'
    )


def _tile_row(tiles: list[str], captions: list[str] | None = None) -> str:
    cells = []
    for position, tile in enumerate(tiles):
        caption = ""
        if captions:
            caption = (
                f'<div style="font-family:{theme.FONT_MONO};font-size:.62rem;'
                f'color:{theme.TYPE_FAINT};text-align:center;margin-top:.22rem;">'
                f"{escape(captions[position])}</div>"
            )
        cells.append(f'<div style="text-align:center;">{tile}{caption}</div>')
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:.5rem;margin:.4rem 0 1rem 0;">'
        + "".join(cells)
        + "</div>"
    )


def _probability_bar(probabilities: np.ndarray, colour: str) -> go.Figure:
    order = np.arange(10)
    tints = [
        colour if d == int(np.argmax(probabilities)) else theme.rgba(colour, 0.28) for d in order
    ]
    figure = go.Figure(
        go.Bar(
            x=[str(d) for d in order], y=probabilities,
            marker_color=tints, marker_line_width=0,
            hovertemplate="Digit %{x}<br>%{y:.1%}<extra></extra>",
        )
    )
    figure.update_yaxes(title_text="Confidence", range=[0, 1], tickformat=".0%")
    figure.update_xaxes(title_text="Digit")
    return figure


def _curve_figure(records: list, colour: str) -> go.Figure:
    history = pd.DataFrame(
        [
            {
                "epoch": r.epoch,
                "train_loss": r.train_loss, "val_loss": r.val_loss,
                "train_accuracy": r.train_accuracy, "val_accuracy": r.val_accuracy,
            }
            for r in records
        ]
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history["epoch"], y=history["train_loss"], name="Training loss",
            mode="lines", line=dict(color=theme.rgba(colour, 0.45), width=2),
            hovertemplate="Epoch %{x}<br>loss %{y:.4f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=history["epoch"], y=history["val_loss"], name="Validation loss",
            mode="lines", line=dict(color=colour, width=2.6),
            hovertemplate="Epoch %{x}<br>loss %{y:.4f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=history["epoch"], y=history["val_accuracy"], name="Validation accuracy",
            mode="lines", line=dict(color="#A5B4FC", width=2, dash="dot"), yaxis="y2",
            hovertemplate="Epoch %{x}<br>accuracy %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis=dict(title="Epoch", dtick=1 if len(history) <= 15 else 5),
        yaxis=dict(title="Cross-entropy loss"),
        yaxis2=dict(
            title="Accuracy", overlaying="y", side="right", range=[0, 1.02],
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(size=10, color=theme.TYPE_FAINT, family=theme.FONT_MONO),
            title_font=dict(size=11, color=theme.TYPE_FAINT),
        ),
    )
    return figure


# --------------------------------------------------------------------------
# canvas preparation
# --------------------------------------------------------------------------
def _prepare_canvas(canvas: np.ndarray) -> np.ndarray:
    """Slide a drawing so its ink sits centred, as the training images do.

    The shift is measured from the bounding box of the ink rather than its
    centre of mass. Centre of mass is pulled around by where a digit happens to
    be heaviest - the bar of a 7, the loop of a 6 - and measuring it that way
    cost about six points of accuracy on real images during testing.

    Only position is corrected. Nothing is smoothed, rescaled or sharpened, so
    a drawing that the network reads badly is being read badly on its merits.
    """
    image = np.asarray(canvas, dtype=float)
    ink = image > 0
    if not ink.any():
        return image

    rows, cols = np.where(ink)
    shift_row = int(round((GRID - 1) / 2 - (rows.min() + rows.max()) / 2))
    shift_col = int(round((GRID - 1) / 2 - (cols.min() + cols.max()) / 2))
    if shift_row == 0 and shift_col == 0:
        return image

    moved = np.zeros_like(image)
    for r in range(GRID):
        for c in range(GRID):
            tr, tc = r + shift_row, c + shift_col
            if 0 <= tr < GRID and 0 <= tc < GRID:
                moved[tr, tc] = image[r, c]
    return moved


# A click lays down one solid cell and only a trace on its neighbours. Heavier
# bleed was tried first and read far worse: fattening a stroke turns it into a
# shape the network was never trained on.
BRUSH_BLEED = 2.0


def _paint(canvas: np.ndarray, row: int, col: int) -> np.ndarray:
    """Lay down ink with a soft edge, echoing the anti-aliasing in the data."""
    painted = canvas.copy()
    painted[row, col] = 16.0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < GRID and 0 <= nc < GRID:
            painted[nr, nc] = min(16.0, painted[nr, nc] + BRUSH_BLEED)
    return painted


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------
def render() -> None:
    colour = ui.page_header("digits")
    data = _splits()

    # ------------------------------------------------------------- controls
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            conv1 = st.select_slider("First layer filters", [8, 16, 24, 32], value=16)
            conv2 = st.select_slider("Second layer filters", [16, 32, 48, 64], value=32)
        with c2:
            dense = st.select_slider("Hidden units", [32, 64, 96, 128], value=64)
            dropout = st.slider("Dropout", 0.0, 0.6, 0.25, 0.05)
        with c3:
            learning_rate = st.select_slider(
                "Learning rate", [0.0003, 0.001, 0.003, 0.01], value=0.003,
                format_func=lambda v: f"{v:g}",
            )
            batch_size = st.select_slider("Batch size", [32, 64, 128], value=64)
        with c4:
            epochs = st.slider("Maximum epochs", 5, 40, 20, 1)
            seed = st.number_input("Random seed", 0, 9999, 42, 1)

    config = TrainConfig(
        conv1_filters=int(conv1), conv2_filters=int(conv2), dense_units=int(dense),
        dropout=float(dropout), learning_rate=float(learning_rate),
        batch_size=int(batch_size), epochs=int(epochs), seed=int(seed),
    )
    signature = (
        config.conv1_filters, config.conv2_filters, config.dense_units,
        config.dropout, config.learning_rate, config.batch_size,
        config.epochs, config.seed,
    )

    run = st.session_state.get("digit_run")
    stale = run is not None and run["signature"] != signature

    action = st.columns([1, 1, 2])
    with action[0]:
        asked = st.button(
            "Retrain with these settings" if run else "Train the network",
            width="stretch",
            type="primary" if stale else "secondary",
        )
    with action[1]:
        if stale:
            st.caption("Settings changed since the last run.")
        elif run:
            st.caption(f"Trained for {run['epochs_run']} epochs.")

    # ------------------------------------------------------------- training
    if asked or run is None:
        curve_slot = st.empty()
        progress = st.progress(0.0, text="Setting up the network")
        records: list = []

        def on_epoch(record) -> None:
            records.append(record)
            progress.progress(
                min(record.epoch / config.epochs, 1.0),
                text=f"Epoch {record.epoch} of {config.epochs} \u2014 "
                     f"validation accuracy {record.val_accuracy:.3f}",
            )
            curve_slot.plotly_chart(
                theme.style(_curve_figure(records, colour), height=330),
                width="stretch",
            )

        network = ConvNet(config)
        history, best_epoch = train(
            network,
            data["x_train"], data["y_train"],
            data["x_val"], data["y_val"],
            config, on_epoch=on_epoch,
        )
        progress.empty()
        curve_slot.empty()

        probabilities = network.predict_proba(data["x_test"])
        run = {
            "signature": signature,
            "network": network,
            "history": history,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "probabilities": probabilities,
            "predictions": probabilities.argmax(axis=1),
            "parameters": network.parameter_count(),
        }
        st.session_state["digit_run"] = run

    network = run["network"]
    predictions = run["predictions"]
    probabilities = run["probabilities"]
    y_test = data["y_test"]
    accuracy = float((predictions == y_test).mean())

    # ------------------------------------------------------------- readouts
    final = run["history"][-1]
    ui.readouts(
        [
            {"label": "Test accuracy", "value": f"{accuracy:.4f}", "accent": True,
             "delta": f"{int((predictions == y_test).sum())} of {len(y_test)} correct",
             "tone": "flat"},
            {"label": "Validation accuracy", "value": f"{final.val_accuracy:.4f}"},
            {"label": "Trainable parameters", "value": f"{run['parameters']:,}"},
            {"label": "Epochs run", "value": f"{run['epochs_run']}",
             "delta": f"best at {run['best_epoch']}", "tone": "flat"},
            {"label": "Framework dependencies", "value": "0", "delta": "NumPy only", "tone": "flat"},
        ],
        colour,
    )

    tab_learning, tab_errors, tab_inside, tab_draw = st.tabs(
        ["How it learned", "Where it fails", "Inside the network", "Draw a digit"]
    )

    # ------------------------------------------------------------- learning
    with tab_learning:
        ui.section(
            "Learning curves",
            "Training stops early once validation loss stops improving, and the weights "
            "from the best epoch are restored rather than the last.",
        )
        st.plotly_chart(
            theme.style(_curve_figure(run["history"], colour), height=380), width="stretch"
        )

        gap = final.train_accuracy - final.val_accuracy
        ui.finding(
            f"The run stopped after <b class='num'>{run['epochs_run']}</b> epochs with the best "
            f"validation loss at epoch <b class='num'>{run['best_epoch']}</b>. Training accuracy "
            f"finished <b class='num'>{gap:+.3f}</b> above validation accuracy. A small positive "
            f"gap is normal and healthy; a large one would mean the network is memorising the "
            f"training images rather than learning the shapes of digits, which is what the "
            f"dropout layer exists to discourage.",
            colour,
        )

        ui.section("The architecture", "Each stage halves the picture and doubles the detail.")
        stages = [
            ("Input", "1 \u00d7 8 \u00d7 8", "The raw writing grid."),
            ("Convolution", f"{config.conv1_filters} \u00d7 8 \u00d7 8", "3\u00d73 filters look for strokes and edges."),
            ("Pooling", f"{config.conv1_filters} \u00d7 4 \u00d7 4", "Keeps the strongest response in each 2\u00d72 block."),
            ("Convolution", f"{config.conv2_filters} \u00d7 4 \u00d7 4", "Combines strokes into corners and loops."),
            ("Pooling", f"{config.conv2_filters} \u00d7 2 \u00d7 2", "Halves the grid again."),
            ("Dense", f"{config.dense_units} units", f"Dropout at {config.dropout:.0%} during training."),
            ("Output", "10 scores", "Softmax turns the scores into probabilities."),
        ]
        cards = []
        for name, shape, note in stages:
            cards.append(
                f'<div style="flex:1 1 130px;min-width:130px;background:{theme.SURFACE};'
                f'border:1px solid {theme.RULE_SOFT};border-top:2px solid {colour};'
                f'border-radius:6px;padding:.7rem .8rem;">'
                f'<div style="font-size:.82rem;font-weight:600;color:{theme.TYPE};">{escape(name)}</div>'
                f'<div style="font-family:{theme.FONT_MONO};font-size:.72rem;color:{colour};'
                f'margin:.2rem 0 .35rem 0;">{shape}</div>'
                f'<div style="font-size:.71rem;color:{theme.TYPE_FAINT};line-height:1.45;">'
                f"{escape(note)}</div></div>"
            )
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:.5rem;margin:.3rem 0 1rem 0;">'
            + "".join(cards)
            + "</div>",
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------- errors
    with tab_errors:
        ui.section(
            "Which digits get confused",
            "Rows are the true digit, columns what the network answered. Everything off the "
            "diagonal is a mistake.",
        )

        matrix = np.zeros((10, 10), dtype=int)
        for actual, predicted in zip(y_test, predictions):
            matrix[actual, predicted] += 1

        heat = go.Figure(
            go.Heatmap(
                z=matrix,
                x=[str(d) for d in range(10)], y=[str(d) for d in range(10)],
                colorscale=theme.sequential(colour),
                text=np.where(matrix > 0, matrix.astype(object), ""),
                texttemplate="%{text}",
                textfont=dict(size=11, family=theme.FONT_MONO, color=theme.TYPE),
                hovertemplate="True %{y} \u2192 said %{x}<br>%{z} cases<extra></extra>",
                showscale=False,
            )
        )
        heat.update_xaxes(title_text="Predicted")
        heat.update_yaxes(title_text="Actual", autorange="reversed")
        st.plotly_chart(theme.style(heat, height=430, legend=False), width="stretch")

        per_class = []
        for digit in range(10):
            true_positive = matrix[digit, digit]
            support = matrix[digit].sum()
            called = matrix[:, digit].sum()
            recall = true_positive / support if support else 0.0
            precision = true_positive / called if called else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            per_class.append(
                {"digit": digit, "precision": precision, "recall": recall,
                 "f1": f1, "support": int(support)}
            )
        per_class_frame = pd.DataFrame(per_class)

        bars = go.Figure()
        for column, label, tint in [
            ("precision", "Precision", theme.rgba(colour, 0.55)),
            ("recall", "Recall", colour),
        ]:
            bars.add_trace(
                go.Bar(
                    x=per_class_frame["digit"].astype(str), y=per_class_frame[column],
                    name=label, marker_color=tint, marker_line_width=0,
                    hovertemplate=label + " %{y:.3f} for digit %{x}<extra></extra>",
                )
            )
        bars.update_layout(barmode="group")
        bars.update_yaxes(title_text="Score", range=[0, 1.05])
        bars.update_xaxes(title_text="Digit")
        st.plotly_chart(theme.style(bars, height=300), width="stretch")

        mistakes = np.flatnonzero(predictions != y_test)
        if len(mistakes):
            weakest = per_class_frame.loc[per_class_frame["recall"].idxmin()]
            ui.section(
                f"Every mistake it made ({len(mistakes)})",
                "The confidence beneath each image shows whether the network was unsure or "
                "confidently wrong. Confidently wrong is the more troubling kind.",
            )
            tiles, captions = [], []
            for index in mistakes[:12]:
                tiles.append(
                    ui.pixel_grid_svg(data["raw_test"][index], colour, size=74, vmax=16.0)
                )
                captions.append(
                    f"{y_test[index]}\u2192{predictions[index]} "
                    f"({probabilities[index].max():.0%})"
                )
            st.markdown(_tile_row(tiles, captions), unsafe_allow_html=True)

            ui.finding(
                f"<b class='num'>{len(mistakes)}</b> of <b class='num'>{len(y_test)}</b> test "
                f"images were misread. Digit <b class='num'>{int(weakest['digit'])}</b> has the "
                f"lowest recall at <b class='num'>{weakest['recall']:.3f}</b>. At eight pixels "
                f"square there is genuinely very little to go on, and several of these would "
                f"give a human pause too.",
                colour,
            )
        else:
            ui.finding("Every test image was classified correctly in this run.", colour)

    # --------------------------------------------------------------- inside
    with tab_inside:
        ui.section(
            "What the first layer learned to look for",
            "Each tile is one 3\u00d73 filter. Amber marks weights that respond to ink, blue "
            "marks weights that respond to its absence, so the pattern shows the edge or "
            "stroke each filter is tuned to.",
        )
        kernels = network.first_layer_kernels()
        st.markdown(
            _tile_row(
                [_signed_tile(kernels[i]) for i in range(len(kernels))],
                [f"f{i + 1}" for i in range(len(kernels))],
            ),
            unsafe_allow_html=True,
        )

        ui.section(
            "Following one image through",
            "The same digit after each stage. Detail falls away while the distinguishing "
            "shape survives.",
        )

        choice = st.selectbox(
            "Digit to trace",
            list(range(10)),
            index=3,
            format_func=lambda d: f"Digit {d}",
        )
        candidates = np.flatnonzero(y_test == choice)
        if len(candidates):
            index = int(candidates[0])
            source = data["x_test"][index, 0]
            maps = network.feature_maps(source)

            st.markdown(
                _tile_row(
                    [ui.pixel_grid_svg(data["raw_test"][index], colour, size=88, vmax=16.0)],
                    ["input"],
                ),
                unsafe_allow_html=True,
            )
            for stage, label in [
                ("conv1", "After the first convolution"),
                ("pool1", "After the first pooling"),
                ("conv2", "After the second convolution"),
            ]:
                activations = maps[stage]
                shown = activations[: min(12, len(activations))]
                peak = float(shown.max()) or 1.0
                st.markdown(
                    f'<div style="font-size:.8rem;color:{theme.TYPE_DIM};margin-top:.5rem;">'
                    f"{label} \u2014 {activations.shape[0]} maps of "
                    f"{activations.shape[1]}\u00d7{activations.shape[2]}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _tile_row(
                        [
                            ui.pixel_grid_svg(shown[i], colour, size=52, vmax=peak)
                            for i in range(len(shown))
                        ]
                    ),
                    unsafe_allow_html=True,
                )

            ui.finding(
                "Early maps still look like the digit. Later ones do not, and that is the "
                "point: the network is discarding position and stroke detail while keeping "
                "whatever separates this digit from the other nine. By the final pooling "
                "stage each map is 2\u00d72, which is four numbers per filter for the "
                "classifier to work with.",
                colour,
            )

    # ----------------------------------------------------------------- draw
    with tab_draw:
        ui.section(
            "Draw one yourself",
            "Click cells to lay down ink. Eight by eight is a cramped canvas, so aim for a "
            "thick, simple shape that fills most of the grid.",
        )

        if "canvas" not in st.session_state:
            st.session_state.canvas = np.zeros((GRID, GRID), dtype=float)

        pad_left, centre, pad_right = st.columns([0.6, 2.4, 1.6])

        with centre:
            for row in range(GRID):
                cells = st.columns(GRID, gap="small")
                for col in range(GRID):
                    inked = st.session_state.canvas[row, col] >= 8
                    with cells[col]:
                        if st.button(
                            "\u25cf" if inked else "\u00b7",
                            key=f"cell_{row}_{col}",
                            width="stretch",
                        ):
                            st.session_state.canvas = _paint(
                                st.session_state.canvas, row, col
                            )
                            st.rerun()

            buttons = st.columns(2)
            with buttons[0]:
                if st.button("Clear the canvas", width="stretch"):
                    st.session_state.canvas = np.zeros((GRID, GRID), dtype=float)
                    st.rerun()
            with buttons[1]:
                if st.button("Copy a test image in", width="stretch"):
                    generator = np.random.default_rng()
                    pick = int(generator.integers(0, len(data["raw_test"])))
                    st.session_state.canvas = data["raw_test"][pick].astype(float).copy()
                    st.rerun()

        with pad_right:
            canvas = st.session_state.canvas
            if canvas.sum() <= 0:
                st.markdown(
                    f'<div style="border:1px dashed {theme.RULE};border-radius:6px;'
                    f'padding:1.4rem 1rem;text-align:center;color:{theme.TYPE_FAINT};'
                    f'font-size:.83rem;line-height:1.55;">Nothing drawn yet.<br>'
                    f"Click a few cells to begin.</div>",
                    unsafe_allow_html=True,
                )
            else:
                centred = _prepare_canvas(canvas)
                scores = network.predict_proba((centred / 16.0)[None, None, :, :])[0]
                answer = int(np.argmax(scores))

                st.markdown(
                    f'<div style="font-size:.78rem;color:{theme.TYPE_FAINT};'
                    f'margin-bottom:.3rem;">What the network receives</div>'
                    + ui.pixel_grid_svg(centred, colour, size=120, vmax=16.0),
                    unsafe_allow_html=True,
                )
                ui.readouts(
                    [
                        {"label": "Reads it as", "value": str(answer), "accent": True},
                        {"label": "Confidence", "value": f"{scores[answer]:.1%}"},
                    ],
                    colour,
                )

        if st.session_state.canvas.sum() > 0:
            centred = _prepare_canvas(st.session_state.canvas)
            scores = network.predict_proba((centred / 16.0)[None, None, :, :])[0]
            st.plotly_chart(
                theme.style(_probability_bar(scores, colour), height=250, legend=False),
                width="stretch",
            )

            runner_up = int(np.argsort(scores)[-2])
            ui.finding(
                f"Second choice is <b class='num'>{runner_up}</b> at "
                f"<b class='num'>{scores[runner_up]:.1%}</b>. Drawings tend to score lower than "
                f"real samples, and the reason is worth noting: every training image was written "
                f"with a pen and downsampled, giving soft grey edges, while a drawing made of "
                f"clicked squares has hard ones. That mismatch between training data and live "
                f"input is one of the most common reasons a model that tested well disappoints "
                f"in practice.",
                colour,
            )

        ui.caveat(
            "<b>Built from scratch.</b> The convolution, pooling, dropout, softmax "
            "cross-entropy and Adam optimiser behind this prediction are written directly in "
            "NumPy, with no deep-learning framework installed. The backward pass is verified "
            "against numerical gradients in the test suite, which is the standard way to prove "
            "a hand-written network computes the gradients it claims to."
        )
