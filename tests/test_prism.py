"""Tests for Prism.

Run from the project root:

    pytest -q

The slowest test trains the convolutional network end to end and takes a few
seconds. Everything else runs in well under a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism import datasets, models, theme
from prism.convnet import (
    ConvNet,
    TrainConfig,
    col2im,
    im2col,
    softmax,
    softmax_cross_entropy,
    train,
)
from prism.modules.digits import _paint, _prepare_canvas, GRID


def raw(function):
    """Call a Streamlit-cached function without needing a running app."""
    return getattr(function, "__wrapped__", function)


# ==========================================================================
# the convolutional network
# ==========================================================================
class TestConvNet:
    def test_im2col_round_trip_accumulates_overlaps(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2, 3, 8, 8))
        columns = im2col(x, 3, 3, pad=1)
        assert columns.shape == (2 * 8 * 8, 3 * 3 * 3)

        # Sending a column of ones back through must count how many windows
        # each input pixel took part in.
        restored = col2im(np.ones_like(columns), x.shape, 3, 3, pad=1)
        assert restored.shape == x.shape
        assert restored[0, 0, 0, 0] == pytest.approx(4.0)   # corner: 4 windows
        assert restored[0, 0, 4, 4] == pytest.approx(9.0)   # interior: 9 windows

    def test_softmax_rows_sum_to_one_and_survive_large_inputs(self):
        logits = np.array([[1.0, 2.0, 3.0], [900.0, 901.0, 902.0]])
        probabilities = softmax(logits)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.isfinite(probabilities).all()

    def test_loss_falls_as_predictions_improve(self):
        labels = np.array([0, 1])
        confident = np.array([[12.0, 0.0], [0.0, 12.0]])
        wrong = np.array([[0.0, 12.0], [12.0, 0.0]])
        assert softmax_cross_entropy(confident, labels)[0] < softmax_cross_entropy(wrong, labels)[0]

    @pytest.mark.parametrize("layer_name", ["conv1", "conv2", "fc1", "fc2"])
    def test_gradients_match_numerical_estimates(self, layer_name):
        """The central check on a hand-written network.

        Every analytic gradient is compared against a finite-difference
        estimate. If backpropagation were wrong anywhere, the two would part
        company here rather than quietly producing a worse model.
        """
        rng = np.random.default_rng(0)
        config = TrainConfig(
            conv1_filters=4, conv2_filters=6, dense_units=12, dropout=0.0, seed=7
        )
        network = ConvNet(config)

        x = rng.normal(size=(5, 1, 8, 8))
        y = rng.integers(0, 10, size=5)

        _, gradient = softmax_cross_entropy(network.forward(x, training=True), y)
        network.backward(gradient)

        step = 1e-6
        layer = getattr(network, layer_name)
        for _, parameter, analytic in layer.params():
            flat, flat_gradient = parameter.ravel(), analytic.ravel()
            for index in rng.choice(flat.size, size=min(10, flat.size), replace=False):
                original = flat[index]

                flat[index] = original + step
                up = softmax_cross_entropy(network.forward(x), y)[0]
                flat[index] = original - step
                down = softmax_cross_entropy(network.forward(x), y)[0]
                flat[index] = original

                numerical = (up - down) / (2 * step)
                scale = max(abs(numerical), abs(flat_gradient[index]), 1e-8)
                assert abs(numerical - flat_gradient[index]) / scale < 1e-5

    def test_dropout_only_acts_while_training(self):
        network = ConvNet(TrainConfig(dropout=0.5, seed=1))
        x = np.random.default_rng(2).random((16, 1, 8, 8))
        assert np.allclose(network.forward(x, training=False), network.forward(x, training=False))

    def test_snapshot_restores_exact_weights(self):
        network = ConvNet(TrainConfig(seed=3))
        saved = network.snapshot()
        network.conv1.W += 1.0
        network.restore(saved)
        assert np.allclose(network.conv1.W, saved[0])

    def test_pooling_rejects_odd_dimensions(self):
        with pytest.raises(ValueError):
            ConvNet(TrainConfig(), image_size=6)

    @pytest.mark.slow
    def test_trains_to_useful_accuracy(self):
        from prism.modules.digits import _splits

        data = raw(_splits)()
        config = TrainConfig(epochs=15, seed=42)
        network = ConvNet(config)
        history, best_epoch = train(
            network,
            data["x_train"], data["y_train"],
            data["x_val"], data["y_val"],
            config,
        )

        accuracy = float((network.predict(data["x_test"]) == data["y_test"]).mean())
        assert accuracy > 0.95, f"only reached {accuracy:.3f}"
        assert history[-1].train_loss < history[0].train_loss
        assert 1 <= best_epoch <= len(history)

    def test_early_stopping_reports_epochs_actually_run(self):
        rng = np.random.default_rng(5)
        x = rng.random((60, 1, 8, 8))
        y = rng.integers(0, 10, size=60)
        config = TrainConfig(epochs=30, patience=2, conv1_filters=4, conv2_filters=4, dense_units=8)
        history, _ = train(ConvNet(config), x, y, x[:20], y[:20], config)
        assert 1 <= len(history) <= 30


# ==========================================================================
# the drawing canvas
# ==========================================================================
class TestCanvas:
    def test_paint_marks_the_cell_and_only_bleeds_onto_neighbours(self):
        canvas = _paint(np.zeros((GRID, GRID)), 4, 4)
        assert canvas[4, 4] == 16.0
        assert 0 < canvas[3, 4] < 16.0
        assert canvas[3, 3] == 0.0          # diagonals stay clean
        assert canvas[0, 0] == 0.0

    def test_paint_stays_inside_the_grid(self):
        for row, col in [(0, 0), (0, GRID - 1), (GRID - 1, 0), (GRID - 1, GRID - 1)]:
            canvas = _paint(np.zeros((GRID, GRID)), row, col)
            assert canvas[row, col] == 16.0
            assert np.isfinite(canvas).all()

    def test_empty_canvas_passes_through_untouched(self):
        blank = np.zeros((GRID, GRID))
        assert np.array_equal(_prepare_canvas(blank), blank)

    def test_centring_moves_a_corner_drawing_to_the_middle(self):
        canvas = np.zeros((GRID, GRID))
        canvas[0:2, 0:2] = 16.0
        centred = _prepare_canvas(canvas)

        rows, cols = np.where(centred > 0)
        assert centred.sum() == canvas.sum()          # no ink invented or lost
        assert abs((rows.min() + rows.max()) / 2 - (GRID - 1) / 2) <= 0.5
        assert abs((cols.min() + cols.max()) / 2 - (GRID - 1) / 2) <= 0.5

    def test_already_centred_drawing_is_left_alone(self):
        canvas = np.zeros((GRID, GRID))
        canvas[3:5, 3:5] = 16.0
        assert np.array_equal(_prepare_canvas(canvas), canvas)


# ==========================================================================
# the cleaning pipelines
# ==========================================================================
class TestAtmosphericPipeline:
    @pytest.mark.parametrize("frequency", ["Daily", "Weekly", "Monthly"])
    def test_output_satisfies_every_invariant(self, frequency):
        clean, report = raw(datasets.clean_co2)(frequency=frequency)

        assert clean["date"].notna().all()
        assert clean["co2_ppm"].notna().all()
        assert (clean["co2_ppm"] > 0).all()
        assert not clean["date"].duplicated().any()
        assert clean["date"].is_monotonic_increasing
        assert report["missing_after_fill"] == 0
        assert report["rows_final"] == len(clean)

    def test_readings_stay_physically_plausible(self):
        clean, _ = raw(datasets.clean_co2)(frequency="Weekly")
        assert 300 < clean["co2_ppm"].min() < 330
        assert 400 < clean["co2_ppm"].max() < 460

    def test_injected_faults_are_caught_not_passed_through(self):
        clean, report = raw(datasets.clean_co2)(frequency="Weekly", inject_faults=True)

        assert report["rows_without_date"] > 0
        assert report["non_positive_readings"] > 0
        assert report["duplicate_dates"] > 0
        # The series still has to come out clean on the other side.
        assert clean["co2_ppm"].notna().all()
        assert (clean["co2_ppm"] > 0).all()
        assert not clean["date"].duplicated().any()

    def test_filled_points_are_flagged_so_charts_can_mark_them(self):
        clean, report = raw(datasets.clean_co2)(frequency="Weekly")
        assert int(clean["was_filled"].sum()) == report["values_synthesised"]
        assert report["values_synthesised"] > 0

    def test_skipping_interpolation_leaves_no_synthesised_points(self):
        clean, _ = raw(datasets.clean_co2)(frequency="Weekly", interpolation="none")
        assert not clean["was_filled"].any()

    def test_tighter_outlier_fence_flags_at_least_as_many(self):
        _, loose = raw(datasets.clean_co2)(frequency="Weekly", iqr_multiplier=3.0)
        _, tight = raw(datasets.clean_co2)(frequency="Weekly", iqr_multiplier=1.0)
        assert tight["outliers_flagged"] >= loose["outliers_flagged"]

    def test_decomposition_reconstructs_the_original_series(self):
        parts = raw(datasets.decompose_co2)("Weekly")
        usable = parts.dropna(subset=["trend"])
        rebuilt = usable["trend"] + usable["seasonal"] + usable["residual"]
        assert np.allclose(rebuilt, usable["co2_ppm"])

    def test_seasonal_cycle_has_the_expected_shape(self):
        parts = raw(datasets.decompose_co2)("Weekly")
        monthly = parts.groupby("month")["seasonal"].mean()
        # Northern spring peak, autumn trough, a few ppm either side.
        assert monthly.idxmax() in (4, 5, 6)
        assert monthly.idxmin() in (9, 10, 11)
        assert 3 < (monthly.max() - monthly.min()) < 12


class TestWineQualityPipeline:
    def test_cleaning_removes_duplicates_and_keeps_scores_in_range(self):
        frame, report = raw(datasets.load_wine_quality)()
        assert report["duplicate_rows"] > 0
        assert report["rows_final"] == len(frame)
        assert not frame.duplicated().any()
        assert frame.isna().sum().sum() == 0
        assert frame["quality"].between(0, 10).all()

    def test_every_expected_column_survives(self):
        frame, _ = raw(datasets.load_wine_quality)()
        for column in datasets.QUALITY_FEATURES + ["quality"]:
            assert column in frame.columns


class TestOtherDatasets:
    def test_cultivars_load_with_labels_held_separately(self):
        features, truth = raw(datasets.load_cultivars)()
        assert features.shape == (178, 13)
        assert len(truth) == 178
        assert set(truth.unique()) == set(datasets.CULTIVAR_LABELS.values())

    def test_malignant_is_the_positive_class(self):
        features, labels = raw(datasets.load_biopsies)()
        assert features.shape == (569, 30)
        # The dataset holds more benign cases than malignant ones.
        assert 0.3 < labels.mean() < 0.45

    def test_ratio_features_add_columns_without_touching_the_originals(self):
        features, _ = raw(datasets.load_biopsies)()
        widened = datasets.add_ratio_features(features)
        assert widened.shape[1] == features.shape[1] + len(datasets.RATIO_FEATURES)
        assert np.isfinite(widened[datasets.RATIO_FEATURES].to_numpy()).all()
        assert widened[features.columns].equals(features)

    def test_digit_images_have_the_expected_shape_and_range(self):
        images, labels = raw(datasets.load_digit_images)()
        assert images.shape == (1797, 8, 8)
        assert images.min() >= 0 and images.max() <= 16
        assert set(np.unique(labels)) == set(range(10))


# ==========================================================================
# the models
# ==========================================================================
class TestClustering:
    def test_labels_cover_every_sample_and_score_sensibly(self):
        features, _ = raw(datasets.load_cultivars)()
        result = raw(models.cluster_cultivars)(tuple(features.columns), 3)

        assert len(result["labels"]) == 178
        assert len(np.unique(result["labels"])) == 3
        assert -1 <= result["scores"]["silhouette"] <= 1
        # Three real cultivars are recoverable from chemistry alone.
        assert result["scores"]["adjusted_rand"] > 0.7

    def test_projection_never_exceeds_the_available_dimensions(self):
        result = raw(models.cluster_cultivars)(("alcohol", "proline"), 2)
        assert result["coords"].shape[1] == 2

    def test_standardising_changes_the_outcome(self):
        features, _ = raw(datasets.load_cultivars)()
        columns = tuple(features.columns)
        scaled = raw(models.cluster_cultivars)(columns, 3, standardise=True)
        unscaled = raw(models.cluster_cultivars)(columns, 3, standardise=False)
        assert scaled["scores"]["adjusted_rand"] > unscaled["scores"]["adjusted_rand"]


class TestClassification:
    def test_split_is_held_out_and_probabilities_are_valid(self):
        bundle = raw(models.fit_classifier)("Logistic regression", 0.25, True)
        assert len(bundle["y_test"]) == pytest.approx(569 * 0.25, abs=2)
        assert set(bundle["x_train"].index).isdisjoint(bundle["x_test"].index)
        assert ((bundle["probabilities"] >= 0) & (bundle["probabilities"] <= 1)).all()

    def test_threshold_counts_add_up_to_the_test_set(self):
        bundle = raw(models.fit_classifier)("Logistic regression", 0.25, True)
        scored = models.threshold_metrics(bundle["y_test"], bundle["probabilities"], 0.5)
        assert scored["tp"] + scored["tn"] + scored["fp"] + scored["fn"] == len(bundle["y_test"])
        assert 0 <= scored["precision"] <= 1
        assert 0 <= scored["recall"] <= 1

    def test_lowering_the_threshold_never_reduces_recall(self):
        bundle = raw(models.fit_classifier)("Logistic regression", 0.25, True)
        recalls = [
            models.threshold_metrics(bundle["y_test"], bundle["probabilities"], t)["recall"]
            for t in (0.1, 0.3, 0.5, 0.7, 0.9)
        ]
        assert all(earlier >= later for earlier, later in zip(recalls, recalls[1:]))

    def test_extreme_thresholds_behave(self):
        bundle = raw(models.fit_classifier)("Logistic regression", 0.25, True)
        everything = models.threshold_metrics(bundle["y_test"], bundle["probabilities"], 0.0)
        nothing = models.threshold_metrics(bundle["y_test"], bundle["probabilities"], 1.01)
        assert everything["fn"] == 0
        assert nothing["tp"] == 0

    def test_model_beats_chance_by_a_wide_margin(self):
        from sklearn.metrics import roc_auc_score

        bundle = raw(models.fit_classifier)("Logistic regression", 0.25, True)
        assert roc_auc_score(bundle["y_test"], bundle["probabilities"]) > 0.95


class TestRegression:
    def test_every_model_is_scored_and_the_best_beats_the_baseline(self):
        bundle = raw(models.fit_quality_models)(0.2)
        results = bundle["results"]

        assert set(results["model"]) == set(models.REGRESSORS)
        baseline = results.loc[results["model"] == "Baseline mean", "rmse"].iloc[0]
        assert results["rmse"].min() < baseline
        assert bundle["best"] != "Baseline mean"

    def test_baseline_explains_nothing(self):
        bundle = raw(models.fit_quality_models)(0.2)
        baseline = bundle["results"].loc[bundle["results"]["model"] == "Baseline mean"].iloc[0]
        assert baseline["r2"] < 0.01

    def test_segments_partition_every_wine(self):
        frame, _ = raw(datasets.load_wine_quality)()
        result = raw(models.segment_wines)(3)
        assert len(result["labels"]) == len(frame)
        assert len(np.unique(result["labels"])) == 3
        assert int(result["summary"]["wines"].sum()) == len(frame)

    def test_contributions_are_measured_against_a_median_wine(self):
        frame, _ = raw(datasets.load_wine_quality)()
        bundle = raw(models.fit_quality_models)(0.2)
        model = bundle["fitted"][bundle["best"]]
        medians = frame[datasets.QUALITY_FEATURES].median()

        table = models.median_contributions(model, medians, medians)
        # A wine already at every median differs from itself by nothing.
        assert np.allclose(table["contribution"], 0.0, atol=1e-9)

        sample = frame[datasets.QUALITY_FEATURES].iloc[0]
        table = models.median_contributions(model, sample, medians)
        assert len(table) == len(datasets.QUALITY_FEATURES)
        assert table["magnitude"].is_monotonic_decreasing


# ==========================================================================
# presentation helpers
# ==========================================================================
class TestTheme:
    def test_every_module_has_a_distinct_spectral_colour(self):
        keys = [module["key"] for module in theme.MODULES]
        assert len(keys) == len(set(keys)) == 6
        colours = [theme.accent(key) for key in keys]
        assert len(set(colours)) == 6
        assert all(colour.startswith("#") and len(colour) == 7 for colour in colours)

    def test_modules_carry_the_text_the_pages_expect(self):
        for module in theme.MODULES:
            assert module["title"] and module["blurb"]
            assert module["source"] and module["shape"]
            assert len(module["methods"]) >= 3

    def test_fade_interpolates_between_the_field_and_the_colour(self):
        colour = theme.SPECTRUM["digits"]
        assert theme.fade(colour, 0.0).lower() == theme.PLATE.lower()
        assert theme.fade(colour, 1.0).lower() == colour.lower()

    def test_rgba_emits_valid_css(self):
        assert theme.rgba("#818CF8", 0.5) == "rgba(129,140,248,0.5)"
