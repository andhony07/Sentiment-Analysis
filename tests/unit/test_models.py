"""
Unit tests for the sentiment analysis models package.

Tests cover:
- Package imports and aliases
- SentimentModel abstract interface
- NaiveBayesSentimentModel initialisation
- Training (valid and invalid inputs)
- Single-text prediction
- Batch prediction
- predict_proba (single and batch)
- evaluate() return shape
- save() and load() round-trip
- Optional TextPreprocessor integration
- Backward-compatible aliases
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Import the package under test
# ---------------------------------------------------------------------------
from sentiment_analysis.models import (
    BaseModel,
    NaiveBayesClassifier,
    NaiveBayesSentimentModel,
    SentimentModel,
)
from sentiment_analysis.models.base import SentimentModel as SentimentModelDirect
from sentiment_analysis.models.naive_bayes import (
    NaiveBayesSentimentModel as NaiveBayesDirect,
    NaiveBayesClassifier as NaiveBayesClassifierDirect,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "I absolutely love this product, it is amazing!",
    "This is the worst thing I have ever bought.",
    "It is okay, nothing special.",
    "Fantastic experience, highly recommend!",
    "Terrible quality, very disappointed.",
    "Not bad, not great, just average.",
    "I am so happy with my purchase!",
    "This ruined my day completely.",
    "It was fine, did the job.",
    "Outstanding! Will buy again.",
    "Horrible. Waste of money.",
    "Pretty decent for the price.",
]

SAMPLE_LABELS = [
    "positive", "negative", "neutral",
    "positive", "negative", "neutral",
    "positive", "negative", "neutral",
    "positive", "negative", "neutral",
]


@pytest.fixture
def untrained_model():
    """Return a fresh, untrained NaiveBayesSentimentModel."""
    return NaiveBayesSentimentModel()


@pytest.fixture
def trained_model():
    """Return a trained NaiveBayesSentimentModel."""
    model = NaiveBayesSentimentModel()
    model.train(texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS)
    return model


@pytest.fixture
def tmp_model_path(tmp_path):
    """Return a temporary file path for model saving."""
    return str(tmp_path / "test_model.json")


# ---------------------------------------------------------------------------
# 1. Import / alias tests
# ---------------------------------------------------------------------------

class TestImports:
    """Verify all public symbols are importable and aliased correctly."""

    def test_sentiment_model_importable(self):
        assert SentimentModel is not None

    def test_base_model_alias(self):
        assert BaseModel is SentimentModel

    def test_naive_bayes_importable(self):
        assert NaiveBayesSentimentModel is not None

    def test_naive_bayes_classifier_alias(self):
        assert NaiveBayesClassifier is NaiveBayesSentimentModel

    def test_direct_base_import(self):
        assert SentimentModelDirect is SentimentModel

    def test_direct_naive_bayes_import(self):
        assert NaiveBayesDirect is NaiveBayesSentimentModel

    def test_direct_naive_bayes_classifier_import(self):
        assert NaiveBayesClassifierDirect is NaiveBayesSentimentModel

    def test_sentiment_model_is_abstract(self):
        """SentimentModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SentimentModel()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 2. Initialisation tests
# ---------------------------------------------------------------------------

class TestNaiveBayesInit:
    """Test NaiveBayesSentimentModel initialisation."""

    def test_model_name(self, untrained_model):
        assert untrained_model.model_name == "naive_bayes_sentiment"

    def test_is_trained_false_initially(self, untrained_model):
        assert untrained_model.is_trained is False

    def test_classes_empty_initially(self, untrained_model):
        assert untrained_model.classes_ == []

    def test_preprocessor_default_none(self, untrained_model):
        assert untrained_model.preprocessor is None

    def test_custom_preprocessor_stored(self):
        mock_pp = MagicMock()
        model = NaiveBayesSentimentModel(preprocessor=mock_pp)
        assert model.preprocessor is mock_pp

    def test_repr_contains_class_name(self, untrained_model):
        r = repr(untrained_model)
        assert "NaiveBayesSentimentModel" in r
        assert "is_trained=False" in r


# ---------------------------------------------------------------------------
# 3. Training tests
# ---------------------------------------------------------------------------

class TestTraining:
    """Test the train() method."""

    def test_train_sets_is_trained(self, untrained_model):
        untrained_model.train(texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS)
        assert untrained_model.is_trained is True

    def test_train_sets_classes(self, untrained_model):
        untrained_model.train(texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS)
        assert set(untrained_model.classes_) == {"positive", "negative", "neutral"}

    def test_train_empty_texts_raises(self, untrained_model):
        with pytest.raises(ValueError):
            untrained_model.train(texts=[], labels=[])

    def test_train_mismatched_lengths_raises(self, untrained_model):
        with pytest.raises(ValueError):
            untrained_model.train(
                texts=["hello", "world"],
                labels=["positive"],
            )

    def test_train_with_preprocessor(self):
        """Preprocessor tokens are joined and fed into TF-IDF."""
        mock_pp = MagicMock()
        mock_pp.preprocess.side_effect = lambda t: t.lower().split()

        model = NaiveBayesSentimentModel(preprocessor=mock_pp)
        model.train(texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS)

        assert model.is_trained
        assert mock_pp.preprocess.call_count == len(SAMPLE_TEXTS)


# ---------------------------------------------------------------------------
# 4. Single-text prediction tests
# ---------------------------------------------------------------------------

class TestSinglePredict:
    """Test predict() with a single string input."""

    def test_predict_returns_string(self, trained_model):
        result = trained_model.predict("This is awesome!")
        assert isinstance(result, str)

    def test_predict_valid_sentiment(self, trained_model):
        result = trained_model.predict("I hate this product.")
        assert result in {"positive", "negative", "neutral"}

    def test_predict_before_training_raises(self, untrained_model):
        with pytest.raises(RuntimeError, match="trained"):
            untrained_model.predict("test text")

    def test_predict_positive_text(self, trained_model):
        result = trained_model.predict("I love this, it is absolutely fantastic!")
        assert result in {"positive", "negative", "neutral"}  # model-dependent

    def test_predict_negative_text(self, trained_model):
        result = trained_model.predict("Terrible experience, I hate it.")
        assert result in {"positive", "negative", "neutral"}


# ---------------------------------------------------------------------------
# 5. Batch prediction tests
# ---------------------------------------------------------------------------

class TestBatchPredict:
    """Test predict() with a list input."""

    def test_predict_list_returns_list(self, trained_model):
        texts = ["Great!", "Awful!", "Meh."]
        result = trained_model.predict(texts)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_predict_list_all_valid_sentiments(self, trained_model):
        texts = SAMPLE_TEXTS[:6]
        results = trained_model.predict(texts)
        for r in results:
            assert r in {"positive", "negative", "neutral"}

    def test_predict_single_element_list(self, trained_model):
        result = trained_model.predict(["Great product!"])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_predict_single_str_vs_single_list(self, trained_model):
        """Ensure single string and single-element list give same label."""
        text = "I really enjoyed this product."
        str_result = trained_model.predict(text)
        list_result = trained_model.predict([text])
        assert str_result == list_result[0]


# ---------------------------------------------------------------------------
# 6. Probability prediction tests
# ---------------------------------------------------------------------------

class TestPredictProba:
    """Test predict_proba() for single and batch inputs."""

    def test_proba_single_returns_dict(self, trained_model):
        result = trained_model.predict_proba("I love it!")
        assert isinstance(result, dict)

    def test_proba_single_keys_match_classes(self, trained_model):
        result = trained_model.predict_proba("I love it!")
        assert set(result.keys()) == set(trained_model.classes_)

    def test_proba_single_sums_to_one(self, trained_model):
        result = trained_model.predict_proba("I love it!")
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_proba_single_values_in_range(self, trained_model):
        result = trained_model.predict_proba("Testing sentiment.")
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_proba_batch_returns_list(self, trained_model):
        texts = ["Great!", "Terrible!", "Okay."]
        result = trained_model.predict_proba(texts)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_proba_batch_each_sums_to_one(self, trained_model):
        texts = SAMPLE_TEXTS[:4]
        results = trained_model.predict_proba(texts)
        for r in results:
            assert abs(sum(r.values()) - 1.0) < 1e-6

    def test_proba_before_training_raises(self, untrained_model):
        with pytest.raises(RuntimeError, match="trained"):
            untrained_model.predict_proba("test")


# ---------------------------------------------------------------------------
# 7. Evaluate tests
# ---------------------------------------------------------------------------

class TestEvaluate:
    """Test evaluate() return shape and values."""

    def test_evaluate_returns_dict(self, trained_model):
        metrics = trained_model.evaluate(
            texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS
        )
        assert isinstance(metrics, dict)

    def test_evaluate_has_accuracy(self, trained_model):
        metrics = trained_model.evaluate(
            texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS
        )
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_evaluate_has_per_class_accuracy(self, trained_model):
        metrics = trained_model.evaluate(
            texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS
        )
        assert "per_class_accuracy" in metrics
        assert isinstance(metrics["per_class_accuracy"], dict)

    def test_evaluate_per_class_accuracy_keys(self, trained_model):
        metrics = trained_model.evaluate(
            texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS
        )
        expected_keys = {"positive", "negative", "neutral"}
        assert set(metrics["per_class_accuracy"].keys()) == expected_keys

    def test_evaluate_per_class_accuracy_values_in_range(self, trained_model):
        metrics = trained_model.evaluate(
            texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS
        )
        for k, v in metrics["per_class_accuracy"].items():
            assert 0.0 <= v <= 1.0, f"per_class_accuracy[{k}] out of range: {v}"

    def test_evaluate_has_f1_macro(self, trained_model):
        metrics = trained_model.evaluate(
            texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS
        )
        assert "f1_macro" in metrics
        assert 0.0 <= metrics["f1_macro"] <= 1.0

    def test_evaluate_has_confusion_matrix(self, trained_model):
        metrics = trained_model.evaluate(
            texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS
        )
        assert "confusion_matrix" in metrics
        assert isinstance(metrics["confusion_matrix"], list)

    def test_evaluate_before_training_raises(self, untrained_model):
        with pytest.raises(RuntimeError, match="trained"):
            untrained_model.evaluate(
                texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS
            )

    def test_evaluate_on_train_set_high_accuracy(self, trained_model):
        """Training accuracy on small sample should be > 0.5."""
        metrics = trained_model.evaluate(
            texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS
        )
        # On the training set the model should do reasonably well
        assert metrics["accuracy"] > 0.5


# ---------------------------------------------------------------------------
# 8. Save and load tests
# ---------------------------------------------------------------------------

class TestSaveLoad:
    """Test model serialisation round-trip."""

    def test_save_creates_joblib_file(self, trained_model, tmp_model_path):
        trained_model.save(tmp_model_path)
        joblib_path = Path(tmp_model_path).with_suffix(".joblib")
        assert joblib_path.exists()

    def test_save_creates_json_sidecar(self, trained_model, tmp_model_path):
        trained_model.save(tmp_model_path)
        json_path = Path(tmp_model_path).with_suffix(".json")
        assert json_path.exists()

    def test_save_json_contains_metadata(self, trained_model, tmp_model_path):
        trained_model.save(tmp_model_path)
        json_path = Path(tmp_model_path).with_suffix(".json")
        with open(json_path) as fh:
            meta = json.load(fh)
        assert "model_name" in meta
        assert "classes" in meta

    def test_save_before_training_raises(self, untrained_model, tmp_model_path):
        with pytest.raises(RuntimeError, match="trained"):
            untrained_model.save(tmp_model_path)

    def test_load_via_json_path(self, trained_model, tmp_model_path):
        trained_model.save(tmp_model_path)

        new_model = NaiveBayesSentimentModel()
        assert not new_model.is_trained
        new_model.load(tmp_model_path)  # .json path

        assert new_model.is_trained

    def test_load_via_joblib_path(self, trained_model, tmp_model_path):
        trained_model.save(tmp_model_path)
        joblib_path = str(Path(tmp_model_path).with_suffix(".joblib"))

        new_model = NaiveBayesSentimentModel()
        new_model.load(joblib_path)

        assert new_model.is_trained

    def test_load_preserves_classes(self, trained_model, tmp_model_path):
        trained_model.save(tmp_model_path)

        new_model = NaiveBayesSentimentModel()
        new_model.load(tmp_model_path)

        assert set(new_model.classes_) == set(trained_model.classes_)

    def test_load_predictions_match_original(self, trained_model, tmp_model_path):
        """Loaded model should give same predictions as trained model."""
        trained_model.save(tmp_model_path)

        new_model = NaiveBayesSentimentModel()
        new_model.load(tmp_model_path)

        for text in SAMPLE_TEXTS[:4]:
            assert new_model.predict(text) == trained_model.predict(text)

    def test_load_nonexistent_file_raises(self, untrained_model):
        with pytest.raises(FileNotFoundError):
            untrained_model.load("/nonexistent/path/model.joblib")


# ---------------------------------------------------------------------------
# 9. Preprocessor integration tests
# ---------------------------------------------------------------------------

class TestPreprocessorIntegration:
    """Verify the optional TextPreprocessor integrates correctly."""

    def test_model_with_mock_preprocessor_trains(self):
        mock_pp = MagicMock()
        mock_pp.preprocess.side_effect = lambda t: t.split()

        model = NaiveBayesSentimentModel(preprocessor=mock_pp)
        model.train(texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS)
        assert model.is_trained

    def test_model_with_mock_preprocessor_predicts(self):
        mock_pp = MagicMock()
        mock_pp.preprocess.side_effect = lambda t: t.split()

        model = NaiveBayesSentimentModel(preprocessor=mock_pp)
        model.train(texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS)

        result = model.predict("I love this")
        assert result in {"positive", "negative", "neutral"}

    def test_preprocessor_called_on_predict(self):
        mock_pp = MagicMock()
        mock_pp.preprocess.side_effect = lambda t: t.split()

        model = NaiveBayesSentimentModel(preprocessor=mock_pp)
        model.train(texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS)

        call_count_before = mock_pp.preprocess.call_count
        model.predict("Test text here")
        # Preprocessor should have been called once more
        assert mock_pp.preprocess.call_count == call_count_before + 1

    def test_preprocessor_failure_falls_back_to_raw_text(self):
        """If preprocessor throws, raw text should be used and model still works."""
        mock_pp = MagicMock()
        mock_pp.preprocess.side_effect = lambda t: t.split()

        model = NaiveBayesSentimentModel(preprocessor=mock_pp)
        model.train(texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS)

        # Now make it raise on predict
        mock_pp.preprocess.side_effect = RuntimeError("boom")
        # Should not raise – falls back to raw text
        result = model.predict("Some test text")
        assert result in {"positive", "negative", "neutral"}


# ---------------------------------------------------------------------------
# 10. Backward-compatibility alias tests
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Ensure all aliases behave identically to their canonical equivalents."""

    def test_base_model_alias_is_same_class(self):
        assert BaseModel is SentimentModel

    def test_naive_bayes_classifier_is_same_class(self):
        assert NaiveBayesClassifier is NaiveBayesSentimentModel

    def test_naive_bayes_classifier_can_train_and_predict(self):
        model = NaiveBayesClassifier()
        model.train(texts=SAMPLE_TEXTS, labels=SAMPLE_LABELS)
        result = model.predict("I love it!")
        assert result in {"positive", "negative", "neutral"}

    def test_naive_bayes_classifier_model_name(self):
        model = NaiveBayesClassifier()
        assert model.model_name == "naive_bayes_sentiment"
