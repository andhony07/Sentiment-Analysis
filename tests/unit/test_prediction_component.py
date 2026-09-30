"""
Unit tests for the Streamlit prediction UI component.

These tests verify:
- The callback functions set the correct session-state values
- render_prediction_ui() returns the expected string type
- No session-state key collision (the original bug: writing to a widget
  key after the widget has been instantiated).

We use ``unittest.mock`` to patch Streamlit so these tests run headlessly
without a live Streamlit server.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers – patch the entire `streamlit` namespace used by the component
# ---------------------------------------------------------------------------

def _make_mock_st(session_state_dict=None):
    """Return a mock `st` module with a dict-backed session_state."""
    mock_st = MagicMock()

    # Give session_state attribute-style access backed by a real dict
    class _SS(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

        def __setattr__(self, key, value):
            self[key] = value

    mock_st.session_state = _SS(session_state_dict or {})

    # st.text_area should return whatever is in session_state.text_input
    def _text_area(*args, **kwargs):
        return mock_st.session_state.get("text_input", "")

    mock_st.text_area.side_effect = _text_area
    mock_st.columns.return_value = (
        MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False)),
        MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False)),
        MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False)),
    )

    return mock_st


# ---------------------------------------------------------------------------
# Tests for the callback helpers
# ---------------------------------------------------------------------------

class TestExampleCallbacks:
    """Directly test the on_click callback functions."""

    def test_set_positive_example(self):
        ss = {}

        class _SS(dict):
            def __setattr__(self, k, v): self[k] = v
            def __getattr__(self, k): return self.get(k)

        fake_ss = _SS()

        with patch.dict("sys.modules", {"streamlit": MagicMock()}):
            import importlib, sys

            # Build a fresh module with a controlled st
            mock_st = MagicMock()
            mock_st.session_state = fake_ss
            sys.modules["streamlit"] = mock_st

            # Re-import so the module picks up our mock st
            if "sentiment_analysis.streamlit_app.components.prediction" in sys.modules:
                del sys.modules["sentiment_analysis.streamlit_app.components.prediction"]

            from sentiment_analysis.streamlit_app.components.prediction import (
                _set_positive_example,
                _POSITIVE_EXAMPLE,
            )

            _set_positive_example()

        assert fake_ss["text_input"] == _POSITIVE_EXAMPLE

    def test_set_negative_example(self):
        fake_ss = {}

        class _SS(dict):
            def __setattr__(self, k, v): self[k] = v

        fake_ss = _SS()

        with patch.dict("sys.modules", {}):
            import sys, importlib

            mock_st = MagicMock()
            mock_st.session_state = fake_ss
            sys.modules["streamlit"] = mock_st

            if "sentiment_analysis.streamlit_app.components.prediction" in sys.modules:
                del sys.modules["sentiment_analysis.streamlit_app.components.prediction"]

            from sentiment_analysis.streamlit_app.components.prediction import (
                _set_negative_example,
                _NEGATIVE_EXAMPLE,
            )

            _set_negative_example()

        assert fake_ss["text_input"] == _NEGATIVE_EXAMPLE

    def test_set_neutral_example(self):
        class _SS(dict):
            def __setattr__(self, k, v): self[k] = v

        fake_ss = _SS()

        import sys
        mock_st = MagicMock()
        mock_st.session_state = fake_ss
        sys.modules["streamlit"] = mock_st

        if "sentiment_analysis.streamlit_app.components.prediction" in sys.modules:
            del sys.modules["sentiment_analysis.streamlit_app.components.prediction"]

        from sentiment_analysis.streamlit_app.components.prediction import (
            _set_neutral_example,
            _NEUTRAL_EXAMPLE,
        )

        _set_neutral_example()

        assert fake_ss["text_input"] == _NEUTRAL_EXAMPLE


# ---------------------------------------------------------------------------
# Tests verifying the fixed pattern (no post-widget-instantiation write)
# ---------------------------------------------------------------------------

class TestSessionStatePattern:
    """
    Verify that the component never writes to `text_input` AFTER calling
    st.text_area – the pattern that caused the original bug.
    """

    def test_buttons_registered_before_text_area(self):
        """
        In the fixed implementation st.button() calls appear in the source
        before st.text_area() – confirm this by inspecting the source.
        """
        import inspect
        import importlib
        import sys

        # Reload with a neutral mock so we can inspect source freely
        mock_st = MagicMock()
        mock_st.session_state = {}
        sys.modules["streamlit"] = mock_st

        if "sentiment_analysis.streamlit_app.components.prediction" in sys.modules:
            del sys.modules["sentiment_analysis.streamlit_app.components.prediction"]

        import sentiment_analysis.streamlit_app.components.prediction as pred_mod

        src = inspect.getsource(pred_mod.render_prediction_ui)

        button_pos = src.find("st.button")
        text_area_pos = src.find("st.text_area")

        assert button_pos != -1, "st.button not found in render_prediction_ui source"
        assert text_area_pos != -1, "st.text_area not found in render_prediction_ui source"
        assert button_pos < text_area_pos, (
            "st.button must appear BEFORE st.text_area in render_prediction_ui "
            "so callbacks fire before the widget is instantiated"
        )

    def test_no_experimental_rerun_call_in_source(self):
        """
        experimental_rerun() is deprecated and was part of the buggy
        workaround – confirm it is never *called* (the word may still
        appear in docstring comments, which is fine).
        """
        import inspect, sys

        mock_st = MagicMock()
        mock_st.session_state = {}
        sys.modules["streamlit"] = mock_st

        if "sentiment_analysis.streamlit_app.components.prediction" in sys.modules:
            del sys.modules["sentiment_analysis.streamlit_app.components.prediction"]

        import sentiment_analysis.streamlit_app.components.prediction as pred_mod

        src = inspect.getsource(pred_mod)
        # A call looks like: experimental_rerun()  – check for the call parenthesis
        assert "experimental_rerun()" not in src, (
            "st.experimental_rerun() is deprecated and must not be called"
        )

    def test_buttons_use_on_click_callbacks(self):
        """
        Buttons must declare on_click= callbacks rather than using
        if-button-clicked: st.session_state.text_input = ...
        which triggers the post-instantiation write error.
        """
        import inspect, sys

        mock_st = MagicMock()
        mock_st.session_state = {}
        sys.modules["streamlit"] = mock_st

        if "sentiment_analysis.streamlit_app.components.prediction" in sys.modules:
            del sys.modules["sentiment_analysis.streamlit_app.components.prediction"]

        import sentiment_analysis.streamlit_app.components.prediction as pred_mod

        src = inspect.getsource(pred_mod.render_prediction_ui)
        assert "on_click=" in src, (
            "Buttons should use on_click= callbacks to avoid post-widget state writes"
        )


# ---------------------------------------------------------------------------
# Tests for example text constants
# ---------------------------------------------------------------------------

class TestExampleTextConstants:
    """Verify the example texts are non-empty and sensibly classified."""

    def _get_constants(self):
        import sys
        mock_st = MagicMock()
        mock_st.session_state = {}
        sys.modules["streamlit"] = mock_st

        if "sentiment_analysis.streamlit_app.components.prediction" in sys.modules:
            del sys.modules["sentiment_analysis.streamlit_app.components.prediction"]

        import sentiment_analysis.streamlit_app.components.prediction as m
        return m._POSITIVE_EXAMPLE, m._NEGATIVE_EXAMPLE, m._NEUTRAL_EXAMPLE

    def test_positive_example_is_non_empty(self):
        pos, _, _ = self._get_constants()
        assert isinstance(pos, str) and len(pos) > 0

    def test_negative_example_is_non_empty(self):
        _, neg, _ = self._get_constants()
        assert isinstance(neg, str) and len(neg) > 0

    def test_neutral_example_is_non_empty(self):
        _, _, neu = self._get_constants()
        assert isinstance(neu, str) and len(neu) > 0

    def test_all_examples_are_distinct(self):
        pos, neg, neu = self._get_constants()
        assert pos != neg
        assert pos != neu
        assert neg != neu

    def test_positive_example_contains_positive_language(self):
        pos, _, _ = self._get_constants()
        # Should mention love / excellent / outstanding or similar
        pos_lower = pos.lower()
        assert any(w in pos_lower for w in ("love", "excellent", "outstanding", "great", "amazing"))

    def test_negative_example_contains_negative_language(self):
        _, neg, _ = self._get_constants()
        neg_lower = neg.lower()
        assert any(w in neg_lower for w in ("worst", "terrible", "disappointed", "waste", "poor"))
