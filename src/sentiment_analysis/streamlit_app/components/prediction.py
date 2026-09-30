"""
Prediction UI components for Streamlit application.

This module provides reusable UI components for sentiment prediction.

Session-state pattern
---------------------
Streamlit raises ``StreamlitAPIException`` if you write to a widget's own
session-state key *after* the widget has already been instantiated in the
current run.  The correct approach is:

1. Place example-button widgets **before** the text-area widget and use
   ``on_click`` callbacks so that ``st.session_state.text_input`` is
   updated *before* ``st.text_area`` is rendered on the same rerun.
2. Remove any ``st.experimental_rerun`` invocations – Streamlit
   automatically reruns the script after any widget interaction.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Example texts
# ---------------------------------------------------------------------------
_POSITIVE_EXAMPLE = (
    "I absolutely love this product! It exceeded all my expectations. "
    "The quality is outstanding and the service was excellent!"
)
_NEGATIVE_EXAMPLE = (
    "This is the worst experience I've ever had. Terrible quality, "
    "poor customer service, and a complete waste of money. Very disappointed!"
)
_NEUTRAL_EXAMPLE = (
    "The product arrived on time. It works as described. "
    "Nothing particularly good or bad to report."
)


# ---------------------------------------------------------------------------
# Button callbacks – run BEFORE the text_area is rendered on the rerun
# ---------------------------------------------------------------------------
def _set_positive_example() -> None:
    st.session_state.text_input = _POSITIVE_EXAMPLE


def _set_negative_example() -> None:
    st.session_state.text_input = _NEGATIVE_EXAMPLE


def _set_neutral_example() -> None:
    st.session_state.text_input = _NEUTRAL_EXAMPLE


# ---------------------------------------------------------------------------
# Public components
# ---------------------------------------------------------------------------

def render_prediction_ui() -> str:
    """
    Render the text input UI for sentiment prediction.

    The example buttons are placed **above** the text-area and use
    ``on_click`` callbacks so that ``st.session_state.text_input`` is
    updated before the widget is rendered on the following rerun –
    avoiding the ``StreamlitAPIException``.

    Returns:
        The current text in the input area.

    Example:
        >>> user_input = render_prediction_ui()
    """
    st.subheader("✍️ Enter Text for Analysis")

    # ------------------------------------------------------------------
    # Example buttons FIRST – callbacks fire before text_area renders
    # ------------------------------------------------------------------
    st.markdown("**💡 Try these examples:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.button(
            "😊 Positive Example",
            key="pos_example",
            on_click=_set_positive_example,
        )

    with col2:
        st.button(
            "😞 Negative Example",
            key="neg_example",
            on_click=_set_negative_example,
        )

    with col3:
        st.button(
            "😐 Neutral Example",
            key="neu_example",
            on_click=_set_neutral_example,
        )

    # ------------------------------------------------------------------
    # Text area AFTER the buttons so session-state is already updated
    # ------------------------------------------------------------------
    user_input = st.text_area(
        label="Your text:",
        height=150,
        placeholder=(
            "Type or paste your text here...\n\nExamples:\n"
            "- I absolutely love this product! It's amazing!\n"
            "- This is the worst experience ever. Very disappointed.\n"
            "- The service was okay, nothing special."
        ),
        key="text_input",
    )

    # Character count
    st.caption(f"Character count: {len(user_input)}")

    return user_input


def render_batch_prediction_ui():
    """
    Render UI for batch prediction (multiple texts).

    Returns:
        List of text inputs
    """
    st.subheader("📋 Batch Analysis")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a CSV or TXT file with texts to analyze", type=["csv", "txt"]
    )

    if uploaded_file is not None:
        # TODO: Implement file processing
        st.info("📁 File uploaded successfully! Processing...")

    # Manual batch input
    st.markdown("**Or enter multiple texts (one per line):**")

    batch_input = st.text_area(
        label="Multiple texts:",
        value="",
        height=200,
        placeholder=(
            "Enter one text per line...\n\n"
            "Example:\n"
            "I love this product!\n"
            "This is terrible.\n"
            "It's okay, nothing special."
        ),
        key="batch_input",
    )

    if batch_input:
        texts = [line.strip() for line in batch_input.split("\n") if line.strip()]
        st.info(f"📊 Ready to analyze {len(texts)} texts")
        return texts

    return []
