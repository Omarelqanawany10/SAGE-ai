"""Local Streamlit dashboard for crop water quantity prediction.

Run this after executing the training notebook:

    streamlit run streamlit_crop_water_dashboard.py

The dashboard loads the fitted sklearn pipeline and metadata from the
`artifacts` directory created by the notebook.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Imported before joblib.load so custom pipeline objects can be reconstructed.
from crop_water_model_utils import IQRClipper, add_engineered_features  # noqa: F401


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "best_crop_water_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


def pretty_label(column_name: str) -> str:
    """Convert dataset-style column names into readable Streamlit labels."""
    return (
        column_name.replace("_", " ")
        .replace("%", "percent")
        .replace("kg ha", "kg/ha")
        .replace("mm day", "mm/day")
        .strip()
        .title()
    )


def finite_float(value: object, default: float = 0.0) -> float:
    """Return a finite float for UI defaults."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(number) or math.isinf(number):
        return default
    return number


@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple[object | None, dict | None, str | None]:
    """Load the trained model pipeline and metadata."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None, (
            "Model artifacts were not found. Run the training notebook first to "
            "create artifacts/best_crop_water_model.joblib and "
            "artifacts/model_metadata.json."
        )

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        return None, None, f"Could not load model artifacts: {exc}"

    return model, metadata, None


def build_feature_input(metadata: dict) -> dict:
    """Render input controls and return one row of raw model inputs."""
    user_input = {}

    categorical_features = metadata.get("categorical_input_features", [])
    numeric_features = metadata.get("numeric_input_features", [])
    categorical_options = metadata.get("categorical_options", {})
    numeric_stats = metadata.get("numeric_input_stats", {})

    with st.form("crop_water_prediction_form"):
        if categorical_features:
            st.subheader("Categorical Inputs")
            category_columns = st.columns(2)

            for index, feature in enumerate(categorical_features):
                options = categorical_options.get(feature, [])
                if options:
                    value = category_columns[index % 2].selectbox(
                        pretty_label(feature),
                        options=options,
                        index=0,
                    )
                else:
                    value = category_columns[index % 2].text_input(pretty_label(feature))
                user_input[feature] = value

        if numeric_features:
            st.subheader("Numerical Inputs")
            numeric_columns = st.columns(2)

            for index, feature in enumerate(numeric_features):
                stats = numeric_stats.get(feature, {})
                min_value = finite_float(stats.get("min"), 0.0)
                max_value = finite_float(stats.get("max"), min_value + 1.0)
                median_value = finite_float(stats.get("median"), (min_value + max_value) / 2)

                if max_value <= min_value:
                    max_value = min_value + 1.0

                median_value = min(max(median_value, min_value), max_value)
                step = max((max_value - min_value) / 100, 0.01)

                value = numeric_columns[index % 2].slider(
                    pretty_label(feature),
                    min_value=float(min_value),
                    max_value=float(max_value),
                    value=float(median_value),
                    step=float(step),
                )
                user_input[feature] = value

        submitted = st.form_submit_button("Predict Water Need")

    return user_input if submitted else {}


def predict_water_need(model: object, metadata: dict, raw_input: dict) -> float:
    """Apply feature engineering, align schema, and generate prediction."""
    input_df = pd.DataFrame([raw_input])
    feature_df = add_engineered_features(input_df)

    model_feature_columns = metadata.get("model_feature_columns", [])
    for column in model_feature_columns:
        if column not in feature_df.columns:
            feature_df[column] = np.nan

    feature_df = feature_df[model_feature_columns]
    prediction = float(model.predict(feature_df)[0])

    # Irrigation demand cannot be negative in practice.
    return max(prediction, 0.0)


def main() -> None:
    """Render the Streamlit app."""
    st.set_page_config(
        page_title="Crop Water Prediction",
        layout="centered",
    )

    st.title("Crop Water Quantity Prediction")

    model, metadata, load_error = load_artifacts()
    if load_error:
        st.error(load_error)
        st.stop()

    st.caption(f"Loaded model: {metadata.get('model_name', 'Unknown model')}")

    raw_input = build_feature_input(metadata)
    if not raw_input:
        return

    predicted_mm_day = predict_water_need(model, metadata, raw_input)

    st.divider()
    st.metric(
        "Predicted daily water need",
        f"{predicted_mm_day:.2f} {metadata.get('target_unit', 'mm/day')}",
    )

    field_area = finite_float(raw_input.get("Field_Area_hectare"), default=np.nan)
    if not math.isnan(field_area) and field_area > 0:
        cubic_meters_per_day = predicted_mm_day * field_area * 10
        liters_per_day = cubic_meters_per_day * 1000

        col1, col2 = st.columns(2)
        col1.metric("Approximate volume", f"{cubic_meters_per_day:,.2f} m3/day")
        col2.metric("Approximate volume", f"{liters_per_day:,.0f} L/day")

    metrics = metadata.get("test_metrics", {}).get(metadata.get("model_name"), {})
    if metrics:
        st.subheader("Saved Test Metrics")
        st.dataframe(
            pd.DataFrame([metrics]).round(4),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
