# ============================================================
# Lumpy Skin Disease AI
# Professional Streamlit Application
# HTML-Free Stable UI Version
# ============================================================

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lumpy Skin Disease AI",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "lumpy_skin_model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"
FEATURES_PATH = BASE_DIR / "artifacts" / "feature_names.json"


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f8fafc;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        min-height: 110px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 800;
    }

    .hero {
        background: linear-gradient(
            135deg,
            #0f172a,
            #172554,
            #1e3a8a
        );

        padding: 2.5rem;
        border-radius: 22px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.15);
    }

    .hero h1 {
        color: white;
        margin: 0;
        font-size: 2.4rem;
        font-weight: 800;
    }

    .hero p {
        color: #cbd5e1;
        margin-top: 0.7rem;
        font-size: 1rem;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 1rem;
        font-size: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        st.error(
            "Trained model was not found."
        )

        st.info(
            f"Expected path: {MODEL_PATH}"
        )

        st.stop()

    try:

        return joblib.load(MODEL_PATH)

    except Exception as e:

        st.error(
            "Failed to load the trained model."
        )

        st.exception(e)

        st.stop()


# ============================================================
# 5. LOAD METRICS
# ============================================================

@st.cache_data
def load_metrics():

    if not METRICS_PATH.exists():
        return {}

    try:

        with open(
            METRICS_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return {}


# ============================================================
# 6. LOAD FEATURES
# ============================================================

@st.cache_data
def load_features():

    if not FEATURES_PATH.exists():

        st.error(
            "Feature configuration was not found."
        )

        st.info(
            f"Expected path: {FEATURES_PATH}"
        )

        st.stop()

    try:

        with open(
            FEATURES_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception as e:

        st.error(
            "Failed to load feature configuration."
        )

        st.exception(e)

        st.stop()


# ============================================================
# 7. LOAD ARTIFACTS
# ============================================================

model = load_model()

metrics = load_metrics()

FEATURES = load_features()


# ============================================================
# 8. FEATURE LABELS
# ============================================================

FEATURE_LABELS = {

    "x":
        "Longitude (x)",

    "y":
        "Latitude (y)",

    "cld":
        "Cloud Cover (cld)",

    "dtr":
        "Diurnal Temperature Range (dtr)",

    "frs":
        "Frost Days (frs)",

    "pet":
        "Potential Evapotranspiration (pet)",

    "pre":
        "Precipitation (pre)",

    "tmn":
        "Minimum Temperature (tmn)",

    "tmp":
        "Mean Temperature (tmp)",

    "tmx":
        "Maximum Temperature (tmx)",

    "vap":
        "Vapor Pressure (vap)",

    "wet":
        "Wet Days (wet)",

    "elevation":
        "Elevation",

    "dominant_land_cover":
        "Dominant Land Cover",

    "X5_Ct_2010_Da":
        "Cattle Density",

    "X5_Bf_2010_Da":
        "Buffalo Density",
}


# ============================================================
# 9. METRICS
# ============================================================

test_metrics = metrics.get(
    "test",
    {},
)

test_accuracy = float(
    test_metrics.get(
        "accuracy",
        0,
    )
)

test_f1 = float(
    test_metrics.get(
        "f1",
        0,
    )
)

test_roc_auc = float(
    test_metrics.get(
        "roc_auc",
        0,
    )
)

best_model = metrics.get(
    "model",
    type(model).__name__,
)


# ============================================================
# 10. HERO
# ============================================================
# This is the ONLY HTML block in the application.
# The rest of the interface uses native Streamlit widgets.
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🐄 Lumpy Skin Disease AI</h1>
        <p>
            Intelligent Machine Learning system for
            Lumpy Skin Disease risk classification
            using environmental, geographical and
            livestock density features.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 11. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🐄 LSD AI")

    st.divider()

    st.subheader("About")

    st.write(
        "This application uses a trained Machine "
        "Learning model to classify whether the "
        "provided environmental and geographical "
        "conditions are associated with "
        "Lumpy Skin Disease."
    )

    st.subheader("Model")

    st.write(
        f"**Current model:** {best_model}"
    )

    st.subheader("Classes")

    st.success(
        "🟢 0 — Healthy"
    )

    st.error(
        "🔴 1 — Lumpy Skin Disease"
    )

    st.divider()

    st.caption(
        "Professional Machine Learning Deployment"
    )


# ============================================================
# 12. MODEL PERFORMANCE
# ============================================================

st.header("📊 Model Performance")

st.caption(
    "Final evaluation results on the held-out test set."
)

metric_columns = st.columns(4)


with metric_columns[0]:

    st.metric(
        "Test Accuracy",
        f"{test_accuracy * 100:.2f}%",
    )


with metric_columns[1]:

    st.metric(
        "Test F1 Score",
        f"{test_f1 * 100:.2f}%",
    )


with metric_columns[2]:

    st.metric(
        "Test ROC-AUC",
        f"{test_roc_auc * 100:.2f}%",
    )


with metric_columns[3]:

    st.metric(
        "Selected Model",
        best_model,
    )


st.divider()


# ============================================================
# 13. TABS
# ============================================================

tab_single, tab_batch, tab_about = st.tabs(
    [
        "🔍 Single Prediction",
        "📁 Batch CSV Analysis",
        "ℹ️ About Model",
    ]
)


# ============================================================
# 14. SINGLE PREDICTION
# ============================================================

with tab_single:

    st.header("🔍 Analyze One Sample")

    st.caption(
        "Enter the environmental, geographical and "
        "livestock density features to generate a "
        "real model prediction."
    )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        x = st.number_input(
            FEATURE_LABELS["x"],
            value=0.0,
            format="%.6f",
        )

        y = st.number_input(
            FEATURE_LABELS["y"],
            value=0.0,
            format="%.6f",
        )

        cld = st.number_input(
            FEATURE_LABELS["cld"],
            value=0.0,
        )

        dtr = st.number_input(
            FEATURE_LABELS["dtr"],
            value=0.0,
        )

    with col2:

        frs = st.number_input(
            FEATURE_LABELS["frs"],
            value=0.0,
        )

        pet = st.number_input(
            FEATURE_LABELS["pet"],
            value=0.0,
        )

        pre = st.number_input(
            FEATURE_LABELS["pre"],
            value=0.0,
        )

        tmn = st.number_input(
            FEATURE_LABELS["tmn"],
            value=0.0,
        )

    with col3:

        tmp = st.number_input(
            FEATURE_LABELS["tmp"],
            value=0.0,
        )

        tmx = st.number_input(
            FEATURE_LABELS["tmx"],
            value=0.0,
        )

        vap = st.number_input(
            FEATURE_LABELS["vap"],
            value=0.0,
        )

        wet = st.number_input(
            FEATURE_LABELS["wet"],
            value=0.0,
        )

    with col4:

        elevation = st.number_input(
            FEATURE_LABELS["elevation"],
            value=0.0,
        )

        dominant_land_cover = st.number_input(
            FEATURE_LABELS["dominant_land_cover"],
            value=0.0,
        )

        cattle_density = st.number_input(
            FEATURE_LABELS["X5_Ct_2010_Da"],
            value=0.0,
        )

        buffalo_density = st.number_input(
            FEATURE_LABELS["X5_Bf_2010_Da"],
            value=0.0,
        )

    st.write("")

    predict_button = st.button(
        "🚀 Predict Disease Risk",
        type="primary",
        use_container_width=True,
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        input_values = {

            "x": x,
            "y": y,
            "cld": cld,
            "dtr": dtr,
            "frs": frs,
            "pet": pet,
            "pre": pre,
            "tmn": tmn,
            "tmp": tmp,
            "tmx": tmx,
            "vap": vap,
            "wet": wet,
            "elevation": elevation,
            "dominant_land_cover": dominant_land_cover,
            "X5_Ct_2010_Da": cattle_density,
            "X5_Bf_2010_Da": buffalo_density,
        }

        # ----------------------------------------------------
        # INPUT DATA
        # ----------------------------------------------------

        try:

            input_data = pd.DataFrame(
                [
                    [
                        input_values[feature]
                        for feature in FEATURES
                    ]
                ],
                columns=FEATURES,
            )

        except Exception as e:

            st.error(
                "Could not construct model input."
            )

            st.exception(e)

            st.stop()

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        try:

            prediction = int(
                model.predict(
                    input_data
                )[0]
            )

        except Exception as e:

            st.error(
                "Model prediction failed."
            )

            st.exception(e)

            st.stop()

        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probabilities = None

        disease_probability = None

        if hasattr(
            model,
            "predict_proba",
        ):

            try:

                probabilities = model.predict_proba(
                    input_data
                )[0]

                disease_probability = float(
                    probabilities[1]
                )

            except Exception:

                probabilities = None

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                "🔴 Lumpy Skin Disease Detected"
            )

            st.write(
                "The model classified this sample "
                "as Lumpy Skin Disease."
            )

        else:

            st.success(
                "🟢 Healthy / No Disease Detected"
            )

            st.write(
                "The model classified this sample "
                "as Healthy."
            )

        # ----------------------------------------------------
        # DISEASE PROBABILITY
        # ----------------------------------------------------

        if disease_probability is not None:

            st.metric(
                "Disease Probability",
                f"{disease_probability * 100:.2f}%",
            )

        else:

            st.info(
                "Probability is not available for this model."
            )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        if probabilities is not None:

            st.subheader(
                "📈 Prediction Probabilities"
            )

            probability_columns = st.columns(2)

            with probability_columns[0]:

                st.metric(
                    "🟢 Healthy",
                    f"{probabilities[0] * 100:.2f}%",
                )

            with probability_columns[1]:

                st.metric(
                    "🔴 Lumpy Skin Disease",
                    f"{probabilities[1] * 100:.2f}%",
                )


# ============================================================
# 15. BATCH CSV
# ============================================================

with tab_batch:

    st.header(
        "📁 Batch CSV Analysis"
    )

    st.caption(
        "Upload a CSV containing the 16 model features."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )

        except Exception as e:

            st.error(
                "Unable to read the uploaded CSV."
            )

            st.exception(e)

            st.stop()

        st.subheader(
            "Uploaded Data"
        )

        st.dataframe(
            batch_df.head(20),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # CHECK COLUMNS
        # ----------------------------------------------------

        missing = [
            feature
            for feature in FEATURES
            if feature not in batch_df.columns
        ]

        if missing:

            st.error(
                "Missing required columns:"
            )

            st.write(
                missing
            )

        else:

            st.success(
                f"All {len(FEATURES)} required "
                "features are present."
            )

            analyze_button = st.button(
                "🚀 Analyze CSV",
                type="primary",
                use_container_width=True,
            )

            if analyze_button:

                prediction_data = batch_df[
                    FEATURES
                ].copy()

                # ------------------------------------------------
                # NUMERIC CONVERSION
                # ------------------------------------------------

                for feature in FEATURES:

                    prediction_data[feature] = pd.to_numeric(
                        prediction_data[feature],
                        errors="coerce",
                    )

                invalid_rows = (
                    prediction_data
                    .isna()
                    .any(axis=1)
                )

                invalid_count = int(
                    invalid_rows.sum()
                )

                if invalid_count > 0:

                    st.error(
                        f"{invalid_count} rows contain "
                        "missing or non-numeric values."
                    )

                else:

                    try:

                        predictions = model.predict(
                            prediction_data
                        )

                    except Exception as e:

                        st.error(
                            "Batch prediction failed."
                        )

                        st.exception(e)

                        st.stop()

                    result_df = batch_df.copy()

                    result_df[
                        "prediction"
                    ] = predictions

                    result_df[
                        "prediction_label"
                    ] = (
                        result_df[
                            "prediction"
                        ]
                        .map(
                            {
                                0: "Healthy",
                                1: "Lumpy Skin Disease",
                            }
                        )
                    )

                    # ------------------------------------------------
                    # PROBABILITY
                    # ------------------------------------------------

                    if hasattr(
                        model,
                        "predict_proba",
                    ):

                        try:

                            probabilities = (
                                model.predict_proba(
                                    prediction_data
                                )[:, 1]
                            )

                            result_df[
                                "disease_probability"
                            ] = probabilities

                            result_df[
                                "disease_probability_percent"
                            ] = (
                                probabilities * 100
                            ).round(2)

                        except Exception:

                            pass

                    # ------------------------------------------------
                    # SUMMARY
                    # ------------------------------------------------

                    total = len(
                        result_df
                    )

                    healthy = int(
                        (
                            result_df["prediction"] == 0
                        ).sum()
                    )

                    disease = int(
                        (
                            result_df["prediction"] == 1
                        ).sum()
                    )

                    st.success(
                        "Batch analysis completed successfully."
                    )

                    st.subheader(
                        "📊 Prediction Summary"
                    )

                    summary_columns = st.columns(3)

                    with summary_columns[0]:

                        st.metric(
                            "Total Samples",
                            total,
                        )

                    with summary_columns[1]:

                        st.metric(
                            "🟢 Healthy",
                            healthy,
                        )

                    with summary_columns[2]:

                        st.metric(
                            "🔴 Disease",
                            disease,
                        )

                    # ------------------------------------------------
                    # RESULTS
                    # ------------------------------------------------

                    st.subheader(
                        "📋 Prediction Results"
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                    )

                    # ------------------------------------------------
                    # DOWNLOAD
                    # ------------------------------------------------

                    csv_output = (
                        result_df
                        .to_csv(
                            index=False
                        )
                        .encode("utf-8")
                    )

                    st.download_button(
                        label="⬇️ Download Predictions",
                        data=csv_output,
                        file_name="lumpy_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )


# ============================================================
# 16. ABOUT MODEL
# ============================================================

with tab_about:

    st.header(
        "ℹ️ About the Machine Learning Model"
    )

    st.caption(
        "Professional overview of the Lumpy Skin Disease "
        "classification system."
    )

    # --------------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------------

    st.subheader(
        "🧠 Project Overview"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.subheader(
                "🎯 Problem"
            )

            st.write(
                "Binary classification system for "
                "identifying conditions associated "
                "with Lumpy Skin Disease."
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "🏷️ Target"
            )

            st.success(
                "0 — Healthy"
            )

            st.error(
                "1 — Lumpy Skin Disease"
            )

    with col3:

        with st.container(border=True):

            st.subheader(
                "🤖 Selected Model"
            )

            st.write(
                best_model
            )

            st.caption(
                "Selected using validation performance "
                "before final evaluation."
            )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "📊 Final Test Performance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🎯 Accuracy",
            f"{test_accuracy * 100:.2f}%",
        )

    with col2:

        st.metric(
            "📈 F1 Score",
            f"{test_f1 * 100:.2f}%",
        )

    with col3:

        st.metric(
            "📐 ROC-AUC",
            f"{test_roc_auc * 100:.2f}%",
        )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.subheader(
        "🗂️ Dataset & Model Input"
    )

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader(
                "📁 Dataset"
            )

            st.write(
                "The model was trained using the "
                "project's real Lumpy Skin Disease dataset."
            )

            st.caption(
                "The application uses the same feature "
                "definitions as the training pipeline."
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "🔢 Model Inputs"
            )

            st.metric(
                "Required Features",
                len(FEATURES),
            )

            st.caption(
                "Feature order is loaded directly "
                "from feature_names.json."
            )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.subheader(
        "🌍 Input Features"
    )

    st.write(
        "Environmental, geographical and livestock "
        "density variables used by the model:"
    )

    feature_columns = st.columns(4)

    for index, feature in enumerate(FEATURES):

        label = FEATURE_LABELS.get(
            feature,
            feature,
        )

        with feature_columns[
            index % 4
        ]:

            st.write(
                f"• {label}"
            )

    # --------------------------------------------------------
    # MODEL INFO
    # --------------------------------------------------------

    st.subheader(
        "🤖 Model Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader(
                "Model"
            )

            st.write(
                f"**Algorithm:** {best_model}"
            )

            st.write(
                "**Task:** Binary Classification"
            )

            st.write(
                "**Output:** Disease Risk Classification"
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "Classes"
            )

            st.success(
                "🟢 0 — Healthy"
            )

            st.error(
                "🔴 1 — Lumpy Skin Disease"
            )

    # --------------------------------------------------------
    # WARNING
    # --------------------------------------------------------

    st.warning(
        "⚠️ This application is a Machine Learning "
        "classification system and should not be considered "
        "a veterinary diagnosis. Predictions should be "
        "interpreted as model-generated risk classifications."
    )


# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.caption(
    "🐄 Lumpy Skin Disease AI • Machine Learning Deployment"
)