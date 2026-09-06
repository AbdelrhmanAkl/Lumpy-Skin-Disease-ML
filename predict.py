"""
Simple inference utility for the deployed
Lumpy Skin Disease model.
"""

import joblib
import pandas as pd


MODEL_PATH = "models/lumpy_skin_model.joblib"


FEATURES = [
    "x",
    "y",
    "cld",
    "dtr",
    "frs",
    "pet",
    "pre",
    "tmn",
    "tmp",
    "tmx",
    "vap",
    "wet",
    "elevation",
    "dominant_land_cover",
    "X5_Ct_2010_Da",
    "X5_Bf_2010_Da",
]


# Load the real trained model.
model = joblib.load(MODEL_PATH)


def predict_lumpy(values: dict) -> dict:
    """
    Predict Lumpy Skin Disease for one sample.

    Parameters
    ----------
    values : dict
        Dictionary containing all required features.

    Returns
    -------
    dict
        Prediction and probability.
    """

    # Validate required features.
    missing = [
        feature
        for feature in FEATURES
        if feature not in values
    ]

    if missing:

        raise ValueError(
            f"Missing features: {missing}"
        )


    # Create DataFrame in exactly
    # the same feature order used during training.
    input_df = pd.DataFrame(
        [[
            values[feature]
            for feature in FEATURES
        ]],
        columns=FEATURES
    )


    # Real prediction.
    prediction = int(
        model.predict(input_df)[0]
    )


    # Probability.
    probability = float(
        model.predict_proba(input_df)[0][1]
    )


    return {

        "prediction": prediction,

        "label":
            "Lumpy Skin Disease"
            if prediction == 1
            else "Healthy",

        "disease_probability":
            probability,

    }