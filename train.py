# ============================================================
# Lumpy Skin Disease Classification
# Final Training Pipeline
# ============================================================

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


# ============================================================
# 1. Paths & Configuration
# ============================================================

DATA_PATH = Path("data/Lumpy skin disease data.csv")

MODEL_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")

MODEL_PATH = MODEL_DIR / "lumpy_skin_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURES_PATH = ARTIFACTS_DIR / "feature_names.json"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"

TARGET_COLUMN = "lumpy"

RANDOM_STATE = 42

# Final split:
# 70% Train
# 10% Validation
# 20% Test
TEST_SIZE = 0.20
VALIDATION_SIZE_FROM_TRAIN = 0.125


# ============================================================
# 2. Features
# ============================================================

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


# ============================================================
# 3. Create Required Directories
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. Load Dataset
# ============================================================

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 5. Validate Columns
# ============================================================

required_columns = FEATURES + [TARGET_COLUMN]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 6. Prepare Data
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET_COLUMN].copy()

# Convert features to numeric
for column in FEATURES:
    X[column] = pd.to_numeric(X[column], errors="coerce")

# Convert target to numeric
y = pd.to_numeric(y, errors="coerce")

# Remove rows containing missing values
valid_rows = X.notna().all(axis=1) & y.notna()

X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)

# Make sure target is binary 0/1
unique_targets = sorted(y.unique().tolist())

if unique_targets != [0, 1]:
    raise ValueError(
        f"Expected target values [0, 1], but found: {unique_targets}"
    )

y = y.astype(int)

print(f"Clean dataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts().sort_index()}")


# ============================================================
# 7. Train / Validation / Test Split
# ============================================================

print("\n" + "=" * 60)
print("Creating Train / Validation / Test splits...")
print("=" * 60)

# First:
# 80% Train+Validation
# 20% Test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Then split Train+Validation:
# 87.5% Train
# 12.5% Validation
#
# Final result:
# 70% Train
# 10% Validation
# 20% Test

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=VALIDATION_SIZE_FROM_TRAIN,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

print(f"Train size:      {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size:       {len(X_test)}")


# ============================================================
# 8. Define Baseline Models
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE,
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# ============================================================
# 9. Train Baseline Models
# ============================================================

print("\n" + "=" * 60)
print("Training baseline models...")
print("=" * 60)

validation_results = []

fitted_models = {}

for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    validation_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "F1": f1,
    })

    fitted_models[name] = model

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1:       {f1:.4f}")


validation_results_df = pd.DataFrame(validation_results)

validation_results_df = validation_results_df.sort_values(
    by="F1",
    ascending=False,
).reset_index(drop=True)

print("\nValidation Results:")
print(validation_results_df)


# ============================================================
# 10. Tune XGBoost
# ============================================================

print("\n" + "=" * 60)
print("Tuning XGBoost...")
print("=" * 60)

xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

param_distributions = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2, 0.3],
}

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_distributions,
    n_iter=10,
    scoring="f1",
    cv=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

random_search.fit(X_train, y_train)

tuned_xgb = random_search.best_estimator_

y_val_pred_xgb = tuned_xgb.predict(X_val)

xgb_val_accuracy = accuracy_score(
    y_val,
    y_val_pred_xgb,
)

xgb_val_f1 = f1_score(
    y_val,
    y_val_pred_xgb,
)

print("\nBest XGBoost Parameters:")
print(random_search.best_params_)

print(
    f"\nTuned XGBoost Validation Accuracy: "
    f"{xgb_val_accuracy:.4f}"
)

print(
    f"Tuned XGBoost Validation F1: "
    f"{xgb_val_f1:.4f}"
)


# ============================================================
# 11. Add Tuned XGBoost to Validation Results
# ============================================================

validation_results.append({
    "Model": "Tuned XGBoost",
    "Accuracy": xgb_val_accuracy,
    "F1": xgb_val_f1,
})

validation_results_df = pd.DataFrame(
    validation_results
)

validation_results_df = validation_results_df.sort_values(
    by="F1",
    ascending=False,
).reset_index(drop=True)

print("\n" + "=" * 60)
print("Final Validation Model Comparison")
print("=" * 60)

print(validation_results_df)


# ============================================================
# 12. Select Best Model Using VALIDATION ONLY
# ============================================================

best_model_name = validation_results_df.iloc[0]["Model"]
best_validation_f1 = validation_results_df.iloc[0]["F1"]

print("\n" + "=" * 60)
print("Best Model Selection")
print("=" * 60)

print(f"Best Model: {best_model_name}")
print(f"Validation F1: {best_validation_f1:.4f}")


if best_model_name == "Tuned XGBoost":
    selected_model = tuned_xgb
else:
    selected_model = fitted_models[best_model_name]


# ============================================================
# 13. Retrain Selected Model on Train + Validation
# ============================================================

print("\n" + "=" * 60)
print("Retraining selected model on Train + Validation...")
print("=" * 60)

# Clone the selected model so we keep the selected
# configuration but train a fresh final model.

final_model = clone(selected_model)

final_model.fit(
    X_train_full,
    y_train_full,
)

print("Final model training completed.")


# ============================================================
# 14. Final Evaluation on TEST SET
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

# IMPORTANT:
# The test set is used ONLY here for the final evaluation.

y_test_pred = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(
    y_test,
    y_test_pred,
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
)

test_roc_auc = roc_auc_score(
    y_test,
    y_test_proba,
)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test F1:       {test_f1:.4f}")
print(f"Test ROC-AUC:  {test_roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=[
            "Healthy",
            "Lumpy Skin Disease",
        ],
    )
)


# ============================================================
# 15. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=[0, 1],
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Healthy",
        "Lumpy Skin Disease",
    ],
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    values_format="d",
)

ax.set_title(
    "Lumpy Skin Disease - Final Test Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"\nConfusion matrix saved to: "
    f"{CONFUSION_MATRIX_PATH}"
)


# ============================================================
# 16. Save Final Best Model
# ============================================================

print("\n" + "=" * 60)
print("Saving final model...")
print("=" * 60)

joblib.dump(
    final_model,
    MODEL_PATH,
)

print(
    f"Final model saved to:\n{MODEL_PATH}"
)


# ============================================================
# 17. Save Feature Names
# ============================================================

with open(
    FEATURES_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        FEATURES,
        f,
        ensure_ascii=False,
        indent=4,
    )

print(
    f"Feature names saved to:\n{FEATURES_PATH}"
)


# ============================================================
# 18. Save Metrics
# ============================================================

metrics = {
    "model": best_model_name,

    "validation": {
        "accuracy": float(
            validation_results_df.iloc[0]["Accuracy"]
        ),
        "f1": float(best_validation_f1),
    },

    "test": {
        "accuracy": float(test_accuracy),
        "f1": float(test_f1),
        "roc_auc": float(test_roc_auc),
    },

    "dataset": {
        "total_samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
    },

    "target_mapping": {
        "0": "Healthy",
        "1": "Lumpy Skin Disease",
    },

    "features": FEATURES,

    "best_xgboost_parameters": random_search.best_params_,
}

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        ensure_ascii=False,
        indent=4,
    )

print(
    f"Metrics saved to:\n{METRICS_PATH}"
)


# ============================================================
# 19. Final Summary
# ============================================================

print("\n" + "=" * 60)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Best Model:       {best_model_name}")
print(f"Validation F1:    {best_validation_f1:.4f}")
print(f"Test Accuracy:    {test_accuracy:.4f}")
print(f"Test F1:          {test_f1:.4f}")
print(f"Test ROC-AUC:     {test_roc_auc:.4f}")

print("\nSaved files:")

print(f"✓ {MODEL_PATH}")
print(f"✓ {METRICS_PATH}")
print(f"✓ {FEATURES_PATH}")
print(f"✓ {CONFUSION_MATRIX_PATH}")

print("\nDone.")