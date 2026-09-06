# Lumpy Skin Disease AI

> A professional machine learning application for binary classification of Lumpy Skin Disease (LSD) risk using environmental, geographic, and livestock-related features.

[🚀 Live Demo](https://lumpy-skin-disease-ml.streamlit.app/)  
[📂 GitHub Repository](https://github.com/AbdelrhmanAkl/lumpy-skin-disease-ml)

> A professional machine learning application for binary classification of **Lumpy Skin Disease (LSD) risk** using environmental, geographic, and livestock-related features.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)

---

## Overview

**Lumpy Skin Disease AI** is a machine learning project designed to classify the risk of Lumpy Skin Disease using a trained binary classification model.

The project combines:

- Geographic information
- Climate and environmental variables
- Elevation and land-cover information
- Cattle density
- Buffalo density

The trained model is integrated into a **Streamlit web application** that provides real-time predictions, prediction probabilities, batch CSV analysis, model information, and performance metrics.

> **Important:** This application is an AI/ML decision-support prototype for educational and research purposes. It is **not a veterinary diagnostic system** and should not replace professional veterinary assessment.

---

## Key Features

### Single Prediction
Enter the 16 required input features and receive:

- Predicted class
- Healthy / Lumpy Skin Disease classification
- Prediction probability
- Model-based risk indication

### Batch CSV Analysis
Upload a CSV file containing multiple observations to:

- Run predictions for all rows
- Generate prediction probabilities
- Review the results
- Export predictions as a CSV file

### Model Performance Dashboard
The application displays the final model's test-set performance:

| Metric | Score |
|---|---:|
| Accuracy | **98.15%** |
| F1 Score | **92.36%** |
| ROC-AUC | **99.30%** |

### Model Information
The application also provides information about:

- The selected machine learning model
- Input features
- Dataset and target definition
- Prediction workflow
- Project limitations

---

## Machine Learning Problem

### Task

Binary classification:

- `0` → Healthy
- `1` → Lumpy Skin Disease

### Final Model

**XGBoost**

The application uses the real trained model stored in:

```text
models/lumpy_skin_model.joblib
```

Predictions are generated from the same 16-feature input structure used by the trained model.

---

## Input Features

The final model uses the following **16 numerical features** in this exact order:

| Feature | Description |
|---|---|
| `x` | Longitude |
| `y` | Latitude |
| `cld` | Cloud Cover |
| `dtr` | Diurnal Temperature Range |
| `frs` | Frost Days |
| `pet` | Potential Evapotranspiration |
| `pre` | Precipitation |
| `tmn` | Minimum Temperature |
| `tmp` | Mean Temperature |
| `tmx` | Maximum Temperature |
| `vap` | Vapor Pressure |
| `wet` | Wet Days |
| `elevation` | Elevation |
| `dominant_land_cover` | Dominant Land Cover |
| `X5_Ct_2010_Da` | Cattle Density |
| `X5_Bf_2010_Da` | Buffalo Density |

Maintaining the feature names and feature order is important because the trained model expects this exact input structure.

---

## Project Architecture

```text
lumpy-skin-disease-ml/
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── Lumpy skin disease data.csv
│
├── models/
│   └── lumpy_skin_model.joblib
│
├── artifacts/
│   ├── metrics.json
│   ├── feature_names.json
│   └── confusion_matrix.png
│
├── notebooks/
│   └── Final_Project_Lumpy_Skin_Disease_data_Classification__.ipynb
│
└── .streamlit/
    └── config.toml
```

### Main Files

**`app.py`**  
Streamlit application containing the production-style user interface and prediction workflow.

**`train.py`**  
Training pipeline used to train and evaluate the machine learning models.

**`predict.py`**  
Prediction utilities used by the application.

**`models/lumpy_skin_model.joblib`**  
Serialized trained XGBoost model used for real predictions.

**`artifacts/metrics.json`**  
Stored model-performance metrics used by the dashboard.

**`artifacts/feature_names.json`**  
Stores the expected model input features.

**`artifacts/confusion_matrix.png`**  
Visualization of the final classification results.

**`notebooks/`**  
Contains the project development and experimentation notebook.

---

## ML Workflow

The intended professional evaluation workflow is:

```text
Dataset
   │
   ▼
Data Preparation
   │
   ▼
Train / Validation / Test Split
   │
   ├──────────────► Validation
   │                  │
   │                  ▼
   │            Model Selection
   │
   ▼
Selected Model
   │
   ▼
Final Test Evaluation
   │
   ▼
Saved Trained Model
   │
   ▼
Streamlit Application
```

### Important Evaluation Principle

The **test set should remain untouched during model selection**.

Model and hyperparameter decisions should be made using the training and validation data. The held-out test set is reserved for the final evaluation.

If the selected model is refit after selection, the professional approach is to optionally train it on **train + validation** using the already-selected hyperparameters and then evaluate it once on the untouched test set.

This separation helps reduce the risk of test-set leakage and provides a more reliable estimate of final model performance.

---

## Running the Project Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd lumpy-skin-disease-ml
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Streamlit Application

The application is organized into three main sections:

### 1. Single Prediction

A user enters the required feature values and receives a prediction with its probability.

### 2. Batch CSV Analysis

A user uploads a CSV containing the required model features. The application processes the rows and allows the prediction results to be exported.

### 3. About Model

Provides project information, model details, dataset information, features, and the project's limitations.

---

## Model Performance

The current application displays the following final test metrics:

- **Accuracy:** 98.15%
- **F1 Score:** 92.36%
- **ROC-AUC:** 99.30%

These values are presented as the project's current reported final evaluation metrics.

Performance should always be interpreted together with the dataset characteristics, class distribution, validation methodology, and real-world clinical/veterinary requirements.

---

## Technology Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Joblib**
- **Streamlit**
- **Matplotlib**
- **Jupyter Notebook**

---

## Reproducibility

The repository contains the trained model and the artifacts required by the Streamlit application.

For a complete training/retraining workflow, use the provided training script and notebook together with the project dataset.

When retraining or modifying the model, keep the following consistent:

1. Target definition
2. Feature names
3. Feature order
4. Train/validation/test methodology
5. Model-selection procedure
6. Prediction pipeline
7. Stored model artifacts

---

## Responsible Use

Lumpy Skin Disease is a veterinary disease, and machine learning predictions should not be treated as definitive diagnoses.

This project is intended for:

- Academic demonstration
- Machine learning research
- Portfolio development
- Decision-support experimentation

It should **not** be used as a substitute for:

- Veterinary diagnosis
- Laboratory testing
- Professional epidemiological assessment
- Official animal-health guidance

---

## Future Improvements

Possible future development areas include:

- More robust external validation
- Geographic cross-validation
- Model calibration
- Explainable AI / feature-level interpretation
- Better handling of class imbalance
- Larger and more diverse datasets
- Model monitoring after deployment
- Automated data validation
- More detailed risk visualization
- API-based deployment

---

## Author

**Abdelrahman Akl**

Machine Learning / AI Project

---

## License

This project is intended primarily for educational, academic, and portfolio purposes.

Add the appropriate license here if the repository is intended for public reuse.

---

## Deployment

The application is designed to be deployed as a Streamlit web application.

Recommended deployment flow:

```text
Local Project
     │
     ▼
GitHub Repository
     │
     ▼
Streamlit Deployment
     │
     ▼
Live Demo
```

## 🚀 Live Demo

👉 **[Try the Lumpy Skin Disease AI Live Demo](https://lumpy-skin-disease-ml.streamlit.app/)**

The application is deployed using Streamlit and provides real-time predictions through an interactive web interface.

Likewise, replace `<YOUR_GITHUB_REPOSITORY_URL>` in the local setup section with the actual repository URL after creating the GitHub repository.
