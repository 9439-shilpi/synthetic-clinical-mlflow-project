# Synthetic Clinical Diagnosis Classification using MLflow

## Project Overview

This project demonstrates an end-to-end Machine Learning workflow for multiclass clinical diagnosis prediction using a synthetic clinical dataset. The project covers data preprocessing, exploratory data analysis, model comparison, cross-validation, experiment tracking with MLflow, model versioning, automatic model promotion, REST API deployment, and real-time inference.

---

## Dataset

- **Dataset:** Synthetic Clinical Tabular Dataset
- **Records:** 10,000
- **Features:** 14 columns
- **Target Variable:** `diagnosis`

The `patient_id` column was removed because it is a unique identifier and does not contribute to prediction.

---

## Project Workflow

- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Train-test split (80:20)
- Feature scaling using StandardScaler
- Categorical encoding using OneHotEncoder
- Model comparison
- Baseline model evaluation
- Stratified 5-Fold Cross-Validation
- MLflow experiment tracking
- Model Registry and Versioning
- Hyperparameter tuning using GridSearchCV
- Automatic champion model replacement
- REST API model serving
- Real-time inference

---

# Model Development and Selection Process

## Stage 1: Initial Model Comparison

Six machine learning algorithms were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Models were compared using:

- Accuracy
- Weighted F1-score
- Macro F1-score

The initial evaluation showed that the **Decision Tree** achieved the highest **Macro F1-score (0.2471)**, indicating better performance across all diagnosis classes.

**Initial Selection:** Decision Tree

---

## Stage 2: Cross-Validation

To verify model stability, Stratified 5-Fold Cross-Validation was performed.

| Model | Mean CV Accuracy |
|-------------------------|----------------:|
| **Logistic Regression** | **0.5590** |
| Support Vector Machine | 0.5586 |
| Random Forest | 0.5561 |
| Naive Bayes | 0.5454 |
| K-Nearest Neighbors | 0.4675 |
| Decision Tree | 0.3765 |

Although Decision Tree achieved the highest Macro F1-score during the initial evaluation, its cross-validation performance was significantly lower.

Logistic Regression achieved the highest and most stable cross-validation accuracy and was therefore selected as the production model.

**Final Selection:** Logistic Regression

---

## Baseline Model

A DummyClassifier using the majority class was used as a baseline.

| Metric | Value |
|--------|------:|
| Accuracy | 0.5590 |

The baseline demonstrated that the dataset is imbalanced because always predicting the majority class produced the same accuracy as the initial Logistic Regression model.

---

## MLflow Model Registration

The selected Logistic Regression pipeline was registered in the MLflow Model Registry.

### Version 1

| Metric | Value |
|--------|------:|
| Accuracy | 0.5590 |
| Weighted F1 | 0.4009 |
| Macro F1 | 0.1793 |

Model Name:

```
clinical-diagnosis-classifier
```

The model was assigned the **champion** alias for deployment.

---

## Model Improvement

The initial Logistic Regression model was improved using:

- Balanced Logistic Regression (`class_weight="balanced"`)
- GridSearchCV
- Stratified 5-Fold Cross-Validation
- Macro F1 optimization

### Version Comparison

| Metric | Version 1 | Version 2 |
|--------|----------:|----------:|
| Accuracy | 0.5590 | 0.2365 |
| Weighted F1 | 0.4009 | 0.2626 |
| Macro F1 | **0.1793** | **0.2116** |

Although overall accuracy decreased, Macro F1 improved from **0.1793** to **0.2116**, indicating better performance across all diagnosis classes.

Since Macro F1 was selected as the promotion metric, the improved model was automatically promoted to the **champion** version in MLflow.

---

## MLflow Features

The project demonstrates the following MLflow capabilities:

- Experiment Tracking
- Model Registry
- Model Versioning
- Champion Alias
- Automatic Model Promotion
- Model Serving

---

## REST API Inference

The deployed model was served using the MLflow REST API.

Example response:

```text
Status Code: 200

Prediction:
{'predictions': ['Sepsis']}
```

A successful HTTP **200** response confirmed that the deployed model accepted inference requests successfully.

---

## Project Workflow Summary

```text
Synthetic Clinical Dataset
            │
            ▼
Data Preprocessing
            │
            ▼
Exploratory Data Analysis
            │
            ▼
Train Six Machine Learning Models
            │
            ▼
Initial Evaluation
            │
            ▼
Decision Tree Selected
(Highest Macro F1)
            │
            ▼
Stratified 5-Fold Cross-Validation
            │
            ▼
Logistic Regression Selected
(Best Cross-Validation Accuracy)
            │
            ▼
MLflow Experiment Tracking
            │
            ▼
Register Version 1
            │
            ▼
Champion Alias Assigned
            │
            ▼
Balanced Logistic Regression
            │
            ▼
GridSearchCV
            │
            ▼
Evaluate Improved Model
            │
            ▼
Register Version 2
            │
            ▼
Champion Alias Updated
            │
            ▼
REST API Deployment
            │
            ▼
Real-Time Prediction
```

---

## Project Structure

```text
synthetic-clinical-mlflow-project/
│
├── data/
│   └── synthetic_clinical_dataset.csv
│
├── notebooks/
│   └── clinical_diagnosis_mlflow.ipynb
│
├── mlartifacts/
│
├── README.md
│
└── requirements.txt
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- Matplotlib
- Jupyter Notebook
- REST API
- SQLite

---

## Results

- Successfully compared six machine learning models.
- Performed exploratory data analysis and preprocessing.
- Evaluated a baseline model.
- Used Stratified 5-Fold Cross-Validation for model selection.
- Registered models in MLflow.
- Implemented model versioning.
- Used the champion alias for deployment.
- Improved the model using Balanced Logistic Regression and GridSearchCV.
- Automatically promoted the improved model.
- Served the model through a REST API.
- Successfully performed real-time inference.

---

## Future Improvements

- Evaluate additional ensemble models (e.g., XGBoost, LightGBM).
- Improve minority-class recall without significantly reducing overall accuracy.
- Use a larger and more realistic clinical dataset.
- Deploy the model as a cloud-based service.

---

## MLflow Configuration

- **Tracking URI:** `http://127.0.0.1:5050`

## Author

**Sunandana Sahoo**

M.Sc. Applied Data Science & Artificial Intelligence

SRH University of Applied Sciences, Hamburg