# synthetic-clinical-mlflow-project
End-to-end clinical classification project


readme_content = """
# Synthetic Clinical Diagnosis Classification with MLflow

## Project Overview

This project develops a multiclass machine learning model to predict a patient's diagnosis using synthetic clinical data.

The project includes:

- Data preprocessing
- Exploratory data analysis
- Model comparison
- Cross-validation
- Hyperparameter tuning
- MLflow experiment tracking
- Model registration
- Model versioning
- Automatic champion model replacement
- REST API model serving
- Real-time inference testing

---

## Dataset

The project uses a synthetic clinical tabular dataset containing 10,000 patient records and 14 columns.

### Dataset Shape

```text
(10000, 14)


Columns:
patient_id
age
sex
bmi
systolic_bp
diastolic_bp
glucose
cholesterol
creatinine
diabetes
hypertension
diagnosis
readmission_30d
mortality


Target Variable: The target variable is: diagnosis
The patient_id column was removed because it is only an identifier and does not provide useful predictive information.

Project Structure:
synthetic-clinical-mlflow-project/
│
├── data/
│   └── synthetic_clinical_data.csv
│
├── notebooks/
│   └── clinical_diagnosis_mlflow.ipynb
│
├── mlflow.db
│
├── README.md
│
└── requirements.txt

Data Preprocessing:
The preprocessing pipeline performs the following steps:
Removes the patient_id column
Separates features and target
Handles numerical and categorical features
Standardizes numerical columns using StandardScaler
Encodes the sex column using OneHotEncoder
Uses ColumnTransformer to combine preprocessing steps
Uses Pipeline to connect preprocessing and classification

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numerical_columns),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


Models Compared:
The following classification algorithms were trained and evaluated:
Logistic Regression
Decision Tree
Random Forest
Naive Bayes
Support Vector Machine
K-Nearest Neighbors

Initial Model Results:
| Model                  | Accuracy | Weighted F1 | Macro F1 |
| ---------------------- | -------: | ----------: | -------: |
| Logistic Regression    |   0.5590 |      0.4009 |   0.1793 |
| Support Vector Machine |   0.5590 |      0.4009 |   0.1793 |
| Random Forest          |   0.5575 |      0.4059 |   0.1864 |
| Naive Bayes            |   0.5475 |      0.4089 |   0.2059 |
| K-Nearest Neighbors    |   0.4625 |      0.4051 |   0.2267 |
| Decision Tree          |   0.3675 |      0.3722 |   0.2471 |


Baseline Model:
A DummyClassifier using the most frequent class was evaluated as a baseline.
Baseline Accuracy: 0.5590
The baseline showed that the dataset is imbalanced because always predicting the majority diagnosis class produced the same accuracy as the initial Logistic Regression model.


Cross-Validation:
Stratified 5-fold cross-validation was used to compare model stability.

| Model                  | Mean CV Accuracy | Standard Deviation |
| ---------------------- | ---------------: | -----------------: |
| Logistic Regression    |           0.5590 |             0.0000 |
| Support Vector Machine |           0.5586 |             0.0006 |
| Random Forest          |           0.5561 |             0.0012 |
| Naive Bayes            |           0.5454 |             0.0026 |
| K-Nearest Neighbors    |           0.4675 |             0.0050 |
| Decision Tree          |           0.3765 |             0.0035 |

Logistic Regression was selected as the initial production model because it achieved the highest mean cross-validation accuracy and showed stable performance across folds.


MLflow Experiment Tracking
MLflow was used to track:
Model parameters
Accuracy
Weighted F1 score
Macro F1 score
Model artifacts
Input examples
Model signatures
Registered model versions
The MLflow tracking server was started using:

mlflow server \
  --host 127.0.0.1 \
  --port 5050 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts

The tracking URI was configured in Python:

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5050")



Version 1: Initial Champion Model
The initial Logistic Regression pipeline was registered in MLflow using the model name:
clinical-diagnosis-classifier
Version 1 was assigned the alias: champion
The alias allows the application to load the current production model without hard-coding a model version number.

model_uri = (
    "models:/clinical-diagnosis-classifier@champion"
)


Model Improvement:
The initial model was improved using:
class_weight="balanced"
Hyperparameter tuning
GridSearchCV
Stratified 5-fold cross-validation
Macro F1 as the optimization metric
Macro F1 was selected because it gives equal importance to every diagnosis class and is more informative for imbalanced multiclass classification.


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

improved_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    )
])

param_grid = {
    "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["lbfgs", "liblinear"]
}

grid_search = GridSearchCV(
    estimator=improved_pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

| Metric      | Version 1 | Version 2 |
| ----------- | --------: | --------: |
| Accuracy    |    0.5590 |    0.2365 |
| Weighted F1 |    0.4009 |    0.2626 |
| Macro F1    |    0.1793 |    0.2116 |

Version 2 improved Macro F1 from: 0.1793 to 0.2116
Because Macro F1 was defined as the model promotion metric, Version 2 was registered and promoted to the champion alias.

Automatic Model Replacement
The candidate model was automatically compared with the previous champion.

if new_macro_f1 > old_macro_f1:
    client.set_registered_model_alias(
        name="clinical-diagnosis-classifier",
        alias="champion",
        version=new_version
    )

After promotion, the MLflow Model Registry showed: @champion -> Version 2
This means Version 2 automatically replaced Version 1 as the active champion model.

Real-Time Inference
A sample patient record was sent to the REST API.

import requests

sample_patient = X_test.iloc[[0]]

payload = {
    "dataframe_split": {
        "columns": sample_patient.columns.tolist(),
        "data": sample_patient.values.tolist()
    }
}

response = requests.post(
    "http://127.0.0.1:5052/invocations",
    json=payload,
    timeout=30
)

print("Status code:", response.status_code)
print("Prediction:", response.json())

Output:
Status code: 200
Prediction: {'predictions': ['Sepsis']}

The HTTP status code 200 confirms that the model was successfully served and accepted the inference request.


Final Outcome
The project successfully completed the following requirements:
Compared multiple machine learning models.
Used cross-validation to select a stable model.
Registered the selected model in MLflow.
Assigned a champion alias to the production model.
Exposed the model through an MLflow REST API.
Improved the model using balanced class weights and GridSearchCV.
Registered the improved model as Version 2.
Automatically moved the champion alias to Version 2.
Successfully tested real-time inference.


Technologies Used:
Python
Pandas
NumPy
Scikit-learn
MLflow
Matplotlib
Jupyter Notebook
REST API
SQLite
Uvicorn


Author
Sunandana Sahoo
M.Sc. Applied Data Science and Artificial Intelligence
SRH University of Applied Sciences, Hamburg
