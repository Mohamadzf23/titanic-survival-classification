# Titanic Survival Classification

A machine learning classification project that predicts whether a passenger survived the Titanic disaster using Python and Scikit-learn.

The project covers the complete machine learning workflow, including data preprocessing, exploratory analysis, feature engineering, model training, cross-validation, hyperparameter tuning, and model evaluation.

## Project Overview

The goal of this project is to build and compare multiple classification models for predicting passenger survival.

The workflow includes:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Feature selection and importance analysis
* Training multiple classification algorithms
* Cross-validation
* Hyperparameter tuning with `GridSearchCV`
* Model evaluation using multiple classification metrics
* Saving the final trained model

## Machine Learning Workflow

```text
Raw Data
    ↓
Data Cleaning
    ↓
Exploratory Data Analysis
    ↓
Feature Engineering
    ↓
Train / Test Split
    ↓
Model Training
    ↓
Cross Validation
    ↓
GridSearchCV
    ↓
Model Evaluation
    ↓
Final Model
```

## Models

The following classification algorithms were implemented and compared:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Gradient Boosting
* Support Vector Machine (SVM)
* Naive Bayes

## Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* Cross-Validation

Cross-validation was used to evaluate model stability, while `GridSearchCV` was used to tune model hyperparameters.

## Results

The tuned models were evaluated on a held-out test set.

| Model               | Test Accuracy |
| ------------------- | ------------: |
| Logistic Regression |        79.01% |
| Random Forest       |        78.63% |
| Decision Tree       |        78.24% |
| Gradient Boosting   |        78.24% |
| SVM                 |        77.48% |
| KNN                 |        77.10% |
| Naive Bayes         |        45.04% |

> Test accuracy is reported on the held-out test set used in this project. Other evaluation metrics are also considered during model analysis.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook

## Project Structure

```text
titanic-survival-classification/
│
├── models/
│   └── logistic_model.pkl
│
├── src/
│   └── titanic_survival_classification.ipynb
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Final Model

The final trained Logistic Regression model is saved using `joblib` in the `models/` directory.

```python
import joblib

model = joblib.load("models/logistic_model.pkl")
```

## Key Concepts Demonstrated

This project demonstrates practical experience with:

* Supervised learning
* Binary classification
* Feature engineering
* Categorical encoding
* Model comparison
* Cross-validation
* Hyperparameter tuning
* Data leakage awareness
* Model evaluation
* Feature importance
* Model persistence

## License

This project is licensed under the MIT License.


## Author

**Mohamadreza Zafari**