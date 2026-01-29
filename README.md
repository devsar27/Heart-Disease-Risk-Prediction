❤️ Heart Disease Prediction using Machine Learning
📌 Problem Statement

Heart disease is a serious medical condition and missing a patient who actually has the disease can be very dangerous.
The goal of this project is to build a Machine Learning model that predicts whether a person has heart disease or not using medical attributes.

Since this is a medical problem, the focus is not only on accuracy, but mainly on:

Recall for disease patients

ROC-AUC score

📂 Dataset

Dataset: Heart Disease (UCI)

Each row represents one patient

Target column: num

0 → No heart disease

1 → Has heart disease

The dataset originally contained:

Missing values

Categorical features

Irrelevant columns

So proper data cleaning and preprocessing was required.

🛠️ Tech Stack

Python

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

🔁 Machine Learning Pipeline

Data Loading

Data Understanding

Data Cleaning:

Dropped useless columns (id)

Dropped columns with too many missing values (ca, thal)

Converted target to binary (0/1)

Filled missing values:

Numerical → Median

Categorical → Mode

Feature Engineering:

One-Hot Encoding for categorical features

Train-Test Split (with stratify)

Feature Scaling using StandardScaler

Model Training:

Logistic Regression

SVM

KNN

Naive Bayes

Decision Tree

Model Comparison using Accuracy

Detailed Evaluation using:

Confusion Matrix

Classification Report

ROC-AUC

Threshold Tuning for medical safety

Final Model Selection

🤖 Model Performance (Accuracy)
Model	Accuracy
🥇 SVM	86%
KNN	85%
Naive Bayes	85%
Logistic Regression	82%
Decision Tree	79%
📊 Final Model Evaluation (SVM)

ROC-AUC ≈ 0.92

Disease Recall (threshold = 0.5) ≈ 91%

This means the model is already good, but it still misses some disease patients.

🎚 Threshold Tuning (Important)

Since this is a medical problem, missing a disease case is very dangerous.

By lowering the decision threshold to 0.1:

Disease Recall ≈ 99%

Missed patients reduced from 9 to 1

Accuracy decreased, but medical safety increased

This shows that:

In real-world medical systems, recall is more important than accuracy.

🏆 Final Model

✅ Support Vector Machine (SVM)

Reasons:

Best ROC-AUC

Best recall for disease class

Stable and reliable performance

Works well after threshold tuning

📌 Conclusion

This project demonstrates a complete end-to-end Machine Learning pipeline on a real, dirty medical dataset including:

Data cleaning

Feature engineering

Model comparison

ROC-AUC based evaluation

Threshold tuning for real-world medical decision making

🚀 How to Run

Open the notebook file

Run all cells from top to bottom

All results will be generated inside the notebook
