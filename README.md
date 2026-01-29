Heart Disease Prediction using Machine Learning
1. Problem Statement
Heart disease is a serious medical problem and missing a patient who actually has the disease can be very dangerous.
The goal of this project is to build a machine learning model that can predict whether a person has heart disease or not using medical information like age, blood pressure, cholesterol, etc.
Since this is a medical problem, recall for disease cases is more important than just accuracy. So in this project, I focused not only on accuracy but also on ROC-AUC and recall.

2. Dataset
I used the Heart Disease dataset (UCI).
Each row represents one patient and columns contain medical details like:
age, sex
chest pain type (cp)
resting blood pressure (trestbps)
cholesterol (chol)
max heart rate (thalch)
exercise induced angina (exang)
oldpeak, slope, etc.
Target column:
num = 0 → No heart disease
num = 1 → Has heart disease
Originally, the dataset was dirty and had many missing values, so proper data cleaning was required.

3. What I Did (ML Pipeline)
I followed a complete machine learning pipeline:
Loaded the dataset
Studied the columns and data types
Removed useless columns like id
Dropped columns with too many missing values (ca, thal)
COnverted target num into binary (0 = no disease, 1 = disease)
Filled missing values:
Numerical columns → filled with median
Categorical columns → filled with mode
Converted categorical columns into numbers using One-Hot Encoding
Split data into:
X (features)
y (target)
Did train-test split with stratify
Applied StandardScaler for feature scaling
Trained multiple models:
Logistic Regression
SVM
KNN
Naive Bayes
Decision Tree
Compared models using accuracy
Selected top models and evaluated using:
Confusion Matrix
Classification Report
ROC-AUC
Selected SVM as final model
Did threshold tuning to increase recall for disease patients

4. Model Comparison (Accuracy)
Model	Accuracy
SVM	~86%
KNN	~85%
Naive Bayes	~85%
Logistic Regression	~82%
Decision Tree	~79%

6. Why Accuracy is Not Enough
This is a medical problem.
If the model says:
"No disease"
but the patient actually has disease → this is very dangerous.
So I focused more on:
Recall for disease class (1)
ROC-AUC score

7. Final Model Evaluation (SVM)
Confusion Matrix (Default Threshold = 0.5)
Disease Recall ≈ 91%
ROC-AUC ≈ 0.92
This means the model is already good, but it is still missing some disease patients.

8. Threshold Tuning (Important Part)
By default, models use threshold = 0.5.
I tried lowering the threshold to make the model more sensitive to disease.
At threshold = 0.1:
Disease Recall ≈ 99%
Missed patients reduced from 9 to only 1
Accuracy reduced, but medical safety increased
This shows that:
  -> In real life, the decision threshold should be chosen based on risk, not just accuracy.

9. Final Conclusion
SVM was selected as final model because:
   -> Best ROC-AUC
   -> Best recall for disease class
By tuning the threshold, I made the model much safer for medical use.
This project taught me that:
   -> There is no “best model”. There is only a best decision based on the problem.

10. What I Learned from This Project
How to clean real dirty data
How to handle missing values properly
How to encode categorical data
How to compare multiple ML models
Why ROC-AUC is important
How threshold tuning changes real-world decisions
How ML is used in real medical decision systems

11. How to Run
Open the notebook file
Run all cells from top to bottom
All results will be generated inside the notebook
