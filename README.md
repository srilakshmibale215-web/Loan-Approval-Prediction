# 🏦 Loan Approval Prediction

## 📌 Project Overview

Loan Approval Prediction is a Machine Learning project that predicts whether a loan application is likely to be approved or rejected based on applicant details.

The project uses a Random Forest Classification algorithm and provides a simple web interface using Streamlit.

## 🎯 Objective

The main objective of this project is to build a machine learning model that can predict loan approval based on factors such as:

- Number of Dependents
- Education
- Self Employment
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets
- Commercial Assets
- Luxury Assets
- Bank Assets

## 🧠 Machine Learning Algorithm

**Random Forest Classifier**

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to make predictions.

## 📊 Model Accuracy

The model achieved approximately:

**97.66% Test Accuracy**

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest
- Joblib
- Streamlit

## 📁 Project Structure

```text
Loan Project
│
├── app
│   └── app.py
│
├── dataset
│   └── loan_approval_dataset.csv
│
├── model
│   └── loan_approval_model.pkl
│
├── notebooks
│   └── train_model.py
│
├── check_accuracy.py
│
└── README.md