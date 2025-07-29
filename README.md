# Customer-Insurance-Purchase-Prediction-Using-Machine-Learning-Classification-Algorithms
# 🛡️ Customer Insurance Purchases Prediction - ML Case Study

## 📌 Project Overview

In this project, we act as analysts for a Bank Insurance Company. The main goal is to build machine learning models that can **predict whether a new customer will purchase insurance** based on their **Age** and **Estimated Salary**.

We perform a **comparative analysis** using multiple ML classification algorithms to identify the most effective model that balances both accuracy and generalization (avoiding overfitting).

---

## 📊 Problem Statement

Use customer data (excluding personal information) to train machine learning classifiers and evaluate their performance. The objective is to select the best-performing model based on metrics such as accuracy, precision, recall, F1-score, and ROC AUC.

---

## 🔍 Dataset

- Source: `Social_Network_Ads.csv`
- Features used:
  - `Age`
  - `EstimatedSalary`
- Target:
  - `Purchased` (1 = Purchased Insurance, 0 = Not Purchased)

---

## ⚙️ Machine Learning Algorithms Used

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Support Vector Machine (SVM)**
4. **Decision Tree**
5. **Random Forest**

---

## 🧪 Steps Followed

1. **Data Preprocessing**
   - Load and explore dataset
   - Feature scaling (Standardization)
   - Train-test split (80-20)
2. **Model Training**
   - Fit each algorithm on training data
3. **Evaluation Metrics**
   - Accuracy
   - Confusion Matrix
   - Classification Report
   - ROC AUC Curve
4. **Prediction & Visualization**
   - Predict insurance purchase for custom Age & Salary values
   - Graphical analysis of decision boundaries

---

## 🎯 Graphical Predictions

We visualize model decision boundaries and test predictions for:

- Age: 30, Salary: 87,000
- Age: 40, Salary: None
- Age: 40, Salary: 100,000
- Age: 50, Salary: None
- Age: 18, Salary: None
- Age: 22, Salary: 600,000
- Age: 35, Salary: 2,500,000
- Age: 60, Salary: 100,000,000

---

## 📌 Hypotheses & Assumptions Tested

- Younger individuals with high salaries are more likely to purchase insurance.
- Salary may influence insurance purchase decisions more than age.
- Older users with high income might still avoid buying insurance.

---

## 📈 Final Output

- Tabular comparison of model performance
- Plots showing decision boundaries and prediction zones
- A final chosen model for deployment
- Code to predict on new data
- Assumptions validated through AI simulations

---

## 💡 Lessons Learned

- Logistic Regression and Random Forest are effective for binary classification.
- Importance of scaling for algorithms like KNN and SVM.
- Visualizations help in understanding model behavior clearly.

---

## 🚀 Real-life Use Cases

1. **Marketing Campaigns for Insurance**
   - Target customers more likely to buy insurance based on age and income.

2. **Bank Loan Approvals**
   - Predict customer creditworthiness using similar classification strategies.

---

## 🛠️ Tech Stack

- Python 3.x
- Jupyter Notebook
- Libraries:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`

---
