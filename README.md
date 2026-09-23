# End-to-End E-Commerce Customer Churn & Retention System

## 📌 Project Overview

Customer churn is one of the major challenges faced by e-commerce and customer-focused businesses. Losing an existing customer can reduce revenue, customer lifetime value, and long-term business growth.

This project develops an **end-to-end machine learning system for predicting customer churn**. The system analyzes customer demographic, financial, behavioral, engagement, complaint, and transaction-related information to estimate the probability that a customer will churn.

The project goes beyond simply training a classification model. It includes:

* Exploratory data preparation
* Domain-driven feature engineering
* Missing-value handling
* Categorical feature encoding
* Class-imbalance handling
* Logistic Regression baseline
* XGBoost classification
* Stratified 5-fold cross-validation
* Precision-Recall AUC evaluation
* F1-score optimization
* Decision-threshold tuning
* Final model retraining
* Churn probability prediction
* Model and preprocessing artifact saving

The final output can be used to identify customers who are at higher risk of leaving and support **data-driven customer retention strategies**.

---

## 🎯 Business Problem

A business may have thousands of customers, but not every customer has the same probability of remaining active.

The central business question is:

> **Which customers are most likely to churn, and how can the business identify them early enough to take retention action?**

Instead of treating every customer equally, a predictive churn system can assign each customer a **churn probability**.

This allows a business to potentially:

* Identify high-risk customers
* Prioritize retention campaigns
* Investigate customer complaints
* Improve customer engagement
* Target personalized offers
* Reduce unnecessary retention costs
* Protect customer lifetime value

---

## 🎯 Project Objective

The main objective is to build a machine learning classification system that predicts whether a customer is likely to churn.

### Target Variable

```text
churn
```

Where:

* `0` = Customer is not predicted as churned
* `1` = Customer is predicted as churned

The model also produces a continuous:

```text
churn_probability
```

This probability represents the model's estimated likelihood of churn and can be used for customer-risk prioritization.

---

# 🔄 Machine Learning Workflow

The project follows an end-to-end workflow:

```text
Raw Customer Data
        ↓
Data Preparation
        ↓
Feature Engineering
        ↓
Missing Value Handling
        ↓
Categorical Encoding
        ↓
Class Imbalance Handling
        ↓
Train / Validation Split
        ↓
Logistic Regression Baseline
        ↓
XGBoost + Cross-Validation
        ↓
Model Evaluation
        ↓
Threshold Optimization
        ↓
Final Model Training
        ↓
Churn Probability Prediction
        ↓
Saved Production Artifacts
```

---

# 📊 Dataset Features

The dataset contains different categories of customer information, including:

### 👤 Customer Demographics

Examples include:

* Gender
* Marital status
* Education level
* Occupation type
* Income band
* Income category
* City tier
* Region
* Customer segment

### 💳 Financial Information

Examples include:

* Annual income
* Current balance
* Credit utilization ratio
* Credit utilization 6-month average
* Customer lifetime value

### 📱 Customer Engagement

Examples include:

* Digital logins
* Branch visits
* Campaign responses
* Campaigns received

### 📞 Customer Service

Examples include:

* Total complaints
* Unresolved complaints
* Customer feedback sentiment

### 🏦 Relationship Information

Examples include:

* Relationship type
* Primary account type
* Card category
* Onboarding channel

---

# 🧠 Feature Engineering

A major part of this project is the creation of business-oriented features rather than relying only on the original columns.

## 1. Balance-to-Income Ratio

```python
balance_to_income_ratio =
    current_balance / annual_income
```

This provides a relative measure of a customer's balance compared with their income.

---

## 2. Credit Utilization Spread

```python
credit_utilization_spread =
    credit_utilization_ratio - credit_utilization_6m_avg
```

This captures how current credit utilization differs from the customer's recent average.

It can provide a signal about changes in financial behavior.

---

## 3. Digital-to-Branch Engagement Ratio

```python
digital_vs_branch_ratio =
    total_digital_logins / (branch_visit_count + 1)
```

This captures the relationship between digital and branch-based customer engagement.

---

## 4. Unresolved Complaint Ratio

```python
unresolved_complaint_ratio =
    unresolved_complaint_count / total_complaints
```

This represents the proportion of complaints that remain unresolved.

It provides a customer-service friction signal.

---

## 5. Campaign Conversion Rate

```python
campaign_conversion_rate =
    campaign_response_count / campaign_received_count
```

This measures how frequently a customer responds to campaigns.

---

## 6. Log Transformations

Log-transformed versions were created for highly skewed numerical variables such as:

* Annual income
* Total transaction amount
* Customer lifetime value
* Current balance

Using:

```python
np.log1p()
```

helps reduce the influence of extremely large values and can make skewed distributions easier for models to learn from.

---

# 🧹 Data Preprocessing

The preprocessing pipeline handles different feature types separately.

## Numerical Features

Missing numerical values are handled using:

```text
Median Imputation
```

## Categorical Features

Missing categorical values are handled using:

```text
Most-Frequent Imputation
```

Categorical variables are then transformed using:

```text
One-Hot Encoding
```

with:

```python
handle_unknown='ignore'
```

This prevents unseen categories in the test data from causing transformation errors.

---

# ⚖️ Handling Class Imbalance

Customer churn datasets commonly contain fewer churned customers than non-churned customers.

Ignoring this imbalance can cause a model to focus too heavily on the majority class.

This project calculates:

```python
scale_pos_weight =
    number_of_negative_samples /
    number_of_positive_samples
```

The resulting value is supplied to XGBoost through:

```python
scale_pos_weight
```

This gives greater importance to the minority churn class during model training.

---

# 🤖 Models

## 1. Logistic Regression

Logistic Regression is used as the baseline model.

```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)
```

The baseline provides a simple reference point against which the more advanced XGBoost model can be evaluated.

---

## 2. XGBoost

The main predictive model is:

```text
XGBoost Classifier
```

The model uses parameters including:

```text
n_estimators = 300
learning_rate = 0.05
max_depth = 5
subsample = 0.8
colsample_bytree = 0.8
```

Class imbalance is handled using the calculated `scale_pos_weight`.

---

# 🔬 Cross-Validation

The XGBoost model is evaluated using:

```text
Stratified 5-Fold Cross-Validation
```

Stratification is important because the target variable is imbalanced.

The dataset is divided into five folds while maintaining approximately the same churn/non-churn distribution in each fold.

The project records the **PR-AUC of each fold** and calculates the mean and standard deviation.

This provides a more reliable estimate of model performance than relying on a single split alone.

---

# 📈 Evaluation Metrics

Several metrics are used to evaluate the model.

### PR-AUC

Precision-Recall AUC is treated as the primary evaluation metric.

This is particularly useful for an imbalanced classification problem where the positive class—churn—is important.

### F1-Score

F1 combines:

```text
Precision + Recall
```

and is useful when both false positives and false negatives matter.

### Precision

Measures how many customers predicted as churners are actually churners.

### Recall

Measures how many actual churners the model successfully identifies.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across probability thresholds.

---

# 🎚️ Decision Threshold Optimization

The model does not simply rely on the default:

```text
0.50
```

classification threshold.

Instead, the validation probabilities are evaluated across different thresholds.

The project calculates:

```python
F1 = 2 × (Precision × Recall) /
     (Precision + Recall)
```

and selects the threshold that produces the highest validation F1-score.

This optimized threshold is then used when generating the final test predictions.

### Why this matters

A customer with:

```text
Churn Probability = 0.45
```

would normally be classified as non-churn under a 0.50 threshold.

However, if the optimized threshold is lower than 0.45, that customer may be classified as high risk.

Therefore, **the probability model and the business decision threshold are separate components**.

---

# 🏭 Final Production Model

After validation and threshold selection, the XGBoost model is retrained using the complete training dataset.

```text
100% of available training data
        ↓
Final XGBoost Model
        ↓
Unseen Test Data
        ↓
Churn Probability
        ↓
Final Churn Prediction
```

For every test customer, the system generates:

```text
churn_probability
predicted_churn
```

---

# 📦 Saved Model Artifacts

The project saves the trained components inside the:

```text
models/
```

directory.

### `preprocessor.pkl`

Contains the fitted preprocessing pipeline.

### `xgboost_model.pkl`

Contains the trained production XGBoost model.

### `model_metadata.pkl`

Stores important model information including:

* Optimal decision threshold
* Feature names
* Class-imbalance weight

This makes it possible to reuse the trained model without rebuilding the entire training process.

---

# 📁 Project Structure

```text
End-to-End-E-Commerce-Customer-Churn-Retention-System/
│
├── notebooks/
│   └── ecommerce_customer_churn.ipynb
│
├── models/
│   ├── preprocessor.pkl
│   ├── xgboost_model.pkl
│   └── model_metadata.pkl
│
├── README.md
│
└── .gitignore
```

---

# 🛠️ Technologies Used

| Technology       | Purpose                                      |
| ---------------- | -------------------------------------------- |
| Python           | Core programming language                    |
| Pandas           | Data manipulation                            |
| NumPy            | Numerical computation                        |
| Scikit-learn     | Preprocessing, baseline model and evaluation |
| XGBoost          | Main classification model                    |
| Matplotlib       | Data visualization                           |
| Joblib           | Model serialization                          |
| Jupyter Notebook | Development and experimentation              |
| Git              | Version control                              |
| GitHub           | Project repository                           |

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/dagihope950-tech/End-to-End-E-Commerce-Customer-Churn-Retention-System.git
```

```bash
cd End-to-End-E-Commerce-Customer-Churn-Retention-System
```

## 2. Install Dependencies

```bash
pip install numpy pandas scikit-learn xgboost matplotlib joblib jupyter
```

## 3. Prepare the Dataset

Place the training and test datasets in the appropriate location and update the dataset paths in the notebook.

The notebook expects:

```text
ChurnZero_dataset_v1.csv
ChurnZero_test_v1.csv
```

## 4. Run the Notebook

Open:

```text
notebooks/ecommerce_customer_churn.ipynb
```

and execute the cells sequentially.

---

# 💼 Business Application

The model can support a customer-retention workflow such as:

```text
Customer Data
      ↓
Churn Prediction
      ↓
Risk Probability
      ↓
Customer Risk Segmentation
      ↓
Retention Action
```

For example, a business could use churn probability to prioritize customers for:

* Personalized offers
* Customer-service follow-up
* Loyalty programs
* Targeted campaigns
* Product recommendations
* Complaint resolution
* Customer engagement campaigns

The model itself does **not** determine which retention strategy should be used. That decision requires business rules, campaign costs, customer value, and experimentation.

---

# 🔍 Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

* Binary classification
* Feature engineering
* Domain-driven feature creation
* Missing-value imputation
* One-hot encoding
* Class imbalance
* Stratified train/validation splitting
* Cross-validation
* XGBoost
* Precision-Recall curves
* PR-AUC
* ROC-AUC
* F1-score
* Threshold optimization
* Model retraining
* Model serialization
* Production artifact management

---

# 📌 Important Design Decisions

### Why XGBoost?

XGBoost is well suited to structured/tabular customer data and can model nonlinear relationships and interactions between customer attributes.

### Why PR-AUC?

Because churn classification can be imbalanced, PR-AUC provides useful information about the precision-recall trade-off for the positive class.

### Why optimize the threshold?

The default 0.50 threshold is not automatically the correct business decision threshold. Threshold selection allows the classification rule to be adjusted according to the desired precision-recall balance.

### Why use cross-validation?

Cross-validation provides multiple validation estimates instead of relying on one arbitrary training/validation split.

---

# ⚠️ Limitations

This project should be interpreted as a **predictive system**, not as a causal system.

A high churn probability does not prove that a particular factor caused a customer to leave.

Additional limitations include:

* Model performance depends on the quality and representativeness of the training data.
* The optimized threshold is selected using the validation data and may need to be reconsidered for a different business objective.
* Customer behavior can change over time, causing model drift.
* Probability estimates should be monitored and calibrated if they are used as direct risk estimates.
* Retention actions should be evaluated with controlled experiments rather than assuming that prediction alone will reduce churn.
* Production deployment would require monitoring, retraining policies, logging, and data-validation procedures.

---

# 🔮 Future Improvements

Potential improvements include:

### 1. Explainable AI

Add:

```text
SHAP
```

to explain why individual customers receive high churn probabilities.

### 2. Customer Risk Segmentation

Create groups such as:

```text
Low Risk
Medium Risk
High Risk
```

using explicitly defined probability thresholds.

### 3. Retention Recommendation System

Extend the project from:

```text
Who may churn?
```

to:

```text
What retention action should be considered?
```

### 4. Model Comparison

Evaluate additional algorithms such as:

* Random Forest
* LightGBM
* CatBoost
* Neural Networks

### 5. Hyperparameter Optimization

Use tools such as:

```text
Optuna
RandomizedSearchCV
GridSearchCV
```

for systematic tuning.

### 6. Deployment

The trained model could be exposed through:

```text
FastAPI / Django API
        ↓
Prediction Service
        ↓
Web Dashboard
```

### 7. Monitoring

A production version should monitor:

* Data drift
* Prediction drift
* Model performance
* Missing values
* Feature distributions
* Prediction volume
* Churn rate changes

---

# 👨‍💻 Author

**Dagim Tesfaye Guta**

Computer Science and Engineering Student

---

# ⭐ Project Summary

This project demonstrates how machine learning can be applied to a real-world customer-retention problem.

Rather than stopping at model training, the workflow covers the complete process:

```text
Business Problem
      ↓
Data Preparation
      ↓
Feature Engineering
      ↓
Imbalance Handling
      ↓
Baseline Modeling
      ↓
XGBoost
      ↓
Cross-Validation
      ↓
Evaluation
      ↓
Threshold Optimization
      ↓
Final Model
      ↓
Customer Churn Probabilities
      ↓
Reusable Model Artifacts
```

The resulting system provides a foundation for building a production-oriented **customer churn prediction and retention analytics solution**.

