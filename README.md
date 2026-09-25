# Real-Time Card-Not-Present Fraud Detection for an E-Commerce Payment Gateway

A machine learning project for detecting fraudulent **card-not-present (CNP) e-commerce transactions** using Support Vector Machines (SVM). The project explores linear and non-linear kernels, hyperparameter optimization, recursive feature elimination, and evaluation metrics designed for highly imbalanced fraud-detection problems.

## 📌 Project Overview

Online payment fraud is a rare-event classification problem where fraudulent transactions represent only a small portion of total transactions. In such scenarios, accuracy alone can be misleading because a model can achieve high accuracy simply by predicting most transactions as legitimate.

This project develops and evaluates an SVM-based fraud detection pipeline that focuses on:

* Fraud vs. legitimate transaction classification
* Handling severe class imbalance
* Numerical feature scaling and categorical encoding
* Linear, RBF, and polynomial SVM kernels
* Stratified cross-validation
* Hyperparameter tuning using `GridSearchCV`
* Recursive Feature Elimination (RFE)
* Precision, Recall, F1-score, PR-AUC, and ROC-AUC
* Training-time comparison between feature configurations

---

## 🎯 Objectives

The main objectives of this project are to:

1. Analyze the characteristics and imbalance of the transaction dataset.
2. Determine whether fraudulent and legitimate transactions are linearly separable.
3. Build a baseline linear SVM classifier.
4. Compare linear, RBF, and polynomial SVM kernels.
5. Tune `C` and `gamma` using stratified cross-validation.
6. Apply Recursive Feature Elimination (RFE) for feature selection.
7. Compare model performance before and after feature selection.
8. Evaluate the final model using fraud-focused metrics.
9. Discuss practical considerations such as false positives, false negatives, and decision thresholds.

---

## 📊 Dataset

The project uses:

`credit_card_fraud_dataset.csv`

Dataset characteristics:

* **1,500 transactions**
* **15 columns**
* Binary target variable: `Class`
* `Class = 0` → Legitimate transaction
* `Class = 1` → Fraudulent transaction
* **1,410 legitimate transactions**
* **90 fraudulent transactions**
* Fraud rate: **6.00%**
* Approximate class imbalance: **1:15**

The dataset is a user-provided synthetic CNP transaction dataset that is structurally comparable to the Kaggle Credit Card Fraud Detection dataset.

### Features

The transaction data contains attributes such as:

* Transaction amount
* Merchant category
* Card type
* Customer age
* Distance from home
* Distance from previous transaction
* Ratio to median purchase price
* Repeat retailer indicator
* Chip usage
* PIN usage
* Online-order indicator
* Transaction timestamp

Identifiers such as `TransactionID` and `CustomerID` are removed from the modeling pipeline because they do not provide generalizable transaction behavior.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Imbalanced-learn
* Google Colab / Jupyter Notebook

### Machine Learning Techniques

* Support Vector Machine (SVM)
* Linear Kernel
* RBF Kernel
* Polynomial Kernel
* GridSearchCV
* Stratified K-Fold Cross-Validation
* Recursive Feature Elimination (RFE)
* Feature Scaling
* One-Hot Encoding
* Precision-Recall Analysis

---

## 🔄 Machine Learning Pipeline

```text
Transaction Dataset
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Class Imbalance Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Train/Test Split
        │
        ▼
Scaling + One-Hot Encoding
        │
        ▼
Baseline Linear SVM
        │
        ▼
Kernel Comparison
 ┌──────┼───────┐
 ▼      ▼       ▼
Linear  RBF    Polynomial
        │
        ▼
Hyperparameter Tuning
     C + Gamma
        │
        ▼
Recursive Feature Elimination
        │
        ▼
Final Evaluation
        │
        ▼
Precision / Recall / F1
        │
        ▼
PR-AUC / ROC-AUC
```

---

## 🔍 Exploratory Data Analysis

The initial analysis confirms that fraud represents only a small percentage of all transactions.

The dataset contains:

| Class      | Transactions |
| ---------- | -----------: |
| Legitimate |        1,410 |
| Fraud      |           90 |
| Total      |        1,500 |

Because of this imbalance, accuracy is not treated as the primary evaluation metric.

The analysis also examines transaction characteristics including:

* Transaction amount
* Distance from the previous transaction
* Ratio to median purchase price
* Chip usage
* PIN usage
* Online ordering behavior
* Merchant category
* Card type

The analysis indicates substantial overlap between fraudulent and legitimate transactions across individual features, motivating the investigation of non-linear SVM kernels.

---

## ⚙️ Preprocessing

The preprocessing pipeline performs the following operations:

### 1. Feature engineering

The transaction timestamp is converted into a transaction-hour feature (`TxnHour`).

### 2. Identifier removal

`TransactionID` and `CustomerID` are excluded from the modeling features.

### 3. Stratified train/test split

The data is divided into:

* Training set: **1,125 transactions**
* Test set: **375 transactions**

Stratification preserves the fraud/legitimate ratio in both datasets.

### 4. Numerical feature scaling

Numerical features are standardized because SVM decision boundaries are sensitive to feature scale.

### 5. Categorical encoding

Categorical features such as `MerchantCategory` and `CardType` are converted using one-hot encoding.

The preprocessing steps are integrated into a Scikit-learn `Pipeline` and `ColumnTransformer` to prevent data leakage during cross-validation.

---

## 🤖 SVM Kernel Comparison

Three SVM kernels are evaluated using stratified 5-fold cross-validation.

| Kernel     | CV Precision | CV Recall |  CV F1 |
| ---------- | -----------: | --------: | -----: |
| Linear     |       0.8992 |    0.9275 | 0.9075 |
| RBF        |       0.9453 |    0.9418 | 0.9416 |
| Polynomial |       0.9750 |    0.8692 | 0.9127 |

The RBF kernel provides the strongest overall F1 score among the compared kernels, while the polynomial kernel achieves higher precision but lower recall.

---

## 🎛️ Hyperparameter Tuning

The RBF SVM is further optimized using `GridSearchCV`.

The primary hyperparameters explored are:

* `C`
* `gamma`

The best configuration obtained from cross-validation was:

```text
C = 1
gamma = 0.1
```

Best cross-validation F1-score:

```text
0.9490
```

The tuning process took approximately:

```text
4.2 seconds
```

---

## 🧩 Feature Selection with RFE

Recursive Feature Elimination (RFE) is used to investigate whether the model can maintain useful fraud-detection performance with fewer features.

The transformed feature space initially contains:

```text
24 features
```

RFE retains:

```text
12 features
```

Selected features include:

* `Amount`
* `DistanceFromHome_km`
* `DistanceFromLastTransaction_km`
* `RatioToMedianPurchasePrice`
* `RepeatRetailer`
* `UsedChip`
* `UsedPIN`
* `OnlineOrder`
* `MerchantCategory_gas_station`
* `MerchantCategory_healthcare`
* `CardType_Amex`
* `CardType_Mastercard`

### Full vs. RFE Feature Set

| Configuration    | Features | Precision | Recall |     F1 | Training Time |
| ---------------- | -------: | --------: | -----: | -----: | ------------: |
| Full feature set |       24 |    0.9500 | 0.8636 | 0.9048 |      0.0196 s |
| RFE-reduced      |       12 |    0.8696 | 0.9091 | 0.8889 |      0.0069 s |

RFE substantially reduces the number of features and training time, while changing the precision-recall trade-off.

---

## 📈 Final Evaluation

The tuned SVM is evaluated on the held-out test set.

### Classification Performance

```text
              precision    recall    f1-score    support

Legit            0.99       1.00       0.99        353
Fraud            0.95       0.86       0.90         22

Accuracy                                    0.99
```

Additional evaluation metrics:

```text
PR-AUC  : 0.9781
ROC-AUC : 0.9983
```

The fraud-class metrics are particularly important because the dataset is imbalanced.

---

## 📊 Why PR-AUC Matters

In fraud detection, accuracy can provide an overly optimistic picture.

For example, because only 6% of the transactions are fraudulent, a model that predicts every transaction as legitimate would still achieve approximately 94% accuracy while detecting **zero fraud**.

Therefore, this project emphasizes:

* **Precision**: How many transactions flagged as fraud are actually fraudulent?
* **Recall**: How much of the actual fraud does the model detect?
* **F1-score**: Balance between precision and recall
* **PR-AUC**: Overall precision-recall performance across different thresholds

ROC-AUC is also reported, but PR-AUC is particularly informative for this rare-event classification problem.

---

## 💳 False Positives vs. False Negatives

A fraud detection system must balance two types of mistakes.

### False Positive

A legitimate transaction is incorrectly flagged as fraudulent.

Potential consequences:

* Customer payment friction
* Declined or challenged transactions
* Cart abandonment
* Additional support requests

### False Negative

A fraudulent transaction is classified as legitimate.

Potential consequences:

* Financial loss
* Chargebacks
* Payment-network penalties
* Increased fraud exposure

The appropriate operating threshold therefore depends on the risk tolerance and business requirements of the payment gateway.

---

## 🚀 Potential Production Approach

A production implementation could use the trained SVM as part of a real-time transaction scoring service.

A simplified architecture could be:

```text
Customer Payment
       │
       ▼
Payment Gateway
       │
       ▼
Feature Extraction
       │
       ▼
Fraud Detection Model
       │
       ▼
Risk Score
   ┌───┴────┐
   ▼        ▼
Approve   Review/Block
```

Instead of relying only on a fixed classification threshold, the SVM decision score could be used to define different risk zones:

```text
Low Risk        → Approve
Medium Risk     → Manual Review / Additional Verification
High Risk       → Block or Challenge
```

The exact threshold should be determined using business costs, fraud-loss targets, and operational requirements.

---

## 📁 Project Structure

```text
real-time-cnp-fraud-detection/
│
├── README.md
├── fraud_detection_svm.ipynb
├── credit_card_fraud_dataset.csv
└── requirements.txt
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/real-time-cnp-fraud-detection.git
cd real-time-cnp-fraud-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or directly install the required libraries:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn imbalanced-learn
```

### 3. Open the notebook

The project can be run using:

* Google Colab
* Jupyter Notebook
* JupyterLab

### 4. Upload the dataset

Place:

```text
credit_card_fraud_dataset.csv
```

in the notebook's working directory.

Run the notebook cells sequentially to reproduce the analysis, training, feature selection, and evaluation.

---

## 📦 Requirements

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
imbalanced-learn
```

---

## 🔮 Future Improvements

Potential extensions include:

* Testing additional imbalance-handling techniques such as SMOTE
* Cost-sensitive threshold optimization
* Real-time API deployment using FastAPI or Flask
* Model serialization using `joblib`
* Docker-based deployment
* Monitoring for data and concept drift
* Periodic model retraining
* Testing ensemble models such as XGBoost or Random Forest
* Adding transaction-level risk scoring
* Integrating manual-review workflows
* Evaluating performance on a larger real-world dataset

---

## 👨‍💻 Author

**Md Shadaan Ashraf**

B.Tech CSE (AI/ML)
VIT Bhopal University

---

## 📜 Disclaimer

This project is intended for educational and experimental purposes. The dataset used is a synthetic CNP transaction dataset and should not be treated as representative of production payment data.

The reported model performance should therefore not be interpreted as evidence of real-world fraud-detection performance without validation on appropriate production-scale datasets.
