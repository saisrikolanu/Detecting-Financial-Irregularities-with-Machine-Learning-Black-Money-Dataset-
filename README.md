# Detecting Financial Irregularities with Machine Learning (Black Money Dataset)

**Course Project**  
**Authors:** Sai Sri Kolanu (50594437), Jyothsna Devi Goru (50560456), Hamsika Rajeshwar Rao (50613199)  
**University at Buffalo**  

---

## 📖 Overview
This project analyzes irregular global financial transactions using the **Black Money Dataset**. The objective is to identify suspicious transactions (e.g., money laundering) by applying **data preprocessing, exploratory data analysis (EDA), and classification models**.  

Key goals:
- Clean and preprocess the dataset  
- Explore relationships between transaction amount, risk score, shell company involvement, and industries  
- Train and compare multiple machine learning models  

---

## 📊 Dataset
- **Source:** [Global Black Money Transactions Dataset (Kaggle)](https://www.kaggle.com/datasets/waqi786/global-black-money-transactions-dataset)  
- **Size:** 10,000 rows × 14 columns  
- **Important Columns:** Transaction ID, Amount (USD), Transaction Type, Country, Destination Country, Tax Haven Country, Industry, Financial Institution, Money Laundering Risk Score, Shell Companies.  
- **Goal:** Predict high-risk vs low-risk transactions.  

---

## ⚙️ Data Preprocessing
1. Handle missing values (drop/median imputation).  
2. Remove duplicates.  
3. Standardize date formats and remove inconsistencies.  
4. Outlier detection (boxplots, IQR).  
5. Risk category labeling (High/Medium/Low).  
6. One-hot encoding of categorical variables.  
7. Non-ASCII cleanup in text fields.  
8. Sorting/organizing by country.  

---

## 📊 Exploratory Data Analysis
- **Correlation heatmap** (Amount, Risk Score, Shell Companies).  
- **Risk score distribution**.  
- **Top 20 countries by transaction volume**.  
- **Industries & financial institutions by transaction size**.  
- **Average risk scores by industry & transaction type**.  
- **Impact of shell companies on risk score**.  
- **Legal vs Illegal transaction breakdown by country**.  

---

## 🤖 Machine Learning Models
- **Logistic Regression** – Accuracy: ~86.9%, balanced Precision/Recall (best model).  
- **Naive Bayes** – Accuracy: ~68%, struggled with high-risk detection.  
- **Support Vector Machine (SVM)** – Accuracy: ~68%, imbalanced performance.  
- **KNN (K-Nearest Neighbor)** – Accuracy: ~61%, weaker on high-risk class.  
- **Multilayer Perceptron (MLP)** – Accuracy: ~68%, failed to capture positive class.  
- **Stochastic Gradient Descent (SGD)** – Accuracy: ~68%, imbalanced predictions.  

Metrics used: **Accuracy, Precision, Recall, F1 Score, Confusion Matrix**  

---

## 📈 Results
- **Best Model:** Logistic Regression (Accuracy ~86.9%)  
- Naive Bayes & SVM performed poorly on minority (high-risk) class.  
- KNN had decent recall but lower precision.  
- MLP and SGD underperformed, biased toward low-risk class.  
- Logistic Regression offered the best trade-off between interpretability and performance.  

---

## 🛠️ Tech Stack
- **Python 3.x**
- **NumPy, Pandas**
- **scikit-learn**
- **Matplotlib, Seaborn**

---

## ▶️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/black-money-ml.git
   cd black-money-ml
