# 🫀 Heart Disease Prediction Using Machine Learning

This project builds a machine learning model to **predict the presence of heart disease** based on patient medical attributes. It uses the **UCI Heart Disease dataset**, performs preprocessing, visualization, and trains a **Random Forest Classifier** for binary classification.

---

## 📌 Project Objective

Develop a classification model to detect heart disease and visualize key insights from the dataset to assist in early medical diagnosis.

---

## 🧠 Key Concepts Covered

- Supervised Machine Learning (Classification)
- Data Cleaning & Preprocessing
- Label Encoding & Feature Scaling
- Random Forest Algorithm
- Model Evaluation: Accuracy, Confusion Matrix, ROC-AUC

---

## 🗂️ Dataset Details

- 📁 Source: UCI Heart Disease Dataset
- 🔢 Rows: ~920
- 📊 Features: age, sex, chest pain type, blood pressure, cholesterol, max heart rate, and more
- 🎯 Target: `1` (disease), `0` (no disease)

---

## ⚙️ Technologies & Libraries

- **Python**
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `sklearn` (scikit-learn)

---

## 🧪 Model Pipeline

### 1. **Data Preprocessing**
- Renamed `num` column to `target` and binarized values
- Dropped irrelevant columns (`id`)
- Filled missing values:
  - Categorical → Mode
  - Numeric → Median
- Label encoded object-type columns
- Feature scaling using `StandardScaler`

### 2. **Exploratory Data Analysis**
- Correlation heatmap
- Countplot of heart disease cases
- Age distribution histogram by disease status

### 3. **Model Building**
- Algorithm: `RandomForestClassifier`
- Train-Test Split: 80/20
- Prediction & Probability estimation

### 4. **Evaluation Metrics**
- Accuracy
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- ROC Curve and AUC Score

---

## 📈 Results

- ✅ High accuracy on test data
- 📊 Strong performance metrics (Precision/Recall)
- 📉 ROC Curve shows good separation ability

---

## 📌 Future Improvements

- Hyperparameter tuning (GridSearchCV)
- Try other models (e.g., XGBoost, Logistic Regression)
- Feature importance analysis
- Deployment via Streamlit or Flask

---

## 📁 Project Structure

