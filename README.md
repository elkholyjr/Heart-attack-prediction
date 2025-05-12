
# 🫀 Heart Attack Prediction using Logistic Regression

This project aims to predict the likelihood of heart disease using machine learning, specifically Logistic Regression. It involves data preprocessing, feature engineering, training, and evaluating a classification model using the [Heart Disease dataset

---

## 📁 Dataset

The dataset (`heart.csv`) contains clinical parameters of patients such as:

- `age`, `sex`, `cp` (chest pain type)
- `trestbps` (resting blood pressure), `chol` (serum cholesterol)
- `fbs` (fasting blood sugar), `restecg` (resting electrocardiographic results)
- `thalach` (maximum heart rate achieved), `exang` (exercise-induced angina)
- `oldpeak`, `slope`, `ca` (number of major vessels), `thal`
- `target` (1: presence of heart disease, 0: absence)

---

## 🧪 Project Workflow

### 1. 📥 Data Loading and Exploration
- Load dataset using `pandas`
- Check for missing values and understand feature distributions
- Visualize data using `seaborn.pairplot()`

### 2. 🧠 Feature Engineering
- Create a new binary feature `low_thalach` based on whether the `thalach` (maximum heart rate) is below the median

```python
heartData['low_thalach'] = heartData['thalach'].apply(lambda x: 1 if x < median_thalach else 0)
```

### 3. 🧹 Preprocessing
- Select relevant features for prediction
- Split the dataset into training (80%) and testing (20%) sets
- Standardize features using `StandardScaler`

### 4. 🤖 Modeling
- Use `LogisticRegression` to model the binary target variable
- Fit the model on the training data
- Predict target values on test data

### 5. 📊 Evaluation Metrics
- **Accuracy**
- **Confusion Matrix** (visualized with `seaborn`)
- **Precision, Recall, F1-Score**
- **Classification Report**

---

## 📈 Results

Evaluation metrics output:

- **Accuracy:** ~[Varies with split]
- **Classification Report:** Displays precision, recall, f1-score per class
- **Confusion Matrix:** Helps visualize TP, FP, FN, TN

---

## 🛠️ Libraries Used

- `pandas`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `scipy`, `pylab`

---

## 💡 Insights

- Maximum heart rate (`thalach`) is a significant indicator.
- Logistic Regression performs well on normalized data.
- The model generalizes well and can be extended to include more clinical features.

---

## 🚀 Future Work

- Try other classifiers like Random Forest, SVM
- Hyperparameter tuning
- Feature importance analysis
- Use SHAP for model interpretability

---

## 📎 How to Run

1. Clone the repository
2. Make sure `heart.csv` is in the root directory
3. Run the Jupyter notebook: `Heart_Attack_Prediction.ipynb`

---

## 📌 Author

Built with ❤️ by [Your Name]

