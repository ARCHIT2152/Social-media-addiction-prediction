# 📊 Social Media Addiction Prediction using Machine Learning

## 📌 Project Overview
This project analyzes student social media usage patterns and predicts their **social media addiction level** using multiple machine learning classification models.  
It is a complete end-to-end ML pipeline including:

- Data cleaning  
- Preprocessing  
- Encoding  
- Scaling  
- Model training  
- Evaluation  
- Model comparison  

## 🧠 Problem Statement
Excessive social media usage can affect students' mental health, academics, and relationships.  
Using machine learning, we aim to predict a student's **Addicted_Score** based on behavioral, academic, and lifestyle features.

---

## 📂 Dataset Details
- Source: **Kaggle**
- Format: `.csv`
- Total Records: **705**
- Target Variable: **Addicted_Score**
- Key Features:
  - Age  
  - Gender  
  - Academic Level  
  - Country  
  - Avg Daily Usage Hours  
  - Sleep Hours  
  - Mental Health Score  
  - Relationship Status  
  - Most Used Platform  
  - Conflicts Over Social Media  

---

## 🛠️ Tools & Technologies Used

### **Programming Language**
- Python 3.11

### **Python Libraries**
- **Pandas** – Data manipulation & cleaning  
- **NumPy** – Mathematical operations  
- **Matplotlib** – Visualizations  
- **Seaborn** – Heatmaps & statistical charts  
- **Scikit-Learn**  
  - `train_test_split`  
  - `LabelEncoder`  
  - `StandardScaler`  
  - Logistic Regression  
  - Random Forest Classifier  
  - Evaluation metrics  
- **XGBoost** – Advanced boosting classifier  

### **Development Tool**
- Visual Studio Code (VS Code)

---

## 🔄 Project Workflow
Load the Data

Inspect & Clean Data

Encode Categorical Columns

Separate Features (X) and Target (y)

Fix Target Labels for XGBoost

Train-Test Split

Feature Scaling

Train 3 Different Models

Evaluate Each Model

Compare Results


---

## 🔍 Step-by-Step Implementation

### 1️⃣ Data Loading
- Loaded CSV file using `pandas.read_csv()`
- Previewed dataset using `df.head()`

### 2️⃣ Data Cleaning
- Filled missing numeric values → column mean  
- Filled missing categorical values → most frequent value  

### 3️⃣ Encoding
- Used `LabelEncoder` to convert text → numbers  
  Example:  
  - Male → 1  
  - Female → 0  

### 4️⃣ Feature & Target Selection
- `X` = all features  
- `y` = Addicted_Score  

### 5️⃣ Label Correction for XGBoost
- Adjusted target values to start from **0**  
- Prevents XGBoost training errors  

### 6️⃣ Train-Test Split
- 80% training  
- 20% testing  
- Ensures unbiased model evaluation  

### 7️⃣ Feature Scaling
- Standardized values using `StandardScaler`  
- Ensures balanced feature importance  

### 8️⃣ Model Training
- **Logistic Regression**  
- **Random Forest Classifier**  
- **XGBoost Classifier**

### 9️⃣ Model Evaluation
- Accuracy Score  
- F1 Score (Macro)  
- Classification Report  
- Confusion Matrix  

### 🔟 Model Comparison
A table comparing accuracy & F1 Score across all models was created.

---

## 🤖 Machine Learning Models Used

### **1. Logistic Regression**
- Fast baseline model  
- Good for initial evaluation  
- Works well for linearly separable data  

### **2. Random Forest Classifier**
- Ensemble of decision trees  
- Reduces overfitting  
- Strong performer on mixed data  

### **3. XGBoost Classifier**
- Boosting technique  
- Learns from previous model mistakes  
- Often provides best accuracy  

---

## 📈 Results Summary
- Logistic Regression gave baseline performance  
- Random Forest improved accuracy  
- XGBoost achieved the strongest performance  

A confusion matrix and classification report were used for detailed evaluation.

---

## 🧠 Key Learnings

From this project, I learned:

- ✔ How to clean and preprocess real-world datasets  
- ✔ Difference between **features (X)** and **target (y)**  
- ✔ Why we use **train-test split**  
- ✔ How `fit()` and `predict()` work  
- ✔ Why scaling helps machine learning models  
- ✔ Why this is a **classification problem**, not regression  
- ✔ How Random Forest & XGBoost work internally  
- ✔ Reading classification reports & confusion matrices  
- ✔ Debugging ML errors (like XGBoost label errors)  
- ✔ Building a full ML pipeline from scratch  

---

## 🚀 Future Improvements
- Convert addiction score into: Low / Medium / High  
- Hyperparameter tuning  
- Add Power BI or Tableau dashboard  
- Deploy with Streamlit  
- Add feature importance analysis  

---

## 👤 Author
**Archit Bankey**  
Machine Learning & Data Analytics Enthusiast  

---

## ⭐ If you find this project useful, consider giving it a ⭐ on GitHub!

