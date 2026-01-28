import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv("students_socialmedia.csv")
print("Dataset Loaded Successfully!")
#print(df.head())

print("Dataset Info -")
#print(df.info())

num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])



le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

#print("Completed")

target = "Addicted_Score"   

X = df.drop(target, axis=1)


y = df[target]
y = y - y.min()       

print("Unique y after fixing:", sorted(y.unique()))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
y_train = y_train - y_train.min()
y_test = y_test - y_test.min()

print("Unique train labels:", sorted(y_train.unique()))
print("Unique test labels:", sorted(y_test.unique()))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Scaling Completed!")

log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)
pred_lr = log_reg.predict(X_test)

acc_lr = accuracy_score(y_test, pred_lr)
f1_lr = f1_score(y_test, pred_lr, average="macro")

print("Logistic Regression Accuracy:", acc_lr)
print("Logistic Regression F1 Score:", f1_lr)

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

acc_rf = accuracy_score(y_test, pred_rf)
f1_rf = f1_score(y_test, pred_rf, average="macro")

print("Random Forest Accuracy:", acc_rf)
print("Random Forest F1 Score:", f1_rf)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss"
)

xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)

acc_xgb = accuracy_score(y_test, pred_xgb)
f1_xgb = f1_score(y_test, pred_xgb, average="macro")

print("XGBoost Accuracy:", acc_xgb)
print("XGBoost F1 Score:", f1_xgb)

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [acc_lr, acc_rf, acc_xgb],
    "F1 Score": [f1_lr, f1_rf, f1_xgb]
})

print("MODEL COMPARISON: ")
print(results)

cm = confusion_matrix(y_test, pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



print(" Classification Report (Random Forest) -")
print(classification_report(y_test, pred_rf))
