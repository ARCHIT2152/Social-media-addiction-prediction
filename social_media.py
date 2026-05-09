import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

df = pd.read_csv(r"C:\Users\archit\Desktop\PROJECTS\social_mediakaggle\students_socialmedia.csv")
print(df.shape)


df = df.fillna(df.mode().iloc[0])
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df["MH_Level"] = df["Mental_Health_Score"].map({4:0, 5:0, 6:1, 7:1, 8:2, 9:2})

print("\nClass distribution:")
print(df["MH_Level"].value_counts().sort_index()
        .rename({0:"Low (4-5)", 1:"Moderate (6-7)", 2:"High (8-9)"}))



drop_cols = ["Mental_Health_Score", "MH_Level", "Student_ID",
             "Addicted_Score", "Affects_Academic_Performance",
             "Conflicts_Over_Social_Media"]
X = df.drop(columns=drop_cols, errors="ignore")
y = df["MH_Level"]

print(f"\nFeatures ({len(X.columns)}): {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")


models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=200, C=0.05, solver="lbfgs"))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=80, max_depth=5, min_samples_leaf=5, random_state=42
    ),
    "XGBoost": GradientBoostingClassifier(
        n_estimators=60, learning_rate=0.05, max_depth=3,
        min_samples_leaf=10, subsample=0.7, random_state=42
    ),
}

cv_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results  = []
preds    = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    preds[name] = pred

    acc = accuracy_score(y_test, pred)
    f1  = f1_score(y_test, pred, average="macro")
    cv  = cross_val_score(model, X, y, cv=cv_split, scoring="accuracy")

    results.append({
        "Model": name,
        "Test Accuracy": acc,
        "Macro F1": f1,
        "CV Accuracy": cv.mean(),
        "CV Std": cv.std()
    })
    print(f"\n{name}:")
    print(f"  Test Acc = {acc:.3f} | Macro F1 = {f1:.3f}")
    print(f"  CV Acc   = {cv.mean():.3f} ± {cv.std():.3f}")

results_df = pd.DataFrame(results)
print("\n── MODEL COMPARISON ──")
print(results_df[["Model","Test Accuracy","Macro F1","CV Accuracy"]].to_string(index=False))

best_name = results_df.loc[results_df["CV Accuracy"].idxmax(), "Model"]
print(f"\nBest model (by CV Accuracy): {best_name}")
print(classification_report(y_test, preds[best_name],
                             target_names=["Low","Moderate","High"]))


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Student Mental Health Level Classifier\n"
             "Predicted from Social Media & Demographic Features",
             fontsize=12, fontweight="bold")

ax = axes[0]
x  = np.arange(len(results_df))
w  = 0.35
dark  = ["#2C5F8A", "#1F6B45", "#8B2020"]
light = ["#6AAED6", "#74C476", "#E08080"]

b1 = ax.bar(x - w/2, results_df["CV Accuracy"],   w, label="CV Accuracy",
            color=dark, alpha=0.9, zorder=3)
b2 = ax.bar(x + w/2, results_df["Test Accuracy"], w, label="Test Accuracy",
            color=light, alpha=0.9, zorder=3)
ax.errorbar(x - w/2, results_df["CV Accuracy"],
            yerr=results_df["CV Std"], fmt="none", color="black", capsize=4, zorder=4)
ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=8.5, fontweight="bold", color="white",
             label_type="center")
ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=8.5, fontweight="bold", color="white",
             label_type="center")
ax.set_xticks(x)
ax.set_xticklabels(results_df["Model"], rotation=12, ha="right", fontsize=9)
ax.set_ylim(0.55, 1.0)
ax.set_title("CV vs Test Accuracy by Model", fontsize=10)
ax.set_ylabel("Accuracy")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3, zorder=0)

ax = axes[1]
cm = confusion_matrix(y_test, preds[best_name])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Low","Moderate","High"],
            yticklabels=["Low","Moderate","High"])
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=10)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")


ax = axes[2]
rf_model    = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top         = importances.sort_values()
colors_imp  = ["#C44E52" if i == top.index[-1] else "#4C72B0" for i in top.index]
top.plot(kind="barh", ax=ax, color=colors_imp, alpha=0.85)
ax.set_title("Feature Importances (Random Forest)\nRed = most important", fontsize=10)
ax.set_xlabel("Importance")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("social_media_results.png", dpi=150, bbox_inches="tight")
plt.show()
