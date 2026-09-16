"""Customer churn ML, written as a simple top-to-bottom script.

No user-defined functions or classes are needed.
Run this file with the Python environment used for this project.
It creates a beginner_outputs folder containing charts and evaluation results.
This is an educational experiment: the dates of feature collection and churn
are unknown, and this dataset has already been explored in previous runs.
"""

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib

# Save charts to files, so the script also works without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
)
from xgboost import XGBClassifier


# ============================================================
# STEP 2: LOAD YOUR DATASET
# ============================================================

PROJECT_FOLDER = Path(__file__).resolve().parent

# Your original dataset was moved to this backup folder.
# Change DATA_PATH if you want to use a different Excel file.
DATA_PATH = (
    PROJECT_FOLDER.parent
    / "surya_backup_20260916_3files"
    / "Customer_Churn_Dataset.xlsx"
)

# Normally results go into beginner_outputs beside this script.
OUTPUT_FOLDER = Path(os.environ.get(
    "CHURN_OUTPUT_DIR", str(PROJECT_FOLDER / "beginner_outputs")
))
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA_PATH)

print("First five customers:")
print(df.head())
print("\nNumber of rows and columns:", df.shape)
print("\nColumn data types:")
print(df.dtypes)


# ============================================================
# STEP 3: EDA - UNDERSTAND THE DATA
# ============================================================

print("\nMissing values in each column:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate customer IDs:", df["Customer_ID"].duplicated().sum())
print("\nBasic statistics:")
print(df.describe().round(2))

# Churn_Flag: 0 = retained customer, 1 = churned customer.
print("\nChurn counts:")
print(df["Churn_Flag"].value_counts())
print("Churn percentage:", round(df["Churn_Flag"].mean() * 100, 2))

# Stop if customer IDs or labels need investigation.
# Removing conflicting customers silently could distort the analysis.
if df["Customer_ID"].isna().any() or df["Customer_ID"].duplicated().any():
    raise ValueError("Customer IDs must be present and unique.")
if not df["Churn_Flag"].isin([0, 1]).all():
    raise ValueError("Churn_Flag must contain only 0 and 1.")

# Dates in this dataset need review. Do not replace them with guessed values.
# Signup_Date is not a churn date, so it is not used for a time-based split.


# ============================================================
# STEP 4: SEPARATE INPUTS (X) AND TARGET (y)
# ============================================================

y = df["Churn_Flag"]

# Customer_ID is just an identifier.
# Signup_Date has uncertain timing; Total_Charges is an accumulated amount.
# Churn_Flag must NEVER be included in the input features.
X = df.drop(columns=[
    "Customer_ID", "Signup_Date", "Total_Charges", "Churn_Flag"
])


# ============================================================
# STEP 5: SPLIT DATA - 70% TRAIN, 15% VALIDATION, 15% TEST
# ============================================================

# random_state=42 makes the split reproducible.
# stratify=y keeps approximately the same churn proportion in each group.
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Half of the remaining 30% is 15% of the complete dataset.
X_validation, X_test, y_validation, y_test = train_test_split(
    X_remaining, y_remaining,
    test_size=0.50, random_state=42, stratify=y_remaining
)

print("\nTraining customers:", len(X_train))       # 2251
print("Validation customers:", len(X_validation)) # 482
print("Test customers:", len(X_test))             # 483

# Verify that the same customer is not in two groups.
assert set(X_train.index).isdisjoint(X_validation.index)
assert set(X_train.index).isdisjoint(X_test.index)
assert set(X_validation.index).isdisjoint(X_test.index)
assert len(X_train) + len(X_validation) + len(X_test) == len(df)

# Detailed EDA below uses only training customers.
training_data = df.loc[X_train.index]
print("\nTraining churn rate by contract:")
print(training_data.groupby("Contract_Type")["Churn_Flag"].mean())
print("\nTraining churn rate by complaints:")
print(training_data.groupby("Complaint_Count")["Churn_Flag"].mean())

training_data["Churn_Flag"].value_counts().sort_index().plot(
    kind="bar", color=["steelblue", "coral"], rot=0
)
plt.title("Training customers: retained (0) and churned (1)")
plt.xlabel("Churn flag")
plt.ylabel("Customer count")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "churn_distribution.png", dpi=150)
plt.close()


# ============================================================
# STEP 6: FEATURE ENGINEERING - CREATE USEFUL EXTRA COLUMNS
# ============================================================

X_train = X_train.copy()
X_validation = X_validation.copy()
X_test = X_test.copy()

# Apply identical arithmetic to all three groups.
# These calculations do not learn averages or patterns from the test set.
for data in [X_train, X_validation, X_test]:
    tenure = data["Tenure_Months"].replace(0, np.nan)

    data["Has_Complaint"] = (data["Complaint_Count"] > 0).astype(float)
    data.loc[data["Complaint_Count"].isna(), "Has_Complaint"] = np.nan

    data["Has_Late_Payment"] = (data["Late_Payment_Count"] > 0).astype(float)
    data.loc[data["Late_Payment_Count"].isna(), "Has_Late_Payment"] = np.nan

    # These ratios assume counts cover the customer's entire tenure.
    data["Complaints_Per_Tenure_Month"] = data["Complaint_Count"] / tenure
    data["Late_Payments_Per_Tenure_Month"] = data["Late_Payment_Count"] / tenure
    data["Tickets_Per_Tenure_Month"] = data["Support_Tickets_Raised"] / tenure

    # Assumes Monthly_Charges is BEFORE discount. This is a model feature,
    # not a verified measure of revenue.
    data["Estimated_Charge_After_Discount"] = (
        data["Monthly_Charges"] * (1 - data["Discount_Applied_Pct"] / 100)
    )
    data["Inactivity_x_NonLoyalty"] = (
        data["Days_Since_Last_Activity"] * (1 - data["Loyalty_Program_Member"])
    )
    data["Complaints_x_LowSatisfaction"] = (
        data["Complaint_Count"] * (10 - data["Satisfaction_Score"])
    )

    data.replace([np.inf, -np.inf], np.nan, inplace=True)


# ============================================================
# STEP 7: PREPROCESSING - MISSING VALUES, SCALING, ENCODING
# ============================================================

numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
categorical_columns = X_train.select_dtypes(exclude="number").columns.tolist()

# Median = middle value. It fills missing numeric values.
# StandardScaler puts numeric features on comparable scales.
numeric_steps = Pipeline([
    ("fill_missing", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("scale", StandardScaler()),
])

# Most frequent = most common category.
# OneHotEncoder converts categories such as Monthly/Annual into numeric columns.
categorical_steps = Pipeline([
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("numbers", numeric_steps, numeric_columns),
    ("categories", categorical_steps, categorical_columns),
])

# IMPORTANT: fit only on training data.
# Validation and test use the training medians, scales and category mapping.
X_train_ready = preprocessor.fit_transform(X_train)
X_validation_ready = preprocessor.transform(X_validation)
X_test_ready = preprocessor.transform(X_test)

print("\nPrepared training data shape:", X_train_ready.shape)


# ============================================================
# STEP 8: TRAIN THE THREE MODELS
# ============================================================

# These fixed settings come from the earlier project tuning.
# This simple script does not repeat the slow hyperparameter search.
models = {
    "Logistic Regression": LogisticRegression(
        C=0.1, max_iter=3000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=3,
        max_features=0.7, random_state=42, n_jobs=2
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.025, max_depth=4,
        min_child_weight=10, subsample=1.0, colsample_bytree=0.8,
        reg_lambda=5, scale_pos_weight=1, eval_metric="logloss",
        tree_method="hist", random_state=42, n_jobs=2
    ),
}


# ============================================================
# STEP 9: USE VALIDATION DATA TO SELECT MODEL AND THRESHOLD
# ============================================================

thresholds = {}
validation_results = []

for name, model in models.items():
    print("\nTraining:", name, flush=True)
    model.fit(X_train_ready, y_train)

    validation_scores = model.predict_proba(X_validation_ready)[:, 1]
    best_accuracy = -1
    best_f1 = -1
    best_threshold = 0.5

    # A score at or above the threshold becomes class 1 (churn).
    # Try possible thresholds using VALIDATION data, never test data.
    possible_thresholds = np.unique(np.append(validation_scores, [0, 0.5, 1.000001]))

    for threshold in possible_thresholds:
        validation_prediction = (validation_scores >= threshold).astype(int)
        accuracy = accuracy_score(y_validation, validation_prediction)
        f1 = f1_score(y_validation, validation_prediction, zero_division=0)

        # Prioritize accuracy. If tied, prefer F1, then proximity to 0.5.
        candidate = (accuracy, f1, -abs(threshold - 0.5))
        current_best = (best_accuracy, best_f1, -abs(best_threshold - 0.5))
        if candidate > current_best:
            best_accuracy = accuracy
            best_f1 = f1
            best_threshold = float(threshold)

    thresholds[name] = best_threshold
    validation_results.append({
        "Model": name,
        "Accuracy": best_accuracy,
        "F1": best_f1,
        "Average precision": average_precision_score(y_validation, validation_scores),
        "Threshold": best_threshold,
    })

validation_table = pd.DataFrame(validation_results).sort_values(
    ["Accuracy", "F1", "Average precision"], ascending=False
)
best_model_name = validation_table.iloc[0]["Model"]

print("\nValidation results:")
print(validation_table.round(4).to_string(index=False))
print("\nSelected model:", best_model_name)
validation_table.to_csv(OUTPUT_FOLDER / "validation_results.csv", index=False)

# Model selection is finished. Do not switch models after seeing test results.


# ============================================================
# STEP 10: FINAL TEST EVALUATION
# ============================================================

# A dummy model always predicts the more frequent class.
# It shows why accuracy alone is not enough for imbalanced data.
baseline = DummyClassifier(strategy="prior")
baseline.fit(X_train_ready, y_train)
models["Dummy baseline"] = baseline
thresholds["Dummy baseline"] = 0.5

test_results = []
figure, axes = plt.subplots(1, 2, figsize=(12, 5))

for name, model in models.items():
    scores = model.predict_proba(X_test_ready)[:, 1]
    predictions = (scores >= thresholds[name]).astype(int)

    precision_curve, recall_curve, unused = precision_recall_curve(y_test, scores)
    pr_auc = auc(recall_curve, precision_curve)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    test_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, scores),
        "PR-AUC": pr_auc,
        "Average precision": average_precision_score(y_test, scores),
        "Threshold": thresholds[name],
        "TN": int(matrix[0, 0]), "FP": int(matrix[0, 1]),
        "FN": int(matrix[1, 0]), "TP": int(matrix[1, 1]),
    })

    print("\n", name, "confusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)

    chart, chart_axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=["Retained", "Churned"]).plot(
        ax=chart_axis, colorbar=False, cmap="Blues"
    )
    chart_axis.set_title(name)
    chart.tight_layout()
    chart.savefig(OUTPUT_FOLDER / (name.replace(" ", "_") + "_confusion.png"), dpi=150)
    plt.close(chart)

    false_positive_rate, true_positive_rate, unused = roc_curve(y_test, scores)
    axes[0].plot(false_positive_rate, true_positive_rate, label=name)
    axes[1].plot(recall_curve, precision_curve, label=name)

axes[0].set(title="Test ROC curves", xlabel="False positive rate", ylabel="True positive rate")
axes[1].set(title="Test precision-recall curves", xlabel="Recall", ylabel="Precision")
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[1].axhline(y_test.mean(), linestyle="--", color="gray")
for axis in axes:
    axis.legend(fontsize=8)
figure.tight_layout()
figure.savefig(OUTPUT_FOLDER / "evaluation_curves.png", dpi=150)
plt.close(figure)

test_table = pd.DataFrame(test_results)
print("\nFinal test results:")
print(test_table.round(4).to_string(index=False))
test_table.to_csv(OUTPUT_FOLDER / "test_results.csv", index=False)

print("\nModel chosen using validation:", best_model_name)
print("Results saved to:", OUTPUT_FOLDER)
print("Accuracy can hide missed churners. Check precision, recall and F1 too.")
print("PR-AUC is trapezoidal area; average precision is a different summary.")
print("This script does not replace the model.pkl used by Flask.")
