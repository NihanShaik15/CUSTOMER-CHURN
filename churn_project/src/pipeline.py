"""
Customer Churn Prediction - End-to-End Pipeline
=================================================
Author: (Your Name)
Dataset: IBM Telco Customer Churn (7,043 customers, 21 features)

This script performs:
1. Data loading & cleaning
2. Exploratory Data Analysis (EDA) with saved visualizations
3. Feature engineering & preprocessing
4. Model training: Logistic Regression, Random Forest, XGBoost
5. Evaluation: Accuracy, Recall, Precision, F1, ROC-AUC, Confusion Matrix
6. Feature importance / explainability (SHAP)
7. Saves the best model + a metrics report
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import xgboost as xgb
import shap
import joblib

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)

    # TotalCharges has blank strings for new customers (tenure=0) -> convert & fill
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    df.drop(columns=["customerID"], inplace=True)

    # Target to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def run_eda(df):
    """Generates and saves the key EDA plots used in the report."""

    # 1. Churn distribution
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x="Churn", data=df, palette=["#4C72B0", "#DD8452"])
    ax.set_xticklabels(["Stayed (0)", "Churned (1)"])
    churn_rate = df["Churn"].mean() * 100
    plt.title(f"Churn Distribution (Overall churn rate: {churn_rate:.1f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_churn_distribution.png"))
    plt.close()

    # 2. Churn by Contract type
    plt.figure(figsize=(6, 4))
    contract_churn = df.groupby("Contract")["Churn"].mean().sort_values() * 100
    contract_churn.plot(kind="barh", color="#C44E52")
    plt.xlabel("Churn Rate (%)")
    plt.title("Churn Rate by Contract Type")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_churn_by_contract.png"))
    plt.close()

    # 3. Tenure vs Churn
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, multiple="stack",
                 palette=["#4C72B0", "#DD8452"])
    plt.title("Tenure Distribution by Churn")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_tenure_vs_churn.png"))
    plt.close()

    # 4. Monthly Charges vs Churn
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette=["#4C72B0", "#DD8452"])
    plt.xticks([0, 1], ["Stayed", "Churned"])
    plt.title("Monthly Charges by Churn Status")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_monthlycharges_vs_churn.png"))
    plt.close()

    # 5. Internet Service vs churn
    plt.figure(figsize=(6, 4))
    internet_churn = df.groupby("InternetService")["Churn"].mean().sort_values() * 100
    internet_churn.plot(kind="barh", color="#55A868")
    plt.xlabel("Churn Rate (%)")
    plt.title("Churn Rate by Internet Service Type")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_churn_by_internet_service.png"))
    plt.close()

    eda_summary = {
        "overall_churn_rate_pct": round(churn_rate, 2),
        "churn_rate_by_contract": contract_churn.round(2).to_dict(),
        "churn_rate_by_internet_service": internet_churn.round(2).to_dict(),
        "avg_tenure_churned": round(df[df.Churn == 1]["tenure"].mean(), 1),
        "avg_tenure_retained": round(df[df.Churn == 0]["tenure"].mean(), 1),
        "avg_monthly_charges_churned": round(df[df.Churn == 1]["MonthlyCharges"].mean(), 2),
        "avg_monthly_charges_retained": round(df[df.Churn == 0]["MonthlyCharges"].mean(), 2),
    }
    return eda_summary


def preprocess(df):
    df_enc = df.copy()
    cat_cols = df_enc.select_dtypes(include="object").columns.tolist()

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le

    X = df_enc.drop(columns=["Churn"])
    y = df_enc["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)


def evaluate_model(name, model, X_test, y_test, results, use_scaled=False):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    results[name] = metrics

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(OUT_DIR, f"cm_{fname}.png"))
    plt.close()

    return y_prob, metrics


def train_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    results = {}
    roc_data = {}

    # 1. Logistic Regression (uses scaled features)
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    prob, _ = evaluate_model("Logistic Regression", log_reg, X_test_scaled, y_test, results)
    roc_data["Logistic Regression"] = prob

    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    prob, _ = evaluate_model("Random Forest", rf, X_test, y_test, results)
    roc_data["Random Forest"] = prob

    # 3. XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42
    )
    xgb_model.fit(X_train, y_train)
    prob, _ = evaluate_model("XGBoost", xgb_model, X_test, y_test, results)
    roc_data["XGBoost"] = prob

    # --- Combined ROC curve ---
    plt.figure(figsize=(6, 5))
    for name, prob in roc_data.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = results[name]["roc_auc"]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_roc_comparison.png"))
    plt.close()

    # --- Feature importance (Random Forest) ---
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
    plt.figure(figsize=(6, 5))
    importances.sort_values().plot(kind="barh", color="#4C72B0")
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_feature_importance.png"))
    plt.close()

    # --- SHAP explainability for XGBoost (best model, typically) ---
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, max_display=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "08_shap_summary.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("SHAP plot skipped:", e)

    # Save best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = {"Logistic Regression": log_reg, "Random Forest": rf, "XGBoost": xgb_model}[best_name]
    joblib.dump(best_model, os.path.join(OUT_DIR, "best_model.pkl"))

    return results, best_name, importances.round(4).to_dict()


def main():
    print("Loading & cleaning data...")
    df = load_and_clean_data()

    print("Running EDA...")
    eda_summary = run_eda(df)

    print("Preprocessing...")
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess(df)

    print("Training models: Logistic Regression, Random Forest, XGBoost...")
    results, best_name, top_features = train_models(
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )

    report = {
        "dataset_shape": list(df.shape),
        "eda_summary": eda_summary,
        "model_results": results,
        "best_model": best_name,
        "top_10_features_rf": top_features,
    }

    with open(os.path.join(OUT_DIR, "metrics_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== RESULTS SUMMARY ===")
    for name, m in results.items():
        print(f"{name:22s} | Acc: {m['accuracy']:.3f} | Recall: {m['recall']:.3f} | "
              f"Precision: {m['precision']:.3f} | F1: {m['f1_score']:.3f} | ROC-AUC: {m['roc_auc']:.3f}")
    print(f"\nBest model (by ROC-AUC): {best_name}")
    print(f"All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
