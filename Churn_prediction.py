"""
Credit Card Churn Prediction - Direct Mail Targeting
=====================================================
Flags top-decile at-risk credit card accounts for retention targeting.

Pipeline:
1. Load and clean the data
2. Select features using chi-square testing
3. Handle class imbalance with SMOTE
4. Train and compare Random Forest and Logistic Regression
5. Evaluate on a held-out test set
6. Export scored predictions for the Power BI dashboard
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

DATA_PATH = "data/BankChurners.csv"
OUTPUT_PATH = "outputs/churn_predictions.csv"
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Load the raw dataset and drop known non-predictive columns."""
    df = pd.read_csv(path)
    drop_cols = [c for c in df.columns if "Naive_Bayes_Classifier" in c]
    drop_cols += ["CLIENTNUM"] if "CLIENTNUM" in df.columns else []
    df = df.drop(columns=drop_cols, errors="ignore")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the churn label to a binary target: 1 = churned, 0 = active."""
    df = df.copy()
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda x: 1 if str(x).strip().lower() == "attrited customer" else 0
    )
    df = df.drop(columns=["Attrition_Flag"])
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns so chi-square testing and models can use them."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def chi_square_feature_ranking(df: pd.DataFrame, target_col: str, top_n: int = 10) -> list:
    """Rank features by chi-square association with the target."""
    features = [c for c in df.columns if c != target_col]
    scores = {}
    for col in features:
        try:
            values = df[col].values.reshape(-1, 1)
            if df[col].nunique() > 10:
                binned = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
                values = binned.fit_transform(values)
            contingency = pd.crosstab(values.flatten(), df[target_col])
            chi2, p, _, _ = chi2_contingency(contingency)
            scores[col] = chi2
        except ValueError:
            continue
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_features = [name for name, _ in ranked[:top_n]]
    return top_features, ranked


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train Random Forest and Logistic Regression, tune via grid search, and return the better model."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    results = {}
    rf_params = {"n_estimators": [200, 400], "max_depth": [8, 12, None]}
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        rf_params,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
    )
    rf_grid.fit(X_train_res, y_train_res)
    rf_best = rf_grid.best_estimator_
    results["Random Forest"] = evaluate_model(rf_best, X_test_scaled, y_test)
    lr_params = {"C": [0.01, 0.1, 1, 10]}
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        lr_params,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
    )
    lr_grid.fit(X_train_res, y_train_res)
    lr_best = lr_grid.best_estimator_
    results["Logistic Regression"] = evaluate_model(lr_best, X_test_scaled, y_test)
    for name, res in results.items():
        print(f"\n{name}")
        print(f"  ROC-AUC:   {res['roc_auc']:.3f}")
        print(f"  Precision: {res['precision']:.3f}")
        print(f"  Recall:    {res['recall']:.3f}")
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = rf_best if best_name == "Random Forest" else lr_best
    print(f"\nSelected model: {best_name}")
    return best_model, scaler, results[best_name]


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Download the dataset from "
            "https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers "
            "and place BankChurners.csv in the data/ folder."
        )
    print("Loading data...")
    df = load_data(DATA_PATH)
    df = encode_target(df)
    df = encode_categoricals(df)
    print("\nRunning chi-square feature ranking...")
    top_features, ranked = chi_square_feature_ranking(df, target_col="Churn", top_n=10)
    print("Top features by chi-square score:")
    for name, score in ranked[:10]:
        print(f"  {name}: {score:.1f}")
    X = df[top_features]
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTraining set: {len(X_train)} rows | Test set: {len(X_test)} rows")
    print(f"Churn rate in full dataset: {y.mean():.1%}")
    best_model, scaler, best_result = train_and_evaluate(X_train, X_test, y_train, y_test)
    os.makedirs("outputs", exist_ok=True)
    scored = X_test.copy()
    scored["actual_churn"] = y_test.values
    scored["predicted_churn"] = best_result["y_pred"]
    scored["churn_probability"] = best_result["y_proba"]
    scored.to_csv(OUTPUT_PATH, index=False)
    print(f"\nScored predictions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
