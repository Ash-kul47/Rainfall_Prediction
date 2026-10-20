"""Machine Learning Training Pipeline for Rainfall Pattern Probability Estimation.

Trains, evaluates, and exports baseline, Logistic Regression, and Random Forest models
using a chronological train-test split to respect time-series ordering and prevent
look-ahead data leakage.
"""

import os
import sys
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Add parent directory to sys.path if running standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import load_and_preprocess_data, DEFAULT_RAINY_THRESHOLD_MM


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """Calculate comprehensive classification and probabilistic evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    
    # ROC-AUC calculation (guarded if test set has only 1 class)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm
    }


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract model feature matrix X and target vector y."""
    feature_cols = ['district', 'state', 'month', 'day_of_year', 'daily_normal']
    # If daily_normal has any NaNs, fill with median for that month
    df_clean = df.copy()
    if 'daily_normal' in df_clean.columns:
        df_clean['daily_normal'] = df_clean['daily_normal'].fillna(df_clean['daily_normal'].median())
    
    X = df_clean[feature_cols]
    y = df_clean['rainy']
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer for categorical and numeric features."""
    cat_cols = ['district', 'state']
    num_cols = ['month', 'day_of_year', 'daily_normal']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cat_cols),
            ('num', StandardScaler(), num_cols)
        ]
    )
    return preprocessor


def train_and_save_models(
    data_path: str = "data/rainfall_districtwise_daily_imd.csv",
    model_output_path: str = "models/rainfall_model.pkl",
    threshold: float = DEFAULT_RAINY_THRESHOLD_MM,
    train_split_ratio: float = 0.8
) -> Dict[str, Any]:
    """Train baseline, Logistic Regression, Random Forest, and Decision Tree pipelines and save artifact."""
    print("=" * 60)
    print("Rainfall Pattern ML Pipeline: Training & Evaluation")
    print("=" * 60)

    # 1. Load and Preprocess Data
    print(f"Loading dataset from: {data_path}")
    df = load_and_preprocess_data(data_path, threshold=threshold)
    print(f"Total records: {len(df)}, Threshold for rainy day: {threshold} mm")
    print(f"Dataset date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    # 2. Chronological Train-Test Split (respecting temporal ordering)
    unique_dates = np.sort(df['date'].unique())
    n_dates = len(unique_dates)
    split_idx = int(n_dates * train_split_ratio)
    split_date = unique_dates[split_idx]

    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()

    train_start = train_df['date'].min().strftime('%d-%b-%Y')
    train_end = train_df['date'].max().strftime('%d-%b-%Y')
    test_start = test_df['date'].min().strftime('%d-%b-%Y')
    test_end = test_df['date'].max().strftime('%d-%b-%Y')

    print(f"\nChronological Split:")
    print(f"  Training Set:   {len(train_df):,} rows ({train_start} to {train_end}) | Rainy: {train_df['rainy'].mean()*100:.1f}%")
    print(f"  Test Set:       {len(test_df):,} rows ({test_start} to {test_end}) | Rainy: {test_df['rainy'].mean()*100:.1f}%")

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    # 3. Baseline Model: Historical Frequency Baseline
    # Predicts the training set historical prior probability
    p_baseline = float(y_train.mean())
    y_pred_base = np.full_like(y_test, int(p_baseline >= 0.5))
    y_prob_base = np.full(len(y_test), p_baseline, dtype=float)
    base_metrics = evaluate_predictions(y_test.values, y_pred_base, y_prob_base)
    print("\n--- Model 1: Historical Frequency Baseline ---")
    for k, v in base_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:15s}: {v}")

    # 4. Model 2: Logistic Regression Pipeline
    print("\n--- Training Model 2: Logistic Regression ---")
    preprocessor_lr = build_preprocessor()
    lr_pipeline = Pipeline([
        ('prep', preprocessor_lr),
        ('clf', LogisticRegression(max_iter=500, random_state=42, solver='lbfgs'))
    ])
    lr_pipeline.fit(X_train, y_train)

    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)
    lr_metrics = evaluate_predictions(y_test.values, lr_preds, lr_probs)
    for k, v in lr_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:15s}: {v}")

    # 5. Model 3: Random Forest Classifier Pipeline
    print("\n--- Training Model 3: Random Forest Classifier ---")
    preprocessor_rf = build_preprocessor()
    rf_pipeline = Pipeline([
        ('prep', preprocessor_rf),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)

    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)
    rf_metrics = evaluate_predictions(y_test.values, rf_preds, rf_probs)
    for k, v in rf_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:15s}: {v}")

    # 6. Model 4: Decision Tree Classifier Pipeline
    print("\n--- Training Model 4: Decision Tree Classifier ---")
    preprocessor_dt = build_preprocessor()
    dt_pipeline = Pipeline([
        ('prep', preprocessor_dt),
        ('clf', DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42
        ))
    ])
    dt_pipeline.fit(X_train, y_train)

    dt_probs = dt_pipeline.predict_proba(X_test)[:, 1]
    dt_preds = (dt_probs >= 0.5).astype(int)
    dt_metrics = evaluate_predictions(y_test.values, dt_preds, dt_probs)
    for k, v in dt_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:15s}: {v}")

    # 7. Save Bundle to Joblib
    os.makedirs(os.path.dirname(os.path.abspath(model_output_path)), exist_ok=True)
    bundle = {
        "lr_pipeline": lr_pipeline,
        "rf_pipeline": rf_pipeline,
        "dt_pipeline": dt_pipeline,
        "historical_prior": p_baseline,
        "threshold": threshold,
        "feature_cols": ['district', 'state', 'month', 'day_of_year', 'daily_normal'],
        "split_info": {
            "train_start": train_start,
            "train_end": train_end,
            "train_size": len(train_df),
            "test_start": test_start,
            "test_end": test_end,
            "test_size": len(test_df),
            "train_rainy_pct": round(float(y_train.mean()) * 100, 2),
            "test_rainy_pct": round(float(y_test.mean()) * 100, 2)
        },
        "metrics": {
            "Baseline": base_metrics,
            "Logistic Regression": lr_metrics,
            "Random Forest": rf_metrics,
            "Decision Tree": dt_metrics
        }
    }

    joblib.dump(bundle, model_output_path)
    print(f"\nTrained models successfully serialized and saved to: {model_output_path}")
    print("=" * 60)
    return bundle


if __name__ == "__main__":
    train_and_save_models()
