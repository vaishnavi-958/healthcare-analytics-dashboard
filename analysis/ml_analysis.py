"""
Machine Learning Analysis Module
=================================
Applies supervised and unsupervised ML to derive predictive insights
from the healthcare patient flow dataset.

Models
------
1. Admission Predictor — Logistic Regression (binary classification)
2. Satisfaction Score Estimator — Random Forest Regressor
3. Patient Segmentation — K-Means Clustering on demographic + clinical features
4. Wait Time Anomaly Detection — IQR-based flagging

Usage
-----
    python analysis/ml_analysis.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, r2_score,
)
import warnings
warnings.filterwarnings("ignore")


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical columns and build a clean numeric feature matrix.

    Returns
    -------
    (X, encoders) where X is the numeric DataFrame and encoders is a dict
    of fitted LabelEncoders for reference.
    """
    cat_cols = ["gender", "race", "department_referral", "time_of_day", "day_of_week"]
    num_cols = ["age", "hour", "wait_time"]

    X = df[cat_cols + num_cols].copy()
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    return X, encoders


# ── Model 1: Admission Predictor ─────────────────────────────────────────────

def train_admission_predictor(df: pd.DataFrame) -> dict:
    """
    Train a Random Forest classifier to predict patient admission.

    Returns
    -------
    dict with model, metrics, and feature importances.
    """
    X, encoders = build_feature_matrix(df)
    y = df["is_admitted"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1", n_jobs=-1)

    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return {
        "model": model,
        "encoders": encoders,
        "accuracy": model.score(X_test, y_test),
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "feature_importances": importances,
        "feature_names": list(X.columns),
    }


# ── Model 2: Satisfaction Estimator ──────────────────────────────────────────

def train_satisfaction_model(df: pd.DataFrame) -> dict:
    """
    Train a Random Forest regressor to estimate satisfaction score.

    Only uses records that have a satisfaction score.

    Returns
    -------
    dict with model, MAE, R², and feature importances.
    """
    scored = df[df["satisfaction_score"].notna()].copy()

    X, encoders = build_feature_matrix(scored)
    y = scored["satisfaction_score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return {
        "model": model,
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "feature_importances": importances,
        "n_training_samples": len(scored),
    }


# ── Model 3: Patient Segmentation (K-Means) ───────────────────────────────────

def patient_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    """
    Segment patients into groups using K-Means clustering.

    Features used: age, wait_time, satisfaction_score (filled), hour,
    is_admitted, has_referral.

    Returns
    -------
    dict with cluster labels added to a copy of df, and cluster profiles.
    """
    features = ["age", "wait_time", "satisfaction_score_filled",
                "hour", "is_admitted", "has_referral"]

    X = df[features].copy()
    X = X.fillna(X.median())
    X["is_admitted"] = X["is_admitted"].astype(int)
    X["has_referral"] = X["has_referral"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["cluster"] = labels

    profile = (
        df_clustered.groupby("cluster")[
            ["age", "wait_time", "satisfaction_score_filled", "is_admitted", "has_referral"]
        ]
        .mean()
        .round(2)
    )
    profile["size"] = df_clustered.groupby("cluster").size()

    return {
        "df_clustered": df_clustered,
        "cluster_profiles": profile,
        "inertia": kmeans.inertia_,
        "n_clusters": n_clusters,
    }


# ── Model 4: Wait Time Anomaly Detection ─────────────────────────────────────

def detect_wait_time_anomalies(df: pd.DataFrame) -> dict:
    """
    Flag wait-time anomalies using the IQR method.

    Returns
    -------
    dict with anomaly DataFrame and thresholds.
    """
    q1 = df["wait_time"].quantile(0.25)
    q3 = df["wait_time"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    anomalies = df[(df["wait_time"] < lower) | (df["wait_time"] > upper)].copy()
    anomalies["anomaly_type"] = np.where(
        anomalies["wait_time"] > upper, "High Wait", "Unusually Low Wait"
    )

    return {
        "anomalies": anomalies,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_anomalies": len(anomalies),
        "pct_anomalies": len(anomalies) / len(df) * 100,
    }


# ── CLI Report ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_data  # noqa: E402

    # Re-use the streamlit cached loader is not available from CLI;
    # load directly instead.
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "healthcare_data.csv")
    df.columns = [
        "patient_id", "admission_date", "admission_time", "patient_name",
        "gender", "age", "race", "department_referral",
        "admission_flag", "satisfaction_score", "wait_time"
    ]
    df["is_admitted"] = df["admission_flag"] == "Admission"
    df["has_referral"] = df["department_referral"] != "None"
    df["hour"] = pd.to_datetime(df["admission_time"], format="%I:%M:%S %p", errors="coerce").dt.hour

    def time_bucket(h):
        if pd.isna(h): return "Unknown"
        if 6 <= h < 12: return "Morning"
        if 12 <= h < 18: return "Afternoon"
        if 18 <= h < 24: return "Evening"
        return "Night"

    df["time_of_day"] = df["hour"].apply(time_bucket)
    df["day_of_week"] = pd.to_datetime(df["admission_date"], format="%m/%d/%Y", errors="coerce").dt.day_name()
    df["satisfaction_score_filled"] = df["satisfaction_score"].fillna(df["satisfaction_score"].median())

    print("=" * 60)
    print("  ML ANALYSIS REPORT — Healthcare Patient Flow")
    print("=" * 60)

    print("\n[1] Admission Predictor (Random Forest)")
    adm = train_admission_predictor(df)
    print(f"  Accuracy:       {adm['accuracy']:.4f}")
    print(f"  CV F1 (5-fold): {adm['cv_f1_mean']:.4f} ± {adm['cv_f1_std']:.4f}")
    print(f"  Top features:   {list(adm['feature_importances'].index[:3])}")

    print("\n[2] Satisfaction Estimator (Random Forest Regressor)")
    sat = train_satisfaction_model(df)
    print(f"  MAE:  {sat['mae']:.4f}")
    print(f"  R²:   {sat['r2']:.4f}")
    print(f"  Trained on {sat['n_training_samples']:,} scored records")

    print("\n[3] Patient Segmentation (K-Means, k=4)")
    seg = patient_segmentation(df)
    print(seg["cluster_profiles"].to_string())

    print("\n[4] Wait-Time Anomaly Detection (IQR)")
    anom = detect_wait_time_anomalies(df)
    print(f"  Bounds:    [{anom['lower_bound']:.1f}, {anom['upper_bound']:.1f}] min")
    print(f"  Anomalies: {anom['n_anomalies']} ({anom['pct_anomalies']:.2f}%)")
