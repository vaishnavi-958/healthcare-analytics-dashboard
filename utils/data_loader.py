"""
Data Loader Utility
===================
Handles loading, cleaning, and preprocessing of the healthcare dataset.
Provides a cached loader for use throughout the Streamlit application.
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent / "data" / "healthcare_data.csv"


@st.cache_data(show_spinner="Loading healthcare dataset...")
def load_data() -> pd.DataFrame:
    """
    Load and preprocess the healthcare patient flow dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-engineered DataFrame ready for analysis.
    """
    df = pd.read_csv(DATA_PATH)

    # ── Rename columns for readability ──────────────────────────────────────
    df.columns = [
        "patient_id", "admission_date", "admission_time", "patient_name",
        "gender", "age", "race", "department_referral",
        "admission_flag", "satisfaction_score", "wait_time"
    ]

    # ── Parse date/time fields ───────────────────────────────────────────────
    df["admission_date"] = pd.to_datetime(df["admission_date"], format="%m/%d/%Y", errors="coerce")
    df["admission_time"] = pd.to_datetime(df["admission_time"], format="%I:%M:%S %p", errors="coerce").dt.time

    # ── Derived temporal features ────────────────────────────────────────────
    df["year"] = df["admission_date"].dt.year
    df["month"] = df["admission_date"].dt.month
    df["month_name"] = df["admission_date"].dt.strftime("%B")
    df["day_of_week"] = df["admission_date"].dt.day_name()
    df["week_number"] = df["admission_date"].dt.isocalendar().week.fillna(0).astype(int)
    df["hour"] = pd.to_datetime(
        df["admission_time"].astype(str), format="%H:%M:%S", errors="coerce"
    ).dt.hour

    # ── Time-of-day bucket ───────────────────────────────────────────────────
    def time_bucket(hour):
        if pd.isna(hour):
            return "Unknown"
        if 6 <= hour < 12:
            return "Morning (6–12)"
        if 12 <= hour < 18:
            return "Afternoon (12–18)"
        if 18 <= hour < 24:
            return "Evening (18–24)"
        return "Night (0–6)"

    df["time_of_day"] = df["hour"].apply(time_bucket)

    # ── Age groups ───────────────────────────────────────────────────────────
    bins = [0, 12, 17, 35, 50, 65, 120]
    labels = ["Child (0-12)", "Teen (13-17)", "Young Adult (18-35)",
              "Adult (36-50)", "Middle Age (51-65)", "Senior (65+)"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

    # ── Wait-time category ───────────────────────────────────────────────────
    df["wait_category"] = pd.cut(
        df["wait_time"],
        bins=[0, 20, 40, 60, 999],
        labels=["Short (<20 min)", "Moderate (20-40 min)", "Long (40-60 min)", "Very Long (60+ min)"]
    )

    # ── Boolean admission flag ───────────────────────────────────────────────
    df["is_admitted"] = df["admission_flag"] == "Admission"

    # ── Satisfaction: fill missing with median (MAR assumption) ──────────────
    median_sat = df["satisfaction_score"].median()
    df["satisfaction_score_filled"] = df["satisfaction_score"].fillna(median_sat)

    # ── Referral flag ────────────────────────────────────────────────────────
    df["has_referral"] = df["department_referral"] != "None"

    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute top-level KPI summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned healthcare DataFrame.

    Returns
    -------
    dict
        Dictionary of KPI name → value.
    """
    return {
        "total_patients": len(df),
        "admission_rate": df["is_admitted"].mean() * 100,
        "avg_wait_time": df["wait_time"].mean(),
        "avg_satisfaction": df["satisfaction_score"].mean(),
        "unique_departments": df["department_referral"].nunique(),
        "date_range_start": df["admission_date"].min().strftime("%b %d, %Y"),
        "date_range_end": df["admission_date"].max().strftime("%b %d, %Y"),
        "pct_with_score": df["satisfaction_score"].notna().mean() * 100,
        "pct_male": (df["gender"] == "Male").mean() * 100,
        "pct_female": (df["gender"] == "Female").mean() * 100,
    }
