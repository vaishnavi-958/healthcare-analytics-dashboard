"""
Core Data Analysis Module
=========================
Statistical analysis functions for the healthcare patient flow dataset.
Run this module directly to print a full analysis report to stdout.

Usage
-----
    python analysis/data_analysis.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import sqlite3
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_data  # noqa: E402


# ── Helper ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ── Analysis Functions ───────────────────────────────────────────────────────

def dataset_overview(df: pd.DataFrame) -> dict:
    """
    Return basic dataset overview statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned healthcare DataFrame.

    Returns
    -------
    dict
        Summary statistics.
    """
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def admission_analysis(df: pd.DataFrame) -> dict:
    """Analyse admission rates across demographic groups."""
    return {
        "overall_rate": df["is_admitted"].mean() * 100,
        "by_gender": df.groupby("gender")["is_admitted"].mean().mul(100).round(2).to_dict(),
        "by_race": df.groupby("race")["is_admitted"].mean().mul(100).round(2).to_dict(),
        "by_age_group": df.groupby("age_group", observed=True)["is_admitted"].mean().mul(100).round(2).to_dict(),
        "by_department": df.groupby("department_referral")["is_admitted"].mean().mul(100).round(2).to_dict(),
    }


def wait_time_analysis(df: pd.DataFrame) -> dict:
    """Descriptive and inferential statistics on wait times."""
    wt = df["wait_time"].dropna()

    # T-test: admitted vs not admitted
    admitted_wt = df[df["is_admitted"]]["wait_time"].dropna()
    not_admitted_wt = df[~df["is_admitted"]]["wait_time"].dropna()
    t_stat, p_val = stats.ttest_ind(admitted_wt, not_admitted_wt)

    return {
        "mean": wt.mean(),
        "median": wt.median(),
        "std": wt.std(),
        "min": wt.min(),
        "max": wt.max(),
        "p25": wt.quantile(0.25),
        "p75": wt.quantile(0.75),
        "by_day": df.groupby("day_of_week")["wait_time"].mean().round(2).to_dict(),
        "by_time_of_day": df.groupby("time_of_day")["wait_time"].mean().round(2).to_dict(),
        "by_department": df.groupby("department_referral")["wait_time"].mean().round(2).to_dict(),
        "ttest_admission_vs_not": {"t_stat": round(t_stat, 4), "p_value": round(p_val, 4)},
    }


def satisfaction_analysis(df: pd.DataFrame) -> dict:
    """Analyse satisfaction scores; pearson correlation with wait time."""
    scored = df[df["satisfaction_score"].notna()]

    corr, p_val = stats.pearsonr(
        scored["satisfaction_score"], scored["wait_time"]
    )

    return {
        "n_scored": len(scored),
        "pct_scored": len(scored) / len(df) * 100,
        "mean": scored["satisfaction_score"].mean(),
        "std": scored["satisfaction_score"].std(),
        "by_gender": scored.groupby("gender")["satisfaction_score"].mean().round(2).to_dict(),
        "by_race": scored.groupby("race")["satisfaction_score"].mean().round(2).to_dict(),
        "by_age_group": scored.groupby("age_group", observed=True)["satisfaction_score"].mean().round(2).to_dict(),
        "by_admission": scored.groupby("admission_flag")["satisfaction_score"].mean().round(2).to_dict(),
        "pearson_corr_with_wait": round(corr, 4),
        "pearson_p_value": round(p_val, 4),
    }


def demographic_breakdown(df: pd.DataFrame) -> dict:
    """Full demographic profile of the patient population."""
    return {
        "gender_dist": df["gender"].value_counts().to_dict(),
        "race_dist": df["race"].value_counts().to_dict(),
        "age_stats": df["age"].describe().round(2).to_dict(),
        "age_group_dist": df["age_group"].value_counts().sort_index().to_dict(),
        "dept_referral_dist": df["department_referral"].value_counts().to_dict(),
        "time_of_day_dist": df["time_of_day"].value_counts().to_dict(),
    }


def temporal_patterns(df: pd.DataFrame) -> dict:
    """Volume and wait-time patterns by time dimensions."""
    return {
        "by_year": df.groupby("year").size().to_dict(),
        "by_month": df.groupby("month_name").size().to_dict(),
        "by_dow": df.groupby("day_of_week").size().to_dict(),
        "by_hour": df.groupby("hour").size().to_dict(),
        "avg_wait_by_hour": df.groupby("hour")["wait_time"].mean().round(2).to_dict(),
    }


# ── SQL-in-Python Analysis ───────────────────────────────────────────────────

def run_sql_analysis(df: pd.DataFrame) -> dict:
    """
    Execute representative SQL queries against an in-memory SQLite database.

    Returns
    -------
    dict
        Query name → result DataFrame.
    """
    conn = sqlite3.connect(":memory:")
    df.to_sql("patients", conn, index=False, if_exists="replace")

    queries = {
        "top_departments_by_volume": """
            SELECT department_referral AS department,
                   COUNT(*) AS patient_count,
                   ROUND(AVG(wait_time), 2) AS avg_wait_min,
                   ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction
            FROM patients
            GROUP BY department_referral
            ORDER BY patient_count DESC;
        """,
        "admission_rate_by_race": """
            SELECT race,
                   COUNT(*) AS total,
                   SUM(CASE WHEN is_admitted = 1 THEN 1 ELSE 0 END) AS admitted,
                   ROUND(
                       100.0 * SUM(CASE WHEN is_admitted = 1 THEN 1 ELSE 0 END) / COUNT(*), 2
                   ) AS admission_rate_pct
            FROM patients
            GROUP BY race
            ORDER BY admission_rate_pct DESC;
        """,
        "hourly_peak_analysis": """
            SELECT hour,
                   COUNT(*) AS patient_count,
                   ROUND(AVG(wait_time), 2) AS avg_wait_min
            FROM patients
            WHERE hour IS NOT NULL
            GROUP BY hour
            ORDER BY hour;
        """,
        "high_wait_admitted_patients": """
            SELECT patient_id, age, gender, race,
                   department_referral, wait_time, satisfaction_score
            FROM patients
            WHERE wait_time > 50 AND is_admitted = 1
            ORDER BY wait_time DESC
            LIMIT 20;
        """,
        "monthly_volume_trend": """
            SELECT year,
                   month,
                   month_name,
                   COUNT(*) AS patient_count,
                   ROUND(AVG(wait_time), 2) AS avg_wait
            FROM patients
            GROUP BY year, month, month_name
            ORDER BY year, month;
        """,
        "satisfaction_low_scorers": """
            SELECT race, gender, age_group,
                   COUNT(*) AS count,
                   ROUND(AVG(satisfaction_score), 2) AS avg_score
            FROM patients
            WHERE satisfaction_score IS NOT NULL
              AND satisfaction_score <= 3
            GROUP BY race, gender, age_group
            ORDER BY count DESC
            LIMIT 15;
        """,
    }

    results = {}
    for name, sql in queries.items():
        results[name] = pd.read_sql_query(sql, conn)

    conn.close()
    return results


# ── CLI Report ───────────────────────────────────────────────────────────────

def print_full_report():
    """Print a comprehensive analysis report to stdout."""
    # Note: when running standalone, load raw CSV directly
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "healthcare_data.csv")
    df.columns = [
        "patient_id", "admission_date", "admission_time", "patient_name",
        "gender", "age", "race", "department_referral",
        "admission_flag", "satisfaction_score", "wait_time"
    ]
    df["admission_date"] = pd.to_datetime(df["admission_date"], format="%m/%d/%Y", errors="coerce")
    df["is_admitted"] = df["admission_flag"] == "Admission"
    df["hour"] = pd.to_datetime(
        df["admission_time"], format="%I:%M:%S %p", errors="coerce"
    ).dt.hour

    def time_bucket(h):
        if pd.isna(h): return "Unknown"
        if 6 <= h < 12: return "Morning (6–12)"
        if 12 <= h < 18: return "Afternoon (12–18)"
        if 18 <= h < 24: return "Evening (18–24)"
        return "Night (0–6)"

    df["time_of_day"] = df["hour"].apply(time_bucket)
    bins = [0, 12, 17, 35, 50, 65, 120]
    labels = ["Child (0-12)", "Teen (13-17)", "Young Adult (18-35)",
              "Adult (36-50)", "Middle Age (51-65)", "Senior (65+)"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    df["year"] = df["admission_date"].dt.year
    df["month"] = df["admission_date"].dt.month
    df["month_name"] = df["admission_date"].dt.strftime("%B")
    df["day_of_week"] = df["admission_date"].dt.day_name()

    _section("DATASET OVERVIEW")
    ov = dataset_overview(df)
    print(f"  Rows: {ov['shape'][0]:,}  |  Columns: {ov['shape'][1]}")
    print("\n  Missing values:")
    for col, n in ov["missing_values"].items():
        if n > 0:
            print(f"    {col:<30} {n:>6} ({ov['missing_pct'][col]:.1f}%)")

    _section("ADMISSION ANALYSIS")
    adm = admission_analysis(df)
    print(f"  Overall Admission Rate: {adm['overall_rate']:.2f}%\n")
    print("  By Gender:")
    for k, v in adm["by_gender"].items():
        print(f"    {k:<25} {v:.2f}%")
    print("\n  By Department:")
    for k, v in sorted(adm["by_department"].items(), key=lambda x: -x[1]):
        print(f"    {k:<30} {v:.2f}%")

    _section("WAIT TIME ANALYSIS")
    wt = wait_time_analysis(df)
    print(f"  Mean:   {wt['mean']:.2f} min")
    print(f"  Median: {wt['median']:.2f} min")
    print(f"  Std:    {wt['std']:.2f} min")
    print(f"  Range:  {wt['min']} – {wt['max']} min")
    t = wt["ttest_admission_vs_not"]
    sig = "SIGNIFICANT" if t["p_value"] < 0.05 else "not significant"
    print(f"\n  T-test (Admitted vs Not): t={t['t_stat']}, p={t['p_value']} [{sig}]")

    _section("SATISFACTION ANALYSIS")
    sat = satisfaction_analysis(df)
    print(f"  Patients with scores: {sat['n_scored']:,} ({sat['pct_scored']:.1f}%)")
    print(f"  Mean score:           {sat['mean']:.2f}")
    corr = sat["pearson_corr_with_wait"]
    sig2 = "SIGNIFICANT" if sat["pearson_p_value"] < 0.05 else "not significant"
    print(f"  Pearson corr with wait time: {corr:.4f} (p={sat['pearson_p_value']}, {sig2})")

    _section("SQL QUERY RESULTS")
    sql_results = run_sql_analysis(df)
    for name, result_df in sql_results.items():
        print(f"\n  Query: {name}")
        print(result_df.to_string(index=False))

    print("\n\n  Analysis complete.")


if __name__ == "__main__":
    print_full_report()
