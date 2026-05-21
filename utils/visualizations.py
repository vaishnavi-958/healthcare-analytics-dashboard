"""
Visualizations Utility
======================
All Plotly chart factory functions used across the Streamlit dashboard.
Each function accepts a DataFrame and returns a Plotly Figure object.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Shared color palette ─────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold
BLUE    = "#1f77b4"
GREEN   = "#2ca02c"
RED     = "#d62728"
ORANGE  = "#ff7f0e"

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Arial, sans-serif", size=13),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ────────────────────────────────────────────────────────────────────────────
#  OVERVIEW CHARTS
# ────────────────────────────────────────────────────────────────────────────

def patient_volume_over_time(df: pd.DataFrame) -> go.Figure:
    """Daily patient volume trend line."""
    daily = df.groupby("admission_date").size().reset_index(name="count")
    fig = px.line(
        daily, x="admission_date", y="count",
        title="Daily Patient Volume Over Time",
        labels={"admission_date": "Date", "count": "Patients"},
        color_discrete_sequence=[BLUE],
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def admission_flag_donut(df: pd.DataFrame) -> go.Figure:
    """Donut chart of admitted vs. not admitted."""
    counts = df["admission_flag"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    fig = px.pie(
        counts, names="Status", values="Count",
        title="Admission Status Breakdown",
        hole=0.55,
        color_discrete_sequence=[BLUE, ORANGE],
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def monthly_volume_bar(df: pd.DataFrame) -> go.Figure:
    """Monthly patient volumes, grouped by year."""
    monthly = (
        df.groupby(["year", "month", "month_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["year", "month"])
    )
    fig = px.bar(
        monthly, x="month_name", y="count", color="year",
        barmode="group",
        title="Monthly Patient Volume by Year",
        labels={"month_name": "Month", "count": "Patients", "year": "Year"},
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


# ────────────────────────────────────────────────────────────────────────────
#  DEMOGRAPHICS CHARTS
# ────────────────────────────────────────────────────────────────────────────

def gender_breakdown(df: pd.DataFrame) -> go.Figure:
    """Bar chart of patient gender distribution."""
    counts = df["gender"].value_counts().reset_index()
    counts.columns = ["Gender", "Count"]
    fig = px.bar(
        counts, x="Gender", y="Count",
        title="Gender Distribution",
        color="Gender",
        color_discrete_sequence=[BLUE, ORANGE, GREEN],
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def race_breakdown(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of racial/ethnic groups."""
    counts = df["race"].value_counts().reset_index()
    counts.columns = ["Race", "Count"]
    fig = px.bar(
        counts, x="Count", y="Race", orientation="h",
        title="Patient Race / Ethnicity",
        color="Race",
        color_discrete_sequence=PALETTE,
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS, showlegend=False)
    return fig


def age_distribution_histogram(df: pd.DataFrame) -> go.Figure:
    """Age distribution histogram with KDE overlay."""
    fig = px.histogram(
        df, x="age", nbins=30,
        title="Patient Age Distribution",
        labels={"age": "Age (years)"},
        color_discrete_sequence=[BLUE],
        marginal="box",
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def age_group_bar(df: pd.DataFrame) -> go.Figure:
    """Patient count by age group, coloured by admission status."""
    grp = df.groupby(["age_group", "admission_flag"]).size().reset_index(name="count")
    fig = px.bar(
        grp, x="age_group", y="count", color="admission_flag",
        barmode="group",
        title="Patient Volume by Age Group & Admission Status",
        labels={"age_group": "Age Group", "count": "Patients", "admission_flag": "Status"},
        color_discrete_sequence=[BLUE, ORANGE],
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


# ────────────────────────────────────────────────────────────────────────────
#  DEPARTMENT CHARTS
# ────────────────────────────────────────────────────────────────────────────

def department_referrals_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of referral counts per department."""
    dept = df.groupby("department_referral").size().reset_index(name="count")
    dept = dept.sort_values("count", ascending=True)
    fig = px.bar(
        dept, x="count", y="department_referral", orientation="h",
        title="Referrals by Department",
        labels={"department_referral": "Department", "count": "Referrals"},
        color="count",
        color_continuous_scale="Blues",
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS, coloraxis_showscale=False)
    return fig


def dept_admission_rate(df: pd.DataFrame) -> go.Figure:
    """Admission rate (%) per department referral."""
    rate = (
        df.groupby("department_referral")["is_admitted"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"is_admitted": "admission_rate"})
        .sort_values("admission_rate", ascending=True)
    )
    fig = px.bar(
        rate, x="admission_rate", y="department_referral", orientation="h",
        title="Admission Rate by Department (%)",
        labels={"department_referral": "Department", "admission_rate": "Admission Rate (%)"},
        color="admission_rate",
        color_continuous_scale="RdYlGn",
        text=rate["admission_rate"].round(1).astype(str) + "%",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS, coloraxis_showscale=False)
    return fig


def dept_avg_wait(df: pd.DataFrame) -> go.Figure:
    """Average wait time per department."""
    wait = (
        df.groupby("department_referral")["wait_time"]
        .mean()
        .reset_index()
        .rename(columns={"wait_time": "avg_wait"})
        .sort_values("avg_wait", ascending=True)
    )
    fig = px.bar(
        wait, x="avg_wait", y="department_referral", orientation="h",
        title="Average Wait Time by Department (minutes)",
        labels={"department_referral": "Department", "avg_wait": "Avg Wait (min)"},
        color="avg_wait",
        color_continuous_scale="Oranges",
        text=wait["avg_wait"].round(1),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS, coloraxis_showscale=False)
    return fig


# ────────────────────────────────────────────────────────────────────────────
#  WAIT TIME CHARTS
# ────────────────────────────────────────────────────────────────────────────

def wait_time_histogram(df: pd.DataFrame) -> go.Figure:
    """Wait time distribution histogram with admission overlay."""
    fig = px.histogram(
        df, x="wait_time", color="admission_flag",
        barmode="overlay",
        nbins=40,
        title="Wait Time Distribution by Admission Status",
        labels={"wait_time": "Wait Time (minutes)", "admission_flag": "Status"},
        color_discrete_sequence=[BLUE, ORANGE],
        opacity=0.75,
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def wait_time_by_day(df: pd.DataFrame) -> go.Figure:
    """Box plot of wait time by day of week."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig = px.box(
        df, x="day_of_week", y="wait_time",
        category_orders={"day_of_week": order},
        title="Wait Time Distribution by Day of Week",
        labels={"day_of_week": "Day", "wait_time": "Wait Time (min)"},
        color="day_of_week",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(**LAYOUT_DEFAULTS, showlegend=False)
    return fig


def wait_time_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: average wait time by day × hour."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = (
        df.groupby(["day_of_week", "hour"])["wait_time"]
        .mean()
        .reset_index()
        .pivot(index="day_of_week", columns="hour", values="wait_time")
        .reindex(order)
    )
    fig = px.imshow(
        heat,
        title="Avg Wait Time Heatmap (Day × Hour)",
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg Wait (min)"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


# ────────────────────────────────────────────────────────────────────────────
#  SATISFACTION CHARTS
# ────────────────────────────────────────────────────────────────────────────

def satisfaction_histogram(df: pd.DataFrame) -> go.Figure:
    """Score distribution for patients who provided a score."""
    scored = df[df["satisfaction_score"].notna()]
    fig = px.histogram(
        scored, x="satisfaction_score",
        nbins=11,
        title="Patient Satisfaction Score Distribution",
        labels={"satisfaction_score": "Score (0-10)"},
        color_discrete_sequence=[GREEN],
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def satisfaction_by_race(df: pd.DataFrame) -> go.Figure:
    """Average satisfaction score by race."""
    sat = (
        df.groupby("race")["satisfaction_score"]
        .mean()
        .reset_index()
        .dropna()
        .sort_values("satisfaction_score", ascending=True)
    )
    fig = px.bar(
        sat, x="satisfaction_score", y="race", orientation="h",
        title="Avg Satisfaction Score by Race / Ethnicity",
        labels={"satisfaction_score": "Avg Score", "race": "Race"},
        color="satisfaction_score",
        color_continuous_scale="Greens",
        text=sat["satisfaction_score"].round(2),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS, coloraxis_showscale=False)
    return fig


def satisfaction_vs_waittime(df: pd.DataFrame) -> go.Figure:
    """Scatter: satisfaction score vs wait time, coloured by admission."""
    scored = df[df["satisfaction_score"].notna()]
    # Add manual trend line via numpy polyfit (avoids statsmodels dependency)
    scored_clean = scored[["wait_time", "satisfaction_score"]].dropna()
    z = np.polyfit(scored_clean["wait_time"], scored_clean["satisfaction_score"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(scored_clean["wait_time"].min(), scored_clean["wait_time"].max(), 100)

    fig = px.scatter(
        scored, x="wait_time", y="satisfaction_score",
        color="admission_flag",
        title="Satisfaction Score vs. Wait Time",
        labels={"wait_time": "Wait Time (min)", "satisfaction_score": "Satisfaction Score", "admission_flag": "Status"},
        opacity=0.6,
        color_discrete_sequence=[BLUE, ORANGE],
    )
    fig.add_trace(go.Scatter(
        x=x_line, y=p(x_line),
        mode="lines", name="Trend",
        line=dict(color="gray", dash="dash", width=2),
    ))
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def satisfaction_by_age_group(df: pd.DataFrame) -> go.Figure:
    """Violin plot of satisfaction score by age group."""
    scored = df[df["satisfaction_score"].notna()]
    fig = px.violin(
        scored, x="age_group", y="satisfaction_score",
        color="age_group",
        box=True, points="outliers",
        title="Satisfaction Score by Age Group",
        labels={"age_group": "Age Group", "satisfaction_score": "Satisfaction Score"},
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(**LAYOUT_DEFAULTS, showlegend=False)
    return fig


# ────────────────────────────────────────────────────────────────────────────
#  TIME-PATTERN CHARTS
# ────────────────────────────────────────────────────────────────────────────

def hourly_volume(df: pd.DataFrame) -> go.Figure:
    """Patient arrivals by hour of day."""
    hourly = df.groupby("hour").size().reset_index(name="count")
    fig = px.bar(
        hourly, x="hour", y="count",
        title="Patient Arrivals by Hour of Day",
        labels={"hour": "Hour (0–23)", "count": "Patient Count"},
        color="count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(**LAYOUT_DEFAULTS, coloraxis_showscale=False)
    return fig


def dow_volume(df: pd.DataFrame) -> go.Figure:
    """Patient volume by day of week."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = df.groupby("day_of_week").size().reset_index(name="count")
    dow["day_of_week"] = pd.Categorical(dow["day_of_week"], categories=order, ordered=True)
    dow = dow.sort_values("day_of_week")
    fig = px.bar(
        dow, x="day_of_week", y="count",
        title="Patient Volume by Day of Week",
        labels={"day_of_week": "Day", "count": "Patients"},
        color="count",
        color_continuous_scale="Purples",
    )
    fig.update_layout(**LAYOUT_DEFAULTS, coloraxis_showscale=False)
    return fig
