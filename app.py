"""
AI Healthcare Analytics Dashboard
===================================
Interactive Streamlit dashboard for exploring patient flow, demographics,
wait-time trends, satisfaction metrics, and ML-driven insights.

Run
---
    streamlit run app.py --server.port 5000
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sqlite3

# ── Local modules ─────────────────────────────────────────────────────────────
from utils.data_loader import load_data, get_summary_stats
from utils import visualizations as viz
from analysis.data_analysis import (
    admission_analysis, wait_time_analysis,
    satisfaction_analysis, run_sql_analysis, temporal_patterns,
)
from analysis.ml_analysis import (
    train_admission_predictor, train_satisfaction_model,
    patient_segmentation, detect_wait_time_anomalies,
)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Load data ─────────────────────────────────────────────────────────────────
df_full = load_data()


# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 Healthcare Analytics")
    st.caption("Patient Flow Intelligence Dashboard")
    st.divider()

    # Year filter
    years = sorted(df_full["year"].dropna().unique().tolist())
    selected_years = st.multiselect("Year", years, default=years)

    # Gender filter
    genders = sorted(df_full["gender"].dropna().unique().tolist())
    selected_genders = st.multiselect("Gender", genders, default=genders)

    # Admission status
    adm_options = ["All", "Admitted Only", "Not Admitted Only"]
    adm_filter = st.radio("Admission Status", adm_options, index=0)

    # Department filter
    depts = sorted(df_full["department_referral"].dropna().unique().tolist())
    selected_depts = st.multiselect("Department Referral", depts, default=depts)

    st.divider()
    st.caption("Data: Patient Flow Dataset · 9,217 records")


# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_full[
    df_full["year"].isin(selected_years) &
    df_full["gender"].isin(selected_genders) &
    df_full["department_referral"].isin(selected_depts)
].copy()

if adm_filter == "Admitted Only":
    df = df[df["is_admitted"]]
elif adm_filter == "Not Admitted Only":
    df = df[~df["is_admitted"]]

if df.empty:
    st.warning("No data matches the current filters. Please adjust the sidebar selections.")
    st.stop()


# ── Navigation tabs ───────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "👥 Demographics",
    "🏢 Departments",
    "⏱️ Wait Times",
    "⭐ Satisfaction",
    "🤖 AI Insights",
    "🗃️ SQL Explorer",
    "📋 Raw Data",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Patient Flow Overview")

    stats = get_summary_stats(df)

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Patients",     f"{stats['total_patients']:,}")
    c2.metric("Admission Rate",     f"{stats['admission_rate']:.1f}%")
    c3.metric("Avg Wait Time",      f"{stats['avg_wait_time']:.1f} min")
    c4.metric("Avg Satisfaction",   f"{stats['avg_satisfaction']:.2f} / 10" if not np.isnan(stats['avg_satisfaction']) else "N/A")
    c5.metric("% with Score",       f"{stats['pct_with_score']:.1f}%")
    c6.metric("Departments",        stats['unique_departments'])

    st.caption(f"Date range: {stats['date_range_start']} → {stats['date_range_end']}")
    st.divider()

    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.plotly_chart(viz.patient_volume_over_time(df), use_container_width=True)
    with col_r:
        st.plotly_chart(viz.admission_flag_donut(df), use_container_width=True)

    st.plotly_chart(viz.monthly_volume_bar(df), use_container_width=True)

    # YoY summary table
    st.subheader("Year-over-Year Summary")
    yoy = (
        df.groupby("year").agg(
            Patients=("patient_id", "count"),
            Admission_Rate=("is_admitted", lambda x: f"{x.mean()*100:.1f}%"),
            Avg_Wait=("wait_time", lambda x: f"{x.mean():.1f} min"),
            Avg_Satisfaction=("satisfaction_score", lambda x: f"{x.mean():.2f}" if x.notna().sum() > 0 else "N/A"),
        )
        .reset_index()
        .rename(columns={"year": "Year"})
    )
    st.dataframe(yoy, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Patient Demographics")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.gender_breakdown(df), use_container_width=True)
        st.plotly_chart(viz.age_distribution_histogram(df), use_container_width=True)
    with col2:
        st.plotly_chart(viz.race_breakdown(df), use_container_width=True)
        st.plotly_chart(viz.age_group_bar(df), use_container_width=True)

    # Admission rates summary table
    st.subheader("Admission Rate by Demographic Group")
    adm = admission_analysis(df)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.write("**By Gender**")
        st.dataframe(
            pd.DataFrame(adm["by_gender"].items(), columns=["Gender", "Admission %"]),
            hide_index=True, use_container_width=True
        )
    with col_b:
        st.write("**By Age Group**")
        st.dataframe(
            pd.DataFrame(adm["by_age_group"].items(), columns=["Age Group", "Admission %"]),
            hide_index=True, use_container_width=True
        )
    with col_c:
        st.write("**By Race**")
        st.dataframe(
            pd.DataFrame(adm["by_race"].items(), columns=["Race", "Admission %"]),
            hide_index=True, use_container_width=True
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — DEPARTMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Department Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.department_referrals_bar(df), use_container_width=True)
        st.plotly_chart(viz.dept_avg_wait(df), use_container_width=True)
    with col2:
        st.plotly_chart(viz.dept_admission_rate(df), use_container_width=True)

        # Department summary table
        dept_tbl = (
            df.groupby("department_referral").agg(
                Patients=("patient_id", "count"),
                Avg_Wait=("wait_time", "mean"),
                Admission_Rate=("is_admitted", "mean"),
                Avg_Score=("satisfaction_score", "mean"),
            )
            .round(2)
            .reset_index()
            .rename(columns={"department_referral": "Department"})
            .sort_values("Patients", ascending=False)
        )
        dept_tbl["Admission_Rate"] = (dept_tbl["Admission_Rate"] * 100).round(1).astype(str) + "%"
        st.dataframe(dept_tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — WAIT TIMES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Wait Time Analysis")

    wt = wait_time_analysis(df)
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Mean Wait",   f"{wt['mean']:.1f} min")
    w2.metric("Median Wait", f"{wt['median']:.1f} min")
    w3.metric("Std Dev",     f"{wt['std']:.1f} min")
    w4.metric("Range",       f"{int(wt['min'])}–{int(wt['max'])} min")

    t = wt["ttest_admission_vs_not"]
    sig_label = "✅ Statistically significant" if t["p_value"] < 0.05 else "❌ Not statistically significant"
    st.info(f"**T-test (Admitted vs Not Admitted):** t = {t['t_stat']}, p = {t['p_value']}  —  {sig_label}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.wait_time_histogram(df), use_container_width=True)
        st.plotly_chart(viz.hourly_volume(df), use_container_width=True)
    with col2:
        st.plotly_chart(viz.wait_time_by_day(df), use_container_width=True)
        st.plotly_chart(viz.dow_volume(df), use_container_width=True)

    st.plotly_chart(viz.wait_time_heatmap(df), use_container_width=True)

    # Anomaly detection
    st.subheader("Wait-Time Anomalies (IQR Method)")
    anom = detect_wait_time_anomalies(df)
    st.warning(
        f"Detected **{anom['n_anomalies']} anomalous records** "
        f"({anom['pct_anomalies']:.2f}% of filtered dataset) — "
        f"bounds: [{anom['lower_bound']:.1f}, {anom['upper_bound']:.1f}] min"
    )
    if not anom["anomalies"].empty:
        st.dataframe(
            anom["anomalies"][[
                "patient_id", "gender", "age", "race",
                "department_referral", "wait_time", "admission_flag",
            ]].head(30),
            use_container_width=True, hide_index=True
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — SATISFACTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Patient Satisfaction Analysis")

    sat = satisfaction_analysis(df)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Patients Scored", f"{sat['n_scored']:,}")
    s2.metric("Response Rate",   f"{sat['pct_scored']:.1f}%")
    s3.metric("Mean Score",      f"{sat['mean']:.2f}")
    s4.metric("Std Dev",         f"{sat['std']:.2f}")

    corr = sat["pearson_corr_with_wait"]
    corr_label = "✅ Significant" if sat["pearson_p_value"] < 0.05 else "❌ Not significant"
    st.info(
        f"**Pearson correlation (Score vs Wait Time):** r = {corr:.4f}, "
        f"p = {sat['pearson_p_value']}  —  {corr_label}"
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.satisfaction_histogram(df), use_container_width=True)
        st.plotly_chart(viz.satisfaction_by_age_group(df), use_container_width=True)
    with col2:
        st.plotly_chart(viz.satisfaction_by_race(df), use_container_width=True)
        st.plotly_chart(viz.satisfaction_vs_waittime(df), use_container_width=True)

    # Score breakdown table
    st.subheader("Avg Satisfaction by Admission Status")
    st.dataframe(
        pd.DataFrame(sat["by_admission"].items(), columns=["Status", "Avg Score"]),
        hide_index=True, use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — AI INSIGHTS (ML)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("AI & Machine Learning Insights")
    st.caption("Models are trained on the current filtered dataset each session.")

    run_ml = st.button("▶ Run ML Analysis", type="primary")

    if run_ml or st.session_state.get("ml_ran"):
        st.session_state["ml_ran"] = True

        with st.spinner("Training models…"):
            adm_model = train_admission_predictor(df)
            sat_model  = train_satisfaction_model(df)
            seg_result = patient_segmentation(df)
            anom_result = detect_wait_time_anomalies(df)

        st.divider()

        # ── Model 1: Admission Predictor ──────────────────────────────────
        st.subheader("1. Admission Predictor — Random Forest Classifier")
        m1, m2, m3 = st.columns(3)
        m1.metric("Test Accuracy",  f"{adm_model['accuracy']:.4f}")
        m2.metric("CV F1 (5-fold)", f"{adm_model['cv_f1_mean']:.4f}")
        m3.metric("CV F1 Std",      f"± {adm_model['cv_f1_std']:.4f}")

        fi = adm_model["feature_importances"].reset_index()
        fi.columns = ["Feature", "Importance"]
        fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                        title="Feature Importances — Admission Predictor",
                        color="Importance", color_continuous_scale="Blues")
        fig_fi.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

        report = adm_model["classification_report"]
        cr_df = pd.DataFrame({
            "Class": ["Not Admitted", "Admitted"],
            "Precision": [report["0"]["precision"], report["1"]["precision"]],
            "Recall":    [report["0"]["recall"],    report["1"]["recall"]],
            "F1-Score":  [report["0"]["f1-score"],  report["1"]["f1-score"]],
        }).round(4)
        st.dataframe(cr_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Model 2: Satisfaction Estimator ───────────────────────────────
        st.subheader("2. Satisfaction Score Estimator — Random Forest Regressor")
        t1, t2, t3 = st.columns(3)
        t1.metric("Mean Absolute Error", f"{sat_model['mae']:.4f}")
        t2.metric("R² Score",            f"{sat_model['r2']:.4f}")
        t3.metric("Training Samples",    f"{sat_model['n_training_samples']:,}")

        fi2 = sat_model["feature_importances"].reset_index()
        fi2.columns = ["Feature", "Importance"]
        fig_fi2 = px.bar(fi2, x="Importance", y="Feature", orientation="h",
                         title="Feature Importances — Satisfaction Estimator",
                         color="Importance", color_continuous_scale="Greens")
        fig_fi2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_fi2, use_container_width=True)

        st.divider()

        # ── Model 3: Patient Segmentation ─────────────────────────────────
        st.subheader("3. Patient Segmentation — K-Means (k=4)")
        profile = seg_result["cluster_profiles"].reset_index()
        profile.columns = [
            "Cluster", "Avg Age", "Avg Wait (min)",
            "Avg Satisfaction", "Admitted Rate", "Has Referral Rate", "Size"
        ]
        st.dataframe(profile.round(2), use_container_width=True, hide_index=True)

        df_seg = seg_result["df_clustered"]
        fig_seg = px.scatter(
            df_seg.sample(min(2000, len(df_seg)), random_state=42),
            x="wait_time", y="satisfaction_score_filled",
            color=df_seg["cluster"].astype(str),
            title="Patient Segments — Wait Time vs Satisfaction",
            labels={"wait_time": "Wait Time (min)",
                    "satisfaction_score_filled": "Satisfaction Score",
                    "color": "Cluster"},
            opacity=0.6,
        )
        st.plotly_chart(fig_seg, use_container_width=True)

        st.divider()

        # ── Model 4: Anomaly Detection ────────────────────────────────────
        st.subheader("4. Wait-Time Anomaly Detection (IQR Method)")
        a1, a2, a3 = st.columns(3)
        a1.metric("Lower Bound", f"{anom_result['lower_bound']:.1f} min")
        a2.metric("Upper Bound", f"{anom_result['upper_bound']:.1f} min")
        a3.metric("Anomalies Found", f"{anom_result['n_anomalies']} ({anom_result['pct_anomalies']:.2f}%)")

        if not anom_result["anomalies"].empty:
            fig_anom = px.scatter(
                df, x="admission_date", y="wait_time",
                color=df["patient_id"].isin(anom_result["anomalies"]["patient_id"]).map(
                    {True: "Anomaly", False: "Normal"}
                ),
                title="Wait-Time Anomalies Over Time",
                labels={"wait_time": "Wait Time (min)", "admission_date": "Date", "color": "Status"},
                color_discrete_map={"Normal": "#1f77b4", "Anomaly": "#d62728"},
                opacity=0.5,
            )
            st.plotly_chart(fig_anom, use_container_width=True)
    else:
        st.info("Click **▶ Run ML Analysis** to train models on the currently filtered dataset.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — SQL EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("SQL Query Explorer")
    st.caption(
        "Queries run against an in-memory SQLite database "
        "populated with the currently filtered dataset."
    )

    # Pre-built queries dropdown
    preset_queries = {
        "Patient Volume Summary": """
SELECT COUNT(*) AS total_patients,
       ROUND(100.0 * SUM(CASE WHEN is_admitted = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS admission_rate_pct,
       ROUND(AVG(wait_time), 2) AS avg_wait_min,
       ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction
FROM patients;""",
        "Department Performance": """
SELECT department_referral AS department,
       COUNT(*) AS patient_count,
       ROUND(AVG(wait_time), 2) AS avg_wait_min,
       ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction,
       ROUND(100.0 * SUM(CASE WHEN is_admitted = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS admission_rate_pct
FROM patients
GROUP BY department_referral
ORDER BY patient_count DESC;""",
        "Admission Rate by Race": """
SELECT race,
       COUNT(*) AS total,
       SUM(CASE WHEN is_admitted = 1 THEN 1 ELSE 0 END) AS admitted,
       ROUND(100.0 * SUM(CASE WHEN is_admitted = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS admission_rate_pct
FROM patients
GROUP BY race
ORDER BY admission_rate_pct DESC;""",
        "Hourly Peak Analysis": """
SELECT hour, COUNT(*) AS patient_count,
       ROUND(AVG(wait_time), 2) AS avg_wait_min
FROM patients
WHERE hour IS NOT NULL
GROUP BY hour
ORDER BY hour;""",
        "Monthly Volume Trend": """
SELECT year, month, month_name,
       COUNT(*) AS patient_count,
       ROUND(AVG(wait_time), 2) AS avg_wait
FROM patients
GROUP BY year, month, month_name
ORDER BY year, month;""",
        "High Wait + Admitted": """
SELECT patient_id, age, gender, race,
       department_referral, wait_time, satisfaction_score
FROM patients
WHERE wait_time > 50 AND is_admitted = 1
ORDER BY wait_time DESC
LIMIT 20;""",
        "Custom Query": "",
    }

    selected_preset = st.selectbox("Pre-built Query", list(preset_queries.keys()))
    default_sql = preset_queries[selected_preset]

    sql_input = st.text_area("SQL Query", value=default_sql, height=180)

    if st.button("▶ Run Query", type="primary") and sql_input.strip():
        try:
            conn = sqlite3.connect(":memory:")
            df.to_sql("patients", conn, index=False, if_exists="replace")
            result_df = pd.read_sql_query(sql_input, conn)
            conn.close()
            st.success(f"Returned {len(result_df):,} rows")
            st.dataframe(result_df, use_container_width=True)

            # Quick visualization if applicable
            if len(result_df) > 1 and result_df.select_dtypes(include="number").shape[1] >= 1:
                num_col = result_df.select_dtypes(include="number").columns[0]
                cat_col = result_df.select_dtypes(exclude="number").columns[0] if result_df.select_dtypes(exclude="number").shape[1] > 0 else None
                if cat_col:
                    fig_q = px.bar(result_df.head(20), x=cat_col, y=num_col,
                                   title=f"Quick Chart: {num_col} by {cat_col}")
                    st.plotly_chart(fig_q, use_container_width=True)
        except Exception as e:
            st.error(f"Query error: {e}")

    st.divider()
    st.caption("Table: `patients` — all columns from the dataset plus derived fields (year, month, age_group, is_admitted, etc.)")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 8 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("Raw Dataset")

    st.write(f"**{len(df):,} records** match the current filters.")

    search = st.text_input("Search by Patient ID or Name")
    display_df = df.copy()
    if search:
        mask = (
            display_df["patient_id"].str.contains(search, case=False, na=False) |
            display_df["patient_name"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    st.dataframe(
        display_df[[
            "patient_id", "admission_date", "gender", "age", "race",
            "department_referral", "admission_flag", "wait_time",
            "satisfaction_score", "time_of_day", "age_group",
        ]].reset_index(drop=True),
        use_container_width=True,
    )

    # Download
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Filtered CSV",
        data=csv,
        file_name="healthcare_filtered.csv",
        mime="text/csv",
    )
