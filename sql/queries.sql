-- =============================================================================
-- Healthcare Analytics — SQL Query Library
-- =============================================================================
-- A comprehensive set of analytical SQL queries for the patient flow dataset.
-- Run against the SQLite in-memory DB (ml_analysis.py) or any RDBMS after
-- importing the CSV via the schema in schema.sql.
-- =============================================================================


-- ── Q1: Patient Volume Summary ────────────────────────────────────────────────
-- Total patients, admission rate, average wait, and average satisfaction.
SELECT
    COUNT(*)                                                          AS total_patients,
    ROUND(100.0 * SUM(CASE WHEN admission_flag = 'Admission' THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                                      AS admission_rate_pct,
    ROUND(AVG(wait_time), 2)                                          AS avg_wait_min,
    ROUND(AVG(satisfaction_score), 2)                                 AS avg_satisfaction_score
FROM patients;


-- ── Q2: Department Performance ────────────────────────────────────────────────
-- Volumes, admission rate, wait time, and satisfaction per department.
SELECT
    department_referral                                                AS department,
    COUNT(*)                                                           AS patient_count,
    ROUND(100.0 * SUM(CASE WHEN admission_flag = 'Admission' THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                                       AS admission_rate_pct,
    ROUND(AVG(wait_time), 2)                                           AS avg_wait_min,
    ROUND(AVG(satisfaction_score), 2)                                  AS avg_satisfaction
FROM patients
GROUP BY department_referral
ORDER BY patient_count DESC;


-- ── Q3: Racial/Ethnic Equity Metrics ─────────────────────────────────────────
-- Compare admission rates, wait times, and satisfaction by race.
SELECT
    race,
    COUNT(*)                                                           AS total,
    ROUND(100.0 * SUM(CASE WHEN admission_flag = 'Admission' THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                                       AS admission_rate_pct,
    ROUND(AVG(wait_time), 2)                                           AS avg_wait_min,
    ROUND(AVG(satisfaction_score), 2)                                  AS avg_satisfaction
FROM patients
GROUP BY race
ORDER BY admission_rate_pct DESC;


-- ── Q4: Gender-Based Analysis ─────────────────────────────────────────────────
SELECT
    gender,
    COUNT(*)                                                           AS patient_count,
    ROUND(100.0 * SUM(CASE WHEN admission_flag = 'Admission' THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                                       AS admission_rate_pct,
    ROUND(AVG(wait_time), 2)                                           AS avg_wait_min,
    ROUND(AVG(satisfaction_score), 2)                                  AS avg_satisfaction
FROM patients
GROUP BY gender;


-- ── Q5: Age Group Breakdown ───────────────────────────────────────────────────
SELECT
    age_group,
    COUNT(*)                                                           AS patient_count,
    ROUND(AVG(age), 1)                                                 AS avg_age,
    ROUND(100.0 * SUM(CASE WHEN admission_flag = 'Admission' THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                                       AS admission_rate_pct,
    ROUND(AVG(wait_time), 2)                                           AS avg_wait_min
FROM patients
GROUP BY age_group
ORDER BY MIN(age);


-- ── Q6: Hourly Arrival Patterns ───────────────────────────────────────────────
-- Peak hours, admission rates, and average wait by hour.
SELECT
    hour,
    COUNT(*)                                                           AS patient_count,
    ROUND(100.0 * SUM(CASE WHEN admission_flag = 'Admission' THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                                       AS admission_rate_pct,
    ROUND(AVG(wait_time), 2)                                           AS avg_wait_min
FROM patients
WHERE hour IS NOT NULL
GROUP BY hour
ORDER BY hour;


-- ── Q7: Day-of-Week Patterns ──────────────────────────────────────────────────
SELECT
    day_of_week,
    COUNT(*)                                                           AS patient_count,
    ROUND(AVG(wait_time), 2)                                           AS avg_wait_min,
    ROUND(AVG(satisfaction_score), 2)                                  AS avg_satisfaction
FROM patients
GROUP BY day_of_week
ORDER BY CASE day_of_week
    WHEN 'Monday'    THEN 1 WHEN 'Tuesday'   THEN 2 WHEN 'Wednesday' THEN 3
    WHEN 'Thursday'  THEN 4 WHEN 'Friday'    THEN 5 WHEN 'Saturday'  THEN 6
    ELSE 7 END;


-- ── Q8: Monthly Trend ─────────────────────────────────────────────────────────
SELECT
    year,
    month,
    month_name,
    COUNT(*)                                                           AS patient_count,
    ROUND(AVG(wait_time), 2)                                           AS avg_wait_min
FROM patients
GROUP BY year, month, month_name
ORDER BY year, month;


-- ── Q9: High Wait + Admitted Patients (Operational Alert) ────────────────────
-- Patients with long waits who were eventually admitted — flag for review.
SELECT
    patient_id,
    age,
    gender,
    race,
    department_referral,
    wait_time,
    satisfaction_score,
    admission_date
FROM patients
WHERE wait_time > 50
  AND admission_flag = 'Admission'
ORDER BY wait_time DESC
LIMIT 50;


-- ── Q10: Low Satisfaction Deep Dive ──────────────────────────────────────────
-- Patients scoring 3 or below — identify patterns.
SELECT
    race,
    gender,
    age_group,
    department_referral,
    COUNT(*)                                                           AS count,
    ROUND(AVG(satisfaction_score), 2)                                  AS avg_score,
    ROUND(AVG(wait_time), 2)                                           AS avg_wait
FROM patients
WHERE satisfaction_score IS NOT NULL
  AND satisfaction_score <= 3
GROUP BY race, gender, age_group, department_referral
ORDER BY count DESC
LIMIT 20;


-- ── Q11: Satisfaction vs Wait-Time Buckets ───────────────────────────────────
-- Compare average satisfaction across wait-time quartile buckets.
SELECT
    CASE
        WHEN wait_time <= 20 THEN '0–20 min'
        WHEN wait_time <= 40 THEN '21–40 min'
        WHEN wait_time <= 60 THEN '41–60 min'
        ELSE '60+ min'
    END                                                                AS wait_bucket,
    COUNT(*)                                                           AS patient_count,
    ROUND(AVG(satisfaction_score), 2)                                  AS avg_satisfaction
FROM patients
WHERE satisfaction_score IS NOT NULL
GROUP BY wait_bucket
ORDER BY MIN(wait_time);


-- ── Q12: Year-over-Year Comparison ────────────────────────────────────────────
SELECT
    year,
    COUNT(*)                                                           AS total_patients,
    ROUND(100.0 * SUM(CASE WHEN admission_flag = 'Admission' THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                                       AS admission_rate_pct,
    ROUND(AVG(wait_time), 2)                                           AS avg_wait_min,
    ROUND(AVG(satisfaction_score), 2)                                  AS avg_satisfaction
FROM patients
GROUP BY year
ORDER BY year;
