-- =============================================================================
-- Healthcare Analytics Database Schema
-- =============================================================================
-- This schema defines the structure for the healthcare patient flow dataset.
-- Compatible with PostgreSQL, MySQL, and SQLite (with minor type adjustments).
-- =============================================================================

-- Drop existing table if re-running
DROP TABLE IF EXISTS patients;

-- ── Main patients table ──────────────────────────────────────────────────────
CREATE TABLE patients (
    patient_id          VARCHAR(20)  PRIMARY KEY,          -- SSN-style masked ID
    admission_date      DATE         NOT NULL,              -- Date of visit
    admission_time      TIME,                               -- Time of arrival
    patient_name        VARCHAR(100),                       -- Masked last name only
    gender              VARCHAR(20),                        -- Male / Female / Other
    age                 INTEGER      CHECK (age >= 0),      -- Age in years
    race                VARCHAR(50),                        -- Self-reported race/ethnicity
    department_referral VARCHAR(50)  DEFAULT 'None',        -- Referring department
    admission_flag      VARCHAR(20)  NOT NULL,              -- Admission / Not Admission
    satisfaction_score  NUMERIC(3,1) CHECK (satisfaction_score BETWEEN 0 AND 10),
    wait_time           INTEGER      CHECK (wait_time >= 0),-- Minutes

    -- Derived / pre-computed columns (optional, populate via application)
    year                INTEGER,
    month               INTEGER,
    month_name          VARCHAR(15),
    day_of_week         VARCHAR(15),
    hour                INTEGER      CHECK (hour BETWEEN 0 AND 23),
    time_of_day         VARCHAR(25),
    age_group           VARCHAR(25),
    is_admitted         BOOLEAN      GENERATED ALWAYS AS (admission_flag = 'Admission') STORED,
    has_referral        BOOLEAN      GENERATED ALWAYS AS (department_referral <> 'None') STORED
);

-- ── Indexes for common query patterns ────────────────────────────────────────
CREATE INDEX idx_admission_date   ON patients (admission_date);
CREATE INDEX idx_admission_flag   ON patients (admission_flag);
CREATE INDEX idx_department       ON patients (department_referral);
CREATE INDEX idx_gender           ON patients (gender);
CREATE INDEX idx_race             ON patients (race);
CREATE INDEX idx_year_month       ON patients (year, month);
CREATE INDEX idx_hour             ON patients (hour);

-- ── Summary view: Department KPIs ────────────────────────────────────────────
CREATE VIEW dept_summary AS
SELECT
    department_referral                                  AS department,
    COUNT(*)                                             AS total_patients,
    ROUND(100.0 * SUM(CASE WHEN admission_flag = 'Admission' THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                         AS admission_rate_pct,
    ROUND(AVG(wait_time), 2)                             AS avg_wait_min,
    ROUND(AVG(satisfaction_score), 2)                    AS avg_satisfaction
FROM patients
GROUP BY department_referral;

-- ── Summary view: Daily volume ────────────────────────────────────────────────
CREATE VIEW daily_volume AS
SELECT
    admission_date,
    COUNT(*)                     AS patient_count,
    ROUND(AVG(wait_time), 2)     AS avg_wait_min
FROM patients
GROUP BY admission_date
ORDER BY admission_date;
