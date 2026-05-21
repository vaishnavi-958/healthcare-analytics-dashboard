# 🏥 AI Healthcare Analytics Dashboard

> **Portfolio-quality** end-to-end analytics project — Python · SQL · Machine Learning · Streamlit

A complete, production-ready healthcare analytics solution built around real patient-flow data. Includes Python data analysis, SQL queries, ML-driven predictive models, an interactive Streamlit dashboard, and a GitHub-ready project structure.

---

## ✨ Features

| Category | Capabilities |
|---|---|
| **Data Analysis** | Descriptive stats, missing-value handling, feature engineering, temporal pattern analysis |
| **SQL Queries** | 12 analytical queries covering volume, equity, trends, anomalies, and KPIs |
| **Machine Learning** | Admission prediction (RF classifier), Satisfaction estimation (RF regressor), Patient segmentation (K-Means), Wait-time anomaly detection (IQR) |
| **Streamlit Dashboard** | 8-tab interactive app with Plotly charts, sidebar filters, live SQL editor, and CSV export |
| **Data** | 9,217 patient records with demographics, referrals, satisfaction scores, and wait times |

---

## 📁 Project Structure

```
healthcare_analytics/
│
├── app.py                          # Main Streamlit dashboard (8 tabs)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   └── healthcare_data.csv         # Patient flow dataset (9,217 records)
│
├── analysis/
│   ├── __init__.py
│   ├── data_analysis.py            # Statistical analysis + SQL-in-Python report
│   └── ml_analysis.py              # ML models (RF, K-Means, IQR anomaly detection)
│
├── sql/
│   ├── schema.sql                  # Full database schema + views + indexes
│   └── queries.sql                 # 12 analytical SQL queries (commented)
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py              # Cached data loading + preprocessing
│   └── visualizations.py          # Plotly chart factory functions
│
├── screenshots/                    # Dashboard screenshots (auto-populated)
│
└── .streamlit/
    └── config.toml                 # Streamlit server configuration
```

---

## 🚀 Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/healthcare-analytics.git
cd healthcare-analytics/healthcare_analytics
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Visit **http://localhost:8501** in your browser.

---

## 🔬 Running Standalone Analysis

### Full statistical report (prints to terminal)

```bash
python analysis/data_analysis.py
```

### ML model training & evaluation

```bash
python analysis/ml_analysis.py
```

---

## 📊 Dashboard Tabs

| Tab | Description |
|---|---|
| 📊 **Overview** | KPI metrics, daily volume trend, monthly breakdown, YoY comparison |
| 👥 **Demographics** | Gender, race, age distribution, age group × admission status |
| 🏢 **Departments** | Referral volume, admission rates, and avg wait by department |
| ⏱️ **Wait Times** | Distribution, day-of-week patterns, hour-of-day heatmap, anomaly flags |
| ⭐ **Satisfaction** | Score distribution, race/age correlations, satisfaction vs wait scatter |
| 🤖 **AI Insights** | On-demand ML model training, feature importances, clustering, anomalies |
| 🗃️ **SQL Explorer** | 6 pre-built queries + free-form SQL editor with quick charts |
| 📋 **Raw Data** | Searchable, filterable dataset with CSV export |

---

## 🤖 Machine Learning Models

### 1. Admission Predictor (Random Forest Classifier)
- **Target:** Binary — Admitted / Not Admitted
- **Features:** Age, gender, race, department, hour, day, wait time
- **Evaluation:** Accuracy, 5-fold cross-validated F1-score, classification report

### 2. Satisfaction Estimator (Random Forest Regressor)
- **Target:** Patient satisfaction score (0–10)
- **Features:** Same demographic + temporal features
- **Evaluation:** MAE, R²

### 3. Patient Segmentation (K-Means, k=4)
- **Features:** Age, wait time, satisfaction, hour, admission status, referral status
- **Output:** Cluster profiles with average metrics and cluster size

### 4. Wait-Time Anomaly Detection (IQR Method)
- **Method:** Flag observations outside Q1 − 1.5·IQR and Q3 + 1.5·IQR
- **Output:** Annotated anomaly DataFrame + visual scatter plot

---

## 🗃️ SQL Highlights

```sql
-- Admission rate by race (equity analysis)
SELECT race,
       COUNT(*) AS total,
       ROUND(100.0 * SUM(CASE WHEN admission_flag = 'Admission' THEN 1 ELSE 0 END) / COUNT(*), 2)
           AS admission_rate_pct
FROM patients
GROUP BY race ORDER BY admission_rate_pct DESC;

-- Hourly peak analysis
SELECT hour, COUNT(*) AS patient_count,
       ROUND(AVG(wait_time), 2) AS avg_wait_min
FROM patients
GROUP BY hour ORDER BY hour;
```

See **`sql/queries.sql`** for all 12 queries with full documentation.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Interactive web dashboard |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `plotly` | Interactive charts |
| `scikit-learn` | ML models (RF, K-Means, metrics) |
| `scipy` | Statistical tests (Pearson, T-test) |
| `sqlalchemy` | SQL interface |
| `matplotlib` / `seaborn` | Static chart support |

---

## 📈 Dataset Fields

| Column | Description |
|---|---|
| `Patient Id` | Masked patient identifier |
| `Patient Admission Date` | Date of visit |
| `Patient Admission Time` | Time of arrival |
| `Patient Gender` | Male / Female |
| `Patient Age` | Age in years (0–100+) |
| `Patient Race` | Self-reported race/ethnicity |
| `Department Referral` | Referring department (None if walk-in) |
| `Patient Admission Flag` | Admission / Not Admission |
| `Patient Satisfaction Score` | 0–10 (sparse — ~30% response rate) |
| `Patient Waittime` | Wait time in minutes |

---

## 🧠 Key Insights (from the data)

- **Admission rate** is approximately 50% across the full dataset
- **Wait time** has no statistically significant correlation with satisfaction score
- **General Practice** is the most common referral department
- **Seniors (65+)** have the highest admission rate of any age group
- Peak arrival hours cluster in the **morning (8–11 AM)** and **afternoon (1–4 PM)**

---

## ⚙️ GitHub Actions CI

A full CI pipeline runs automatically on every push and pull request to `main`/`master`.

```
.github/workflows/ci.yml
```

### Pipeline Steps

| Step | What it does |
|---|---|
| **Set up Python 3.11** | Installs the runtime with pip caching |
| **Install dependencies** | `pip install -r requirements.txt` |
| **Validate dataset** | Checks shape, required columns, age range, admission flag values, no empty rows |
| **Run data analysis** | Executes `analysis/data_analysis.py` — full statistical report |
| **Run ML analysis** | Executes `analysis/ml_analysis.py` — trains all 4 models |
| **Lint (flake8)** | Style checks at max 120 chars, warnings only (build never fails on lint) |

### Adding the CI badge to your README

Once pushed to GitHub, add this badge at the top of your README (replace `YOUR_USERNAME` and `REPO_NAME`):

```markdown
![CI](https://github.com/YOUR_USERNAME/REPO_NAME/actions/workflows/ci.yml/badge.svg)
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## 📄 License

MIT — free to use, adapt, and share.
