# Location-Based Rainfall Pattern Analysis and Prediction System

A machine learning and data analytics application built with **Python**, **Scikit-Learn**, **Plotly**, and **Streamlit**.

This system allows users to select any Indian district (such as **Pune**, **Mumbai**, or **Nagpur**) and automatically retrieve its historical rainfall records, analyze rainfall trends and distributions, generate interactive visualizations, and estimate rainy-day probabilities for any selected month or season without entering manual weather measurements.

---

## 1. Project Objective

The objective of this project is to provide a location-centric historical rainfall analysis and probabilistic modeling tool.

- **No Manual Weather Inputs:** Users do not need to input temperature, atmospheric pressure, humidity, or rainfall measurements.
- **Historical Analysis vs Live Forecasting:** This project is designed for historical pattern analysis based on recorded observations, not real-time atmospheric weather forecasting.
- **District-Level Resolution:** Covers 728 districts across all 39 States and Union Territories of India.

---

## 2. Dataset Architecture & Source

The application utilizes genuine daily district-level rainfall bulletin data published by the **India Meteorological Department (IMD)**.

- **Dataset File:** `data/rainfall_districtwise_daily_imd.csv`
- **Total Records:** 40,007 daily observations
- **Districts Covered:** 728 unique districts across 39 States / Union Territories
- **Temporal Coverage:** 19 August 2026 to 12 October 2026 (55 consecutive calendar days spanning the late South-West Monsoon and onset of the Post-Monsoon transition)
- **Measurement Units:** All rainfall observations (`Daily Actual`, `Daily Normal`, `Weekly Actual`, `Cumulative Actual`, etc.) are recorded in **millimeters (mm)**.
- **Standard IMD Rain Categories:**
  - `NR`: No Rain (0.0 mm)
  - `LD`: Large Deficient (-60% to -99% departure from normal)
  - `D`: Deficient (-20% to -59% departure)
  - `N`: Normal (-19% to +19% departure)
  - `E`: Excess (+20% to +59% departure)
  - `LE`: Large Excess (≥ +60% departure)
  - `ND`: No Climatological Normal Data (Normal = 0)

---

## 3. Machine Learning Methodology

### Target Definition
A calendar day is classified into a binary probabilistic target according to the official IMD meteorological standard:
- **Rainy Day ($y = 1$):** Recorded daily rainfall $\ge 2.5\text{ mm}$.
- **Non-Rainy Day ($y = 0$):** Recorded daily rainfall $< 2.5\text{ mm}$.
- *(The threshold is configurable in the application sidebar).*

### Feature Engineering
Features are derived strictly from the location and calendar information present in the dataset without inventing unavailable weather variables:
- **Categorical:** `district`, `state` (one-hot encoded via Scikit-Learn `ColumnTransformer`)
- **Numerical:** `month`, `day_of_year`, `daily_normal` (IMD climatological normal baseline for that district-day, scaled via `StandardScaler`)

### Chronological Train-Test Split
To prevent data leakage and respect time-series dependencies:
- **Training Set (80% early dates):** 19 August 2026 to 01 October 2026 (32,007 observations).
- **Test Set (20% late dates):** 02 October 2026 to 12 October 2026 (8,000 observations).

### Evaluated Models
1. **Historical Frequency Baseline:** Predicts the historical prior class frequency ($35.9\%$ rainy rate during the training period).
2. **Logistic Regression Pipeline:** Scikit-Learn pipeline with one-hot encoding, feature standardization, and lbfgs solver.
3. **Random Forest Classifier Pipeline:** Scikit-Learn nonlinear ensemble pipeline with balanced class weighting (`class_weight='balanced_subsample'`) and calibrated probability estimation.

### Evaluation Metrics (Chronological Test Set)

| Metric | Historical Frequency Baseline | Logistic Regression Pipeline | Random Forest Classifier Pipeline |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 87.56% | 86.58% | **87.09%** |
| **Precision** | 0.00% | 29.53% | **27.38%** |
| **Recall** | 0.00% | 5.73% | **2.31%** |
| **F1 Score** | 0.0000 | 0.0960 | **0.0426** |
| **ROC-AUC** | 0.5000 | 0.6066 | **0.6732** |
| **Brier Score (Calibration)** | 0.1641 | 0.1258 | **0.1489** |

*Key finding on temporal evaluation:* During October, South-West Monsoon withdrawal causes rainfall prevalence to drop drastically from $35.9\%$ to $12.4\%$. A naive baseline achieves $87.56\%$ accuracy simply by predicting zero rain everywhere, illustrating why accuracy alone is deceptive and why continuous probabilistic calibration (ROC-AUC and Brier Score) is critical.

---

## 4. How Rainfall Probability is Calculated

The system clearly differentiates between:
1. **Historical Observed Frequency (%):**
   $$\text{Observed Frequency} = \left(\frac{\text{Rainy Days Observed in Target Period}}{\text{Total Days Recorded in Target Period}}\right) \times 100$$
2. **Machine Learning Model Probability (%):**
   The trained Scikit-Learn pipeline estimates the continuous daily conditional probability $P(\text{Rainy} \mid \text{District}, \text{Month}, \text{DayOfYear}, \text{Normal})$ averaged across all days of the selected period.

---

## 5. Visualizations Implemented

The application provides a clean, focused dashboard featuring two interactive Plotly visualizations:
- **Chart 1 (Monthly Rainfall Pattern):** Bar chart displaying monthly rainfall totals across the observed months, with the selected date's month clearly highlighted and values labeled in millimeters (mm).
- **Chart 2 (Historical Daily Rainfall Trend):** Interactive time-series showing the selected district's recorded daily rainfall, with the selected date highlighted when a direct observation exists in the record and an IMD rainy-day threshold reference line.

---

## 6. Project Structure

```
Rainfall_Prediction/
│
├── data/
│   └── rainfall_districtwise_daily_imd.csv   # IMD daily district-wise dataset (40,007 rows)
│
├── models/
│   └── rainfall_model.pkl                    # Serialized Scikit-Learn pipelines & metrics
│
├── notebooks/
│   └── rainfall_analysis.ipynb               # Exploratory analysis & modeling notebook
│
├── src/
│   ├── __init__.py                           # Package initialization
│   ├── data_processing.py                    # Preprocessing, normalization, and statistical pipeline
│   ├── visualization.py                      # Interactive Plotly chart generation
│   ├── train_model.py                        # ML training, chronological split & evaluation
│   └── prediction.py                         # Probability inference & interpretation engine
│
├── app.py                                    # Minimalist interactive Streamlit dashboard
├── train_model.py                            # Root CLI training entry point
├── requirements.txt                          # Minimal dependencies
└── README.md                                 # Complete documentation
```

---

## 7. Installation and Execution

### Prerequisites
- Python 3.9+ (tested on Python 3.13)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Re-train Machine Learning Models
```bash
python train_model.py
```
*(The trained model is already saved in `models/rainfall_model.pkl` and will be loaded automatically).*

### Step 3: Launch Streamlit Dashboard
```bash
streamlit run app.py
```

Open the local URL displayed in the terminal (typically `http://localhost:8501`) in your browser.

---

## 8. Dataset Limitations

1. **Date Window:** The dataset spans 55 consecutive calendar days (19 August 2026 to 12 October 2026). It captures late monsoon dynamics and the beginning of monsoon retreat, but does not span complete 365-day multi-year cycles.
2. **Partial-Year Records:** Annual summaries for 2026 reflect the 55 recorded days and must not be confused with total annual precipitation.
3. **District Resolution:** Observations reflect district-wide rain gauge averages. Microclimatic variations within large districts (e.g. urban vs rural rain gauges) are not separated.
4. **No Weather Fabrication:** Missing observations are not silently zero-filled, and extra synthetic weather columns are not fabricated.
