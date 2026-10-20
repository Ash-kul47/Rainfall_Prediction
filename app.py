"""Rainfall Probability Explorer - Minimalist Streamlit Application.

A clean, focused tool for selecting an Indian district and date to view its
historical rainfall analysis and estimated probability of rain.
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Rainfall Probability Explorer",
    page_icon="🌧️",
    layout="wide"
)

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import (
    load_and_preprocess_data,
    get_available_locations,
    filter_by_location,
    DEFAULT_RAINY_THRESHOLD_MM
)
from src.prediction import (
    load_model_bundle,
    estimate_rainfall_probability
)
from src.visualization import (
    plot_monthly_rainfall_pattern,
    plot_daily_rainfall_trend
)

# Cached dataset loading
@st.cache_data(show_spinner="Loading rainfall dataset...")
def get_dataset():
    return load_and_preprocess_data(threshold=DEFAULT_RAINY_THRESHOLD_MM)

# Cached model bundle loading
@st.cache_resource(show_spinner="Loading predictive model...")
def get_models():
    bundle = load_model_bundle()
    if bundle is None:
        from src.train_model import train_and_save_models
        bundle = train_and_save_models()
    return bundle


# Initialize data and models
try:
    df_raw = get_dataset()
    models_bundle = get_models()
except Exception as e:
    st.error(f"Error loading required resources: {e}")
    st.stop()

# ---------------------------------------------------------
# A. Header
# ---------------------------------------------------------
st.title("🌧️ Rainfall Probability Explorer")
st.caption("Historical rainfall analysis and probability estimation by district")

st.write("")

# ---------------------------------------------------------
# B. Input Section (District selector and Date picker)
# ---------------------------------------------------------
locations_list = get_available_locations(df_raw)
location_labels = [loc['location_label'] for loc in locations_list]

# Default to Pune if available
default_idx = 0
for idx, lbl in enumerate(location_labels):
    if "PUNE" in lbl:
        default_idx = idx
        break

input_col1, input_col2 = st.columns(2)

with input_col1:
    selected_label = st.selectbox(
        "District / Location",
        options=location_labels,
        index=default_idx,
        help="Search and select an Indian district."
    )

# Retrieve chosen district & state
chosen_record = next(loc for loc in locations_list if loc['location_label'] == selected_label)
selected_district = chosen_record['district']
selected_state = chosen_record['state']

with input_col2:
    # Default date within available dataset coverage
    default_date = datetime.date(2026, 9, 15)
    selected_date = st.date_input(
        "Date",
        value=default_date,
        help="Select the date to analyze."
    )

# Filter dataset for selected district
df_district = filter_by_location(df_raw, selected_district, selected_state)

if len(df_district) == 0:
    st.warning(f"No historical records available for district '{selected_district}'.")
    st.stop()

# Compute probability and period statistics
prob_data = estimate_rainfall_probability(
    district=selected_district,
    state=selected_state,
    target_date=selected_date,
    df_all=df_raw,
    model_bundle=models_bundle,
    threshold=DEFAULT_RAINY_THRESHOLD_MM
)

if "error" in prob_data:
    st.error(prob_data["error"])
    st.stop()

# Date coverage notice if selected date is outside observation window
if not prob_data.get("is_within_coverage", True):
    st.info(
        f"Notice: The selected date ({selected_date.strftime('%d %B %Y')}) falls outside the available observation "
        f"window ({df_district['date'].min().strftime('%d %b %Y')} to {df_district['date'].max().strftime('%d %b %Y')}). "
        "The result below is a historical-pattern estimate based on the model's learned calendar features rather than "
        "an actual observation for that specific date."
    )

st.write("")

# ---------------------------------------------------------
# C. Four Compact Metric Cards
# ---------------------------------------------------------
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

prob_display = prob_data.get('rf_probability_pct', prob_data.get('historical_frequency_pct', 0.0))
period_label = prob_data.get('period', 'Historical Period')

with m_col1:
    st.metric(
        label="Historical Rainfall Probability",
        value=f"{prob_display:.0f}%",
        help=f"Estimated probability of rain for {selected_district.title()} on {selected_date.strftime('%d %B')}."
    )

with m_col2:
    st.metric(
        label="Average Historical Rainfall",
        value=f"{prob_data['avg_daily_rain_period_mm']:.2f} mm",
        help=f"Average recorded rainfall per day for {period_label}."
    )

with m_col3:
    st.metric(
        label="Rainy-Day Frequency",
        value=f"{prob_data['historical_frequency_pct']:.1f}%",
        help=f"Percentage of valid historical observations in {period_label} with rainfall ≥ {DEFAULT_RAINY_THRESHOLD_MM} mm."
    )

with m_col4:
    st.metric(
        label="Historical Rainfall Total",
        value=f"{prob_data['total_rain_period_mm']:.1f} mm",
        help=f"Cumulative rainfall recorded over {period_label}."
    )

st.write("")

# ---------------------------------------------------------
# D. Main Probability Result & Three-Model Comparison
# ---------------------------------------------------------
st.markdown(f"### {selected_district.title()} — {selected_date.strftime('%d %B %Y')}")

lr_prob = prob_data.get('lr_probability_pct')
rf_prob = prob_data.get('rf_probability_pct')
dt_prob = prob_data.get('dt_probability_pct')

# Compact Three-Model Comparison Side by Side
comp_col1, comp_col2, comp_col3 = st.columns(3)

with comp_col1:
    st.markdown("##### Logistic Regression")
    val_lr = f"{lr_prob:.1f}%" if lr_prob is not None else "N/A"
    st.markdown(f"### {val_lr}")
    if lr_prob is not None:
        st.progress(min(max(float(lr_prob) / 100.0, 0.0), 1.0))
    st.caption("Linear model (Brier score: 0.1258, ROC-AUC: 0.6066)")

with comp_col2:
    st.markdown("##### Random Forest")
    val_rf = f"{rf_prob:.1f}%" if rf_prob is not None else "N/A"
    st.markdown(f"### {val_rf}")
    if rf_prob is not None:
        st.progress(min(max(float(rf_prob) / 100.0, 0.0), 1.0))
    st.caption("Ensemble model (ROC-AUC: 0.6732, Brier score: 0.1489)")

with comp_col3:
    st.markdown("##### Decision Tree")
    val_dt = f"{dt_prob:.1f}%" if dt_prob is not None else "N/A"
    st.markdown(f"### {val_dt}")
    if dt_prob is not None:
        st.progress(min(max(float(dt_prob) / 100.0, 0.0), 1.0))
    st.caption("Single tree model (Recall: 10.65%, F1: 0.1464, ROC-AUC: 0.6436)")

st.caption(
    "These estimates are based on historical rainfall patterns for the selected district and calendar period (19 August to 12 October 2026). "
    "They are not live weather forecasts."
)

st.divider()

# ---------------------------------------------------------
# E. Exactly Two Historical Visualizations
# ---------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_monthly = plot_monthly_rainfall_pattern(
        df_district,
        highlight_month=selected_date.month
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

with chart_col2:
    fig_daily = plot_daily_rainfall_trend(
        df_district,
        highlight_date=selected_date,
        threshold=DEFAULT_RAINY_THRESHOLD_MM
    )
    st.plotly_chart(fig_daily, use_container_width=True)
