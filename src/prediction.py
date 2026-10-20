"""Inference and Rainfall Probability Estimation Engine.

Computes historical rainy-day frequencies and machine learning model estimated
probabilities for a user-selected district and date or period.
"""

import os
import datetime
from typing import Dict, Any, Optional, List
import joblib
import numpy as np
import pandas as pd


def load_model_bundle(model_path: str = "models/rainfall_model.pkl") -> Optional[Dict[str, Any]]:
    """Load serialized model bundle containing pipelines, baseline, and evaluation metrics."""
    if not os.path.exists(model_path):
        alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            return None
    try:
        bundle = joblib.load(model_path)
        return bundle
    except Exception as e:
        print(f"Error loading model bundle: {e}")
        return None


def estimate_rainfall_probability(
    district: str,
    state: Optional[str] = None,
    target_date: Optional[datetime.date] = None,
    month: Optional[int] = None,
    season: Optional[str] = None,
    df_all: Optional[pd.DataFrame] = None,
    model_bundle: Optional[Dict[str, Any]] = None,
    threshold: float = 2.5
) -> Dict[str, Any]:
    """Estimate rainy-day probability and historical observed frequency for a selected district and date.
    
    Parameters:
        district: District name (e.g., 'PUNE', 'MUMBAI CITY').
        state: Optional state name to disambiguate.
        target_date: Optional calendar date to analyze (datetime.date).
        month: Target month number (8 for August, 9 for September, 10 for October).
        season: Target season ('South-West Monsoon', 'Post-Monsoon (Transition)').
        df_all: Cleaned dataset DataFrame.
        model_bundle: Loaded model dictionary from load_model_bundle.
        threshold: Rainfall in mm for defining a rainy day.
        
    Returns:
        Structured dictionary with historical frequency, ML probabilities, sample size,
        and transparent interpretation text.
    """
    if df_all is None:
        return {"error": "Dataset not provided."}

    dist_upper = district.strip().upper()
    sub_df = df_all[df_all['district'] == dist_upper].copy()
    if state:
        sub_df = sub_df[sub_df['state'] == state.strip().upper()]

    if len(sub_df) == 0:
        return {
            "error": f"District '{district}' was not found in the dataset.",
            "district": district
        }

    detected_state = sub_df['state'].iloc[0]

    # Handle Target Date if provided
    has_direct_observation = False
    observed_rain_mm = None
    is_within_coverage = False
    
    if target_date is not None:
        target_month = target_date.month
        target_day = target_date.day
        target_day_of_year = target_date.timetuple().tm_yday
        
        # Check if direct observation exists for this exact date
        matching_day = sub_df[sub_df['date'].dt.date == target_date]
        if len(matching_day) > 0:
            has_direct_observation = True
            is_within_coverage = True
            observed_rain_mm = float(matching_day['daily_actual'].iloc[0])
            normal_val = (
                float(matching_day['daily_normal'].iloc[0])
                if 'daily_normal' in matching_day and pd.notnull(matching_day['daily_normal'].iloc[0])
                else sub_df['daily_normal'].median()
            )
        else:
            # Check if date is within min/max date coverage of the dataset
            min_cov = sub_df['date'].min().date()
            max_cov = sub_df['date'].max().date()
            is_within_coverage = (min_cov <= target_date <= max_cov)
            
            # Use month normal if available, else median normal for district
            month_match = sub_df[sub_df['month'] == target_month]
            if len(month_match) > 0 and 'daily_normal' in month_match:
                normal_val = float(month_match['daily_normal'].median())
            else:
                normal_val = float(sub_df['daily_normal'].median() if 'daily_normal' in sub_df else 5.0)

        # Period for summary statistics: if month is observed, use that month's observations
        month_records = sub_df[sub_df['month'] == target_month]
        if len(month_records) > 0:
            period_df = month_records
            month_names = {8: "August", 9: "September", 10: "October"}
            m_name = month_names.get(target_month, target_date.strftime('%B'))
            period_label = f"{m_name} (Historical Record: {len(period_df)} days)"
        else:
            # Fall back to full available historical record
            period_df = sub_df
            period_label = f"Available Observation Window ({len(period_df)} days)"

        # ML Model Inference for target date
        rf_prob_pct = None
        lr_prob_pct = None
        dt_prob_pct = None
        
        if model_bundle is not None:
            feature_cols = ['district', 'state', 'month', 'day_of_year', 'daily_normal']
            X_single = pd.DataFrame([{
                'district': dist_upper,
                'state': detected_state,
                'month': target_month,
                'day_of_year': target_day_of_year,
                'daily_normal': normal_val
            }])
            
            lr_pipe = model_bundle.get("lr_pipeline")
            rf_pipe = model_bundle.get("rf_pipeline")
            dt_pipe = model_bundle.get("dt_pipeline")
            
            try:
                if rf_pipe:
                    rf_prob_pct = round(float(rf_pipe.predict_proba(X_single)[0, 1]) * 100.0, 1)
            except Exception as e:
                print(f"Random Forest inference error: {e}")

            try:
                if lr_pipe:
                    lr_prob_pct = round(float(lr_pipe.predict_proba(X_single)[0, 1]) * 100.0, 1)
            except Exception as e:
                print(f"Logistic Regression inference error: {e}")

            try:
                if dt_pipe:
                    dt_prob_pct = round(float(dt_pipe.predict_proba(X_single)[0, 1]) * 100.0, 1)
            except Exception as e:
                print(f"Decision Tree inference error: {e}")

        # Summary statistics for the period
        observed_days = len(period_df)
        rainy_days = int((period_df['daily_actual'] >= threshold).sum())
        historical_frequency = (rainy_days / observed_days) * 100.0 if observed_days > 0 else 0.0
        total_rain_period = float(period_df['daily_actual'].sum())
        avg_daily_rain_period = float(period_df['daily_actual'].mean()) if observed_days > 0 else 0.0

        # Build short, clear explanation
        if is_within_coverage:
            interpretation = (
                "This estimate is based on historical rainfall patterns for the selected district and calendar period. "
                "It is not a live weather forecast."
            )
        else:
            interpretation = (
                f"Selected date ({target_date.strftime('%d %B %Y')}) is outside the available observation window "
                f"({sub_df['date'].min().strftime('%d %b %Y')} to {sub_df['date'].max().strftime('%d %b %Y')}). "
                "This estimate is a historical-pattern estimate based on the model's learned calendar dynamics "
                "rather than an observation-backed record for that specific date. It is not a live weather forecast."
            )

        return {
            "district": dist_upper,
            "state": detected_state,
            "target_date": target_date,
            "has_direct_observation": has_direct_observation,
            "observed_rain_mm": observed_rain_mm,
            "is_within_coverage": is_within_coverage,
            "period": period_label,
            "observed_days": observed_days,
            "rainy_days": rainy_days,
            "threshold_mm": threshold,
            "historical_frequency_pct": round(historical_frequency, 1),
            "rf_probability_pct": rf_prob_pct,
            "lr_probability_pct": lr_prob_pct,
            "dt_probability_pct": dt_prob_pct,
            "total_rain_period_mm": round(total_rain_period, 1),
            "avg_daily_rain_period_mm": round(avg_daily_rain_period, 2),
            "interpretation": interpretation
        }

    # Fallback / Legacy: Period-based calculation (month or season)
    period_label = "Full Observed Period (Aug-Oct 2026)"
    period_df = sub_df.copy()

    if month is not None:
        period_df = sub_df[sub_df['month'] == month]
        month_names = {8: "August", 9: "September", 10: "October"}
        period_label = f"Month of {month_names.get(month, f'Month {month}')}"
    elif season is not None:
        period_df = sub_df[sub_df['season'] == season]
        period_label = f"Season: {season}"

    observed_days = len(period_df)
    if observed_days == 0:
        return {
            "error": f"No historical records found for {dist_upper} during {period_label}.",
            "district": dist_upper,
            "state": detected_state,
            "period": period_label
        }

    rainy_days = int((period_df['daily_actual'] >= threshold).sum())
    historical_frequency = (rainy_days / observed_days) * 100.0
    total_rain_period = float(period_df['daily_actual'].sum())
    avg_daily_rain_period = float(period_df['daily_actual'].mean())

    rf_prob_pct = None
    lr_prob_pct = None
    dt_prob_pct = None

    if model_bundle is not None:
        feature_cols = ['district', 'state', 'month', 'day_of_year', 'daily_normal']
        X_infer = period_df[feature_cols].copy()
        if 'daily_normal' in X_infer.columns:
            X_infer['daily_normal'] = X_infer['daily_normal'].fillna(sub_df['daily_normal'].median())

        lr_pipe = model_bundle.get("lr_pipeline")
        rf_pipe = model_bundle.get("rf_pipeline")
        dt_pipe = model_bundle.get("dt_pipeline")

        try:
            if lr_pipe:
                lr_day_probs = lr_pipe.predict_proba(X_infer)[:, 1]
                lr_prob_pct = round(float(np.mean(lr_day_probs)) * 100.0, 1)
        except Exception:
            pass

        try:
            if rf_pipe:
                rf_day_probs = rf_pipe.predict_proba(X_infer)[:, 1]
                rf_prob_pct = round(float(np.mean(rf_day_probs)) * 100.0, 1)
        except Exception:
            pass

        try:
            if dt_pipe:
                dt_day_probs = dt_pipe.predict_proba(X_infer)[:, 1]
                dt_prob_pct = round(float(np.mean(dt_day_probs)) * 100.0, 1)
        except Exception:
            pass

    interpretation = (
        "This estimate is based on historical rainfall patterns for the selected district and calendar period. "
        "It is not a live weather forecast."
    )

    return {
        "district": dist_upper,
        "state": detected_state,
        "target_date": None,
        "has_direct_observation": False,
        "observed_rain_mm": None,
        "is_within_coverage": True,
        "period": period_label,
        "observed_days": observed_days,
        "rainy_days": rainy_days,
        "threshold_mm": threshold,
        "historical_frequency_pct": round(historical_frequency, 1),
        "rf_probability_pct": rf_prob_pct,
        "lr_probability_pct": lr_prob_pct,
        "dt_probability_pct": dt_prob_pct,
        "total_rain_period_mm": round(total_rain_period, 1),
        "avg_daily_rain_period_mm": round(avg_daily_rain_period, 2),
        "interpretation": interpretation
    }
