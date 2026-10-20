"""Data Loading, Preprocessing, and Historical Statistics Pipeline for Rainfall Analysis.

This module loads the India Meteorological Department (IMD) district-wise daily rainfall
dataset, cleans and normalizes column names, parses dates, validates numerical fields,
and provides utility functions for location filtering and statistical aggregation.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Default IMD threshold for a "Rainy Day" is 2.5 mm
DEFAULT_RAINY_THRESHOLD_MM = 2.5

# Standard IMD Category Descriptions
IMD_CATEGORY_MAP = {
    'NR': 'No Rain (0 mm)',
    'LD': 'Large Deficient (-60% to -99%)',
    'D': 'Deficient (-20% to -59%)',
    'N': 'Normal (-19% to +19%)',
    'E': 'Excess (+20% to +59%)',
    'LE': 'Large Excess (>= +60%)',
    'ND': 'No Normal Data (Normal = 0)'
}


def normalize_column_name(col: str) -> str:
    """Normalize raw CSV column name to clean snake_case identifier."""
    cleaned = col.replace('\n', ' ').strip()
    cleaned = cleaned.lower()
    cleaned = cleaned.replace(' ', '_')
    # Standardize known misspellings in IMD bulletin exports
    cleaned = cleaned.replace('acutual', 'actual')
    cleaned = cleaned.replace('departue', 'departure')
    return cleaned


def load_and_preprocess_data(
    filepath: str = "data/rainfall_districtwise_daily_imd.csv",
    threshold: float = DEFAULT_RAINY_THRESHOLD_MM
) -> pd.DataFrame:
    """Load and clean the district-wise daily rainfall CSV dataset.
    
    Parameters:
        filepath: Relative or absolute path to the CSV file.
        threshold: Rainfall in mm required to classify a day as 'rainy' (default: 2.5 mm).
        
    Returns:
        Cleaned, chronologically sorted DataFrame with derived features.
    """
    if not os.path.exists(filepath):
        # Check relative to project root
        alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Rainfall dataset not found at: {filepath}")

    # Read CSV
    df = pd.read_csv(filepath)

    # Rename columns with normalized names
    rename_dict = {col: normalize_column_name(col) for col in df.columns}
    df = df.rename(columns=rename_dict)

    # Clean text columns
    if 'state' in df.columns:
        df['state'] = df['state'].astype(str).str.strip().str.upper()
    if 'district' in df.columns:
        df['district'] = df['district'].astype(str).str.strip().str.upper()

    # Parse Date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        # Drop rows where date failed to parse (if any)
        df = df.dropna(subset=['date'])

    # Numeric conversion for rainfall measurements (units: millimeters mm)
    numeric_columns = [
        'daily_actual', 'daily_normal',
        'weekly_actual', 'weekly_normal',
        'cumulative_actual', 'cumulative_normal',
        'monthly_actual', 'monthly_normal'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure daily_actual is non-negative and valid
    if 'daily_actual' in df.columns:
        # Avoid treating missing records as 0; keep only valid observed records
        df = df.dropna(subset=['daily_actual'])
        df['daily_actual'] = df['daily_actual'].clip(lower=0.0)

    # Parse departure percentages (strip '%' and convert to float)
    departure_cols = ['daily_departure_per', 'weekly_departure_per', 'cumulative_departure_per', 'monthly_departure_per']
    for col in departure_cols:
        if col in df.columns:
            df[col + '_val'] = (
                df[col]
                .astype(str)
                .str.replace('%', '', regex=False)
                .str.strip()
            )
            df[col + '_val'] = pd.to_numeric(df[col + '_val'], errors='coerce')

    # Categorize rainfall category descriptions
    if 'daily_category' in df.columns:
        df['daily_category'] = df['daily_category'].astype(str).str.strip().str.upper()
        df['daily_category_desc'] = df['daily_category'].map(IMD_CATEGORY_MAP).fillna(df['daily_category'])

    # Sort chronologically
    df = df.sort_values(by=['date', 'state', 'district']).reset_index(drop=True)

    # Deduplicate location-date pairs (keep first observation)
    subset_keys = [c for c in ['state', 'district', 'date'] if c in df.columns]
    if len(subset_keys) == 3:
        df = df.drop_duplicates(subset=subset_keys, keep='first').reset_index(drop=True)

    # Feature Engineering: Calendar features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%B')
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.day_name()

    # Define Seasons in Indian Climatology:
    # August - September: South-West Monsoon Core / Late Monsoon
    # October: Post-Monsoon / Retreating Monsoon Transition
    def assign_season(m: int) -> str:
        if m in [6, 7, 8, 9]:
            return "South-West Monsoon"
        elif m in [10, 11, 12]:
            return "Post-Monsoon (Transition)"
        elif m in [1, 2]:
            return "Winter"
        else:
            return "Pre-Monsoon / Summer"

    df['season'] = df['month'].apply(assign_season)

    # Define Binary Rainy Day Target (1 = Rainy, 0 = Non-rainy)
    df['rainy'] = (df['daily_actual'] >= threshold).astype(int)

    # Display label for UI selectbox: "DISTRICT (STATE)"
    df['location_label'] = df['district'] + " (" + df['state'] + ")"

    return df


def get_available_locations(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Return a sorted list of unique locations with district, state, and display label."""
    unique_locs = (
        df[['district', 'state', 'location_label']]
        .drop_duplicates()
        .sort_values(by=['state', 'district'])
    )
    return unique_locs.to_dict(orient='records')


def get_all_states(df: pd.DataFrame) -> List[str]:
    """Return a sorted list of unique states in the dataset."""
    return sorted(df['state'].dropna().unique().tolist())


def filter_by_location(
    df: pd.DataFrame,
    district: str,
    state: Optional[str] = None
) -> pd.DataFrame:
    """Filter records for a specific district (and optionally state)."""
    dist_upper = district.strip().upper()
    sub = df[df['district'] == dist_upper]
    if state:
        state_upper = state.strip().upper()
        sub = sub[sub['state'] == state_upper]
    return sub.sort_values(by='date').reset_index(drop=True)


def calculate_historical_statistics(
    df_loc: pd.DataFrame,
    threshold: float = DEFAULT_RAINY_THRESHOLD_MM
) -> Dict[str, Any]:
    """Calculate thorough historical rainfall metrics for a filtered location dataset.
    
    Parameters:
        df_loc: Preprocessed DataFrame filtered for one location.
        threshold: Threshold in mm for classifying a rainy day.
        
    Returns:
        Dictionary of comprehensive historical metrics and labels.
    """
    total_days = len(df_loc)
    if total_days == 0:
        return {
            "total_days": 0,
            "error": "No observations available for this location."
        }

    daily_rain = df_loc['daily_actual']
    rainy_mask = daily_rain >= threshold
    rainy_days = int(rainy_mask.sum())
    non_rainy_days = total_days - rainy_days
    rainy_percentage = (rainy_days / total_days) * 100.0 if total_days > 0 else 0.0

    total_rainfall = float(daily_rain.sum())
    avg_daily_rainfall = float(daily_rain.mean())
    avg_rainy_day_rainfall = float(daily_rain[rainy_mask].mean()) if rainy_days > 0 else 0.0

    # Max single-day record
    max_idx = daily_rain.idxmax()
    max_single_day_rain = float(daily_rain.loc[max_idx])
    max_single_day_date = df_loc.loc[max_idx, 'date'].strftime('%d-%b-%Y')

    # Date range
    min_date = df_loc['date'].min()
    max_date = df_loc['date'].max()

    # Monthly rainfall totals & counts
    monthly_summary = []
    for month_num, grp in df_loc.groupby('month'):
        m_name = grp['month_name'].iloc[0]
        m_total = float(grp['daily_actual'].sum())
        m_days = len(grp)
        m_rainy = int((grp['daily_actual'] >= threshold).sum())
        m_freq = (m_rainy / m_days) * 100.0 if m_days > 0 else 0.0
        m_avg_daily = float(grp['daily_actual'].mean())
        monthly_summary.append({
            "month": month_num,
            "month_name": m_name,
            "total_rainfall_mm": round(m_total, 2),
            "observed_days": m_days,
            "rainy_days": m_rainy,
            "rainy_frequency_pct": round(m_freq, 2),
            "avg_daily_mm": round(m_avg_daily, 2)
        })

    # Sort monthly summary by calendar order
    monthly_summary = sorted(monthly_summary, key=lambda x: x['month'])

    # Wettest and driest months
    if monthly_summary:
        wettest_month = max(monthly_summary, key=lambda x: x['total_rainfall_mm'])
        driest_month = min(monthly_summary, key=lambda x: x['total_rainfall_mm'])
    else:
        wettest_month = None
        driest_month = None

    # Seasonal totals
    seasonal_summary = []
    for s_name, grp in df_loc.groupby('season'):
        s_total = float(grp['daily_actual'].sum())
        s_days = len(grp)
        s_rainy = int((grp['daily_actual'] >= threshold).sum())
        s_freq = (s_rainy / s_days) * 100.0 if s_days > 0 else 0.0
        seasonal_summary.append({
            "season": s_name,
            "total_rainfall_mm": round(s_total, 2),
            "observed_days": s_days,
            "rainy_days": s_rainy,
            "rainy_frequency_pct": round(s_freq, 2)
        })

    # Normal comparison (if daily_normal is present)
    has_normal = 'daily_normal' in df_loc.columns and not df_loc['daily_normal'].isnull().all()
    if has_normal:
        total_normal = float(df_loc['daily_normal'].sum())
        normal_departure_pct = (
            ((total_rainfall - total_normal) / total_normal) * 100.0
            if total_normal > 0 else 0.0
        )
    else:
        total_normal = 0.0
        normal_departure_pct = 0.0

    # Annual summary note: The dataset covers 2026 for 55 days (Aug 19 to Oct 12).
    # Clearly label as a partial year to prevent misleading comparisons.
    annual_summary = [{
        "year": int(df_loc['year'].iloc[0]),
        "total_rainfall_mm": round(total_rainfall, 2),
        "observed_days": total_days,
        "is_complete_year": False,
        "note": f"Partial Year: {total_days} days recorded ({min_date.strftime('%d-%b-%Y')} to {max_date.strftime('%d-%b-%Y')})"
    }]

    return {
        "district": df_loc['district'].iloc[0],
        "state": df_loc['state'].iloc[0],
        "total_days": total_days,
        "date_range_start": min_date.strftime('%d-%b-%Y'),
        "date_range_end": max_date.strftime('%d-%b-%Y'),
        "total_rainfall_mm": round(total_rainfall, 2),
        "avg_daily_rainfall_mm": round(avg_daily_rainfall, 2),
        "avg_rainy_day_rainfall_mm": round(avg_rainy_day_rainfall, 2),
        "rainy_days_count": rainy_days,
        "non_rainy_days_count": non_rainy_days,
        "rainy_percentage": round(rainy_percentage, 2),
        "threshold_mm": threshold,
        "max_single_day_rain_mm": round(max_single_day_rain, 2),
        "max_single_day_date": max_single_day_date,
        "monthly_summary": monthly_summary,
        "wettest_month": wettest_month,
        "driest_month": driest_month,
        "seasonal_summary": seasonal_summary,
        "annual_summary": annual_summary,
        "has_normal": has_normal,
        "total_normal_mm": round(total_normal, 2),
        "normal_departure_pct": round(normal_departure_pct, 2)
    }


def get_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Return top-level metadata and dataset audit information."""
    min_date = df['date'].min()
    max_date = df['date'].max()
    unique_dates = df['date'].nunique()
    unique_states = df['state'].nunique()
    unique_districts = df['district'].nunique()
    total_records = len(df)
    national_rainy_pct = (df['rainy'].mean()) * 100.0

    return {
        "total_records": total_records,
        "unique_districts": unique_districts,
        "unique_states": unique_states,
        "unique_calendar_days": unique_dates,
        "start_date": min_date.strftime('%d-%b-%Y'),
        "end_date": max_date.strftime('%d-%b-%Y'),
        "national_rainy_percentage": round(national_rainy_pct, 2),
        "units": "millimeters (mm)",
        "source": "India Meteorological Department (IMD) District-wise Daily Rainfall Bulletins",
        "coverage_period": "August 19, 2026 to October 12, 2026 (South-West Monsoon Late Phase & Early Post-Monsoon Transition)"
    }
