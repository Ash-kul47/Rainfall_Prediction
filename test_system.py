"""Verification and test suite for simplified Rainfall Probability Explorer."""

import datetime
import pandas as pd
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

def run_tests():
    print("--- 1. Testing Dataset & Locations ---")
    df = load_and_preprocess_data("data/rainfall_districtwise_daily_imd.csv")
    assert len(df) == 40007
    locations = get_available_locations(df)
    assert len(locations) == 728
    print(f"Locations test passed: {len(locations)} districts available.")

    print("\n--- 2. Testing Date Selection & Metric Calculations (Within Coverage) ---")
    bundle = load_model_bundle("models/rainfall_model.pkl")
    assert bundle is not None
    assert "lr_pipeline" in bundle, "Missing lr_pipeline in bundle"
    assert "rf_pipeline" in bundle, "Missing rf_pipeline in bundle"
    assert "dt_pipeline" in bundle, "Missing dt_pipeline in bundle"
    assert "metrics" in bundle, "Missing metrics in bundle"
    assert "Logistic Regression" in bundle["metrics"]
    assert "Random Forest" in bundle["metrics"]
    assert "Decision Tree" in bundle["metrics"]

    test_date_in = datetime.date(2026, 9, 15)
    res_pune = estimate_rainfall_probability(
        district="PUNE",
        target_date=test_date_in,
        df_all=df,
        model_bundle=bundle,
        threshold=DEFAULT_RAINY_THRESHOLD_MM
    )

    assert res_pune["is_within_coverage"] is True
    assert res_pune["has_direct_observation"] is True
    assert res_pune["observed_rain_mm"] is not None
    assert "rf_probability_pct" in res_pune
    assert "lr_probability_pct" in res_pune
    assert "dt_probability_pct" in res_pune
    assert "avg_daily_rain_period_mm" in res_pune
    assert "historical_frequency_pct" in res_pune
    assert "total_rain_period_mm" in res_pune

    # Validate valid probability ranges [0, 100]
    for model_key in ["rf_probability_pct", "lr_probability_pct", "dt_probability_pct"]:
        prob = res_pune[model_key]
        assert prob is not None, f"{model_key} is None"
        assert 0.0 <= prob <= 100.0, f"{model_key} {prob} is out of range [0, 100]"

    print(f"Pune on {test_date_in}:")
    print(f"  Random Forest Probability:     {res_pune['rf_probability_pct']:.1f}%")
    print(f"  Logistic Regression Probability:{res_pune['lr_probability_pct']:.1f}%")
    print(f"  Decision Tree Probability:      {res_pune['dt_probability_pct']:.1f}%")
    print(f"  Average Daily Rain: {res_pune['avg_daily_rain_period_mm']:.2f} mm")
    print(f"  Rainy-Day Frequency: {res_pune['historical_frequency_pct']:.1f}%")
    print(f"  Total Period Rain: {res_pune['total_rain_period_mm']:.1f} mm")
    print(f"  Actual Observation: {res_pune['observed_rain_mm']:.1f} mm")

    print("\n--- 3. Testing Date Selection (Outside Coverage) ---")
    test_date_out = datetime.date(2026, 7, 15)
    res_pune_out = estimate_rainfall_probability(
        district="PUNE",
        target_date=test_date_out,
        df_all=df,
        model_bundle=bundle,
        threshold=DEFAULT_RAINY_THRESHOLD_MM
    )
    assert res_pune_out["is_within_coverage"] is False
    assert res_pune_out["has_direct_observation"] is False
    assert "rf_probability_pct" in res_pune_out
    assert "lr_probability_pct" in res_pune_out
    assert "dt_probability_pct" in res_pune_out
    for model_key in ["rf_probability_pct", "lr_probability_pct", "dt_probability_pct"]:
        prob = res_pune_out[model_key]
        assert prob is not None, f"{model_key} is None"
        assert 0.0 <= prob <= 100.0, f"{model_key} {prob} is out of range [0, 100]"

    print(f"Pune on {test_date_out} (outside coverage):")
    print(f"  RF: {res_pune_out['rf_probability_pct']:.1f}%, LR: {res_pune_out['lr_probability_pct']:.1f}%, DT: {res_pune_out['dt_probability_pct']:.1f}%")
    print(f"  Within Coverage Flag: {res_pune_out['is_within_coverage']}")
    print(f"  Interpretation: {res_pune_out['interpretation'][:100]}...")

    print("\n--- 4. Testing Different Districts ---")
    for dist in ["MUMBAI CITY", "NAGPUR"]:
        res_dist = estimate_rainfall_probability(
            district=dist,
            target_date=test_date_in,
            df_all=df,
            model_bundle=bundle,
            threshold=DEFAULT_RAINY_THRESHOLD_MM
        )
        print(f"{dist} on {test_date_in}: RF = {res_dist['rf_probability_pct']:.1f}%, LR = {res_dist['lr_probability_pct']:.1f}%, DT = {res_dist['dt_probability_pct']:.1f}%, Hist Freq = {res_dist['historical_frequency_pct']:.1f}%")

    print("\n--- 5. Testing the Exactly Two Visualizations ---")
    pune_df = filter_by_location(df, "PUNE")
    fig1 = plot_monthly_rainfall_pattern(pune_df, highlight_month=test_date_in.month)
    assert fig1 is not None

    fig2 = plot_daily_rainfall_trend(pune_df, highlight_date=test_date_in, threshold=DEFAULT_RAINY_THRESHOLD_MM)
    assert fig2 is not None

    # Test outside coverage date for Chart 2 (must not invent value)
    fig2_out = plot_daily_rainfall_trend(pune_df, highlight_date=test_date_out, threshold=DEFAULT_RAINY_THRESHOLD_MM)
    assert fig2_out is not None

    print("Charts 1 and 2 generated cleanly without error.")
    print("\n=== ALL SIMPLIFIED DASHBOARD TESTS PASSED! ===")

if __name__ == "__main__":
    run_tests()
