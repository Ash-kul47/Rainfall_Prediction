"""Interactive Plotly Visualizations for Rainfall Pattern Analysis and Prediction.

Generates clean, minimalistic, and responsive Plotly charts adhering to modern
aesthetic standards: curated color palettes, elegant typography, clear units, and
informative tooltips.
"""

import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Design Palette - Restrained & Clean
PRIMARY_COLOR = "#0284C7"     # Sky / Cyan Blue
HIGHLIGHT_COLOR = "#2563EB"   # Deep Royal Blue
ACCENT_RED = "#EF4444"        # Crimson Highlight
MUTED_SLATE = "#CBD5E1"       # Light Slate
DARK_SLATE = "#334155"        # Text Slate
THRESHOLD_RED = "#DC2626"     # Threshold Line

PLOTLY_TEMPLATE = "plotly_white"


def format_fig(fig: go.Figure, title: str, x_title: str = "", y_title: str = "") -> go.Figure:
    """Apply consistent, professional styling to a Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title={
            "text": f"<b>{title}</b>",
            "y": 0.95,
            "x": 0.03,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 15, "family": "Inter, Roboto, sans-serif", "color": "#0F172A"}
        },
        margin=dict(l=45, r=30, t=50, b=40),
        xaxis_title=x_title,
        yaxis_title=y_title,
        font=dict(family="Inter, Roboto, sans-serif", size=12, color="#334155"),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_family="Inter, Roboto, sans-serif",
            font_color="#FFFFFF"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig


def plot_monthly_rainfall_pattern(
    df_loc: pd.DataFrame,
    highlight_month: Optional[int] = None
) -> go.Figure:
    """Chart 1: Monthly rainfall pattern with the selected date's month highlighted."""
    month_data = []
    month_names = {8: "August", 9: "September", 10: "October"}

    for m in sorted(df_loc['month'].unique()):
        sub = df_loc[df_loc['month'] == m]
        m_name = month_names.get(m, f"Month {m}")
        total_mm = sub['daily_actual'].sum()
        avg_mm = sub['daily_actual'].mean()
        days = len(sub)
        rainy_days = (sub['daily_actual'] >= 2.5).sum()
        is_selected = (highlight_month is not None and m == highlight_month)

        month_data.append({
            "month": m,
            "month_name": m_name,
            "total_rainfall_mm": round(total_mm, 2),
            "avg_daily_mm": round(avg_mm, 2),
            "rainy_days": rainy_days,
            "days": days,
            "is_selected": is_selected
        })

    m_df = pd.DataFrame(month_data)

    # Dynamic colors: Highlight the selected month
    bar_colors = [
        HIGHLIGHT_COLOR if r['is_selected'] else MUTED_SLATE
        for _, r in m_df.iterrows()
    ]
    border_colors = [
        "#1D4ED8" if r['is_selected'] else "#94A3B8"
        for _, r in m_df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=m_df['month_name'],
        y=m_df['total_rainfall_mm'],
        text=m_df['total_rainfall_mm'].apply(lambda v: f"{v:.1f} mm"),
        textposition="outside",
        marker=dict(
            color=bar_colors,
            line=dict(color=border_colors, width=1.5),
            cornerradius=4
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Total Rainfall: %{y:.1f} mm<br>"
            "Daily Average: %{customdata[0]:.2f} mm/day<br>"
            "Observed Days: %{customdata[1]} days<br>"
            "Rainy Days: %{customdata[2]} days<extra></extra>"
        ),
        customdata=m_df[['avg_daily_mm', 'days', 'rainy_days']].values
    ))

    max_val = m_df['total_rainfall_mm'].max() if len(m_df) > 0 else 10
    fig.update_layout(
        yaxis=dict(title="Total Rainfall (mm)", range=[0, max_val * 1.25]),
        xaxis=dict(title="Observed Month"),
        height=350
    )

    title = "Monthly Rainfall Pattern"
    if highlight_month in month_names:
        title += f" (Highlighting {month_names[highlight_month]})"

    return format_fig(fig, title, "Observed Month", "Total Rainfall (mm)")


def plot_daily_rainfall_trend(
    df_loc: pd.DataFrame,
    highlight_date: Optional[datetime.date] = None,
    threshold: float = 2.5
) -> go.Figure:
    """Chart 2: Historical daily rainfall trend with selected date highlighted if observed."""
    df_sorted = df_loc.sort_values(by='date').copy()

    # Determine bar colors
    bar_colors = []
    has_match = False
    match_date_str = None
    match_rain_val = 0.0

    for _, r in df_sorted.iterrows():
        r_date = r['date'].date()
        if highlight_date is not None and r_date == highlight_date:
            bar_colors.append(ACCENT_RED)
            has_match = True
            match_date_str = r['date'].strftime('%d-%b-%Y')
            match_rain_val = float(r['daily_actual'])
        else:
            bar_colors.append(PRIMARY_COLOR if r['daily_actual'] >= threshold else MUTED_SLATE)

    fig = go.Figure()

    # Daily Actual Rainfall Bars
    hover_texts = []
    for _, r in df_sorted.iterrows():
        d_str = r['date'].strftime('%d-%b-%Y')
        act = r['daily_actual']
        cat = r['daily_category_desc'] if 'daily_category_desc' in r else ""
        hover_texts.append(f"<b>{d_str}</b><br>Rainfall: {act:.1f} mm<br>Category: {cat}")

    fig.add_trace(go.Bar(
        x=df_sorted['date'],
        y=df_sorted['daily_actual'],
        name="Daily Rainfall (mm)",
        marker=dict(color=bar_colors, cornerradius=2),
        hovertext=hover_texts,
        hoverinfo="text"
    ))

    # Reference line for Rainy Threshold
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=THRESHOLD_RED,
        line_width=1.2,
        annotation_text=f"Rainy threshold ({threshold} mm)",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color=THRESHOLD_RED
    )

    # If the selected date was matched, add a clean annotation
    if has_match and match_date_str:
        fig.add_annotation(
            x=highlight_date,
            y=match_rain_val,
            text=f"Selected: {match_rain_val:.1f} mm",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=ACCENT_RED,
            ax=0,
            ay=-35,
            font=dict(size=11, color=ACCENT_RED, family="Inter, Roboto, sans-serif")
        )

    title = "Historical Daily Rainfall Trend"
    if has_match and match_date_str:
        title += f" (Selected: {match_date_str})"
    elif highlight_date is not None:
        title += f" (Selected: {highlight_date.strftime('%d-%b-%Y')} outside dataset)"

    fig.update_layout(
        xaxis=dict(title="Observation Date", type="date"),
        yaxis=dict(title="Daily Rainfall (mm)"),
        showlegend=False,
        height=350
    )
    return format_fig(fig, title, "Observation Date", "Daily Rainfall (mm)")


# Legacy wrappers to maintain backward compatibility
def plot_monthly_rainfall(df_loc: pd.DataFrame, metric: str = "total") -> go.Figure:
    return plot_monthly_rainfall_pattern(df_loc)

def plot_yearly_rainfall_trend(df_loc: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = go.Figure()
    return format_fig(fig, "Yearly Trend")

def plot_monthly_rainfall_heatmap(df_loc: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    return format_fig(fig, "Heatmap")

def plot_rainy_vs_non_rainy(df_loc: pd.DataFrame, threshold: float = 2.5) -> go.Figure:
    fig = go.Figure()
    return format_fig(fig, "Rainy vs Non-Rainy")

def plot_daily_rainfall_timeline(df_loc: pd.DataFrame, threshold: float = 2.5) -> go.Figure:
    return plot_daily_rainfall_trend(df_loc, threshold=threshold)

def plot_model_comparison(metrics_dict: Dict[str, Dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    return format_fig(fig, "Model Comparison")

def plot_rainfall_probability_by_period(prob_list: List[Dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    return format_fig(fig, "Probability by Period")
