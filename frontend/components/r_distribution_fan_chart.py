"""
R Distribution Fan Chart Component

This component creates visualizations for R_multiple distribution analysis:
1. Fan chart showing P10-P90 range with P50 center line
2. Overlay of actual R_multiple values as scatter points
3. Coverage percentage calculation and display
4. Skewness metric with interpretation
5. Comparison histogram (predicted vs actual distribution)
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def calculate_coverage(y_true, y_pred_p10, y_pred_p90):
    """
    Calculate coverage percentage (how many actual values fall within predicted interval).
    
    Parameters
    ----------
    y_true : array-like
        Actual R_multiple values
    y_pred_p10 : array-like
        Predicted P10 values
    y_pred_p90 : array-like
        Predicted P90 values
    
    Returns
    -------
    float
        Coverage percentage (0-100)
    """
    y_true = np.array(y_true)
    y_pred_p10 = np.array(y_pred_p10)
    y_pred_p90 = np.array(y_pred_p90)
    
    # Check if actual values fall within interval
    within_interval = (y_true >= y_pred_p10) & (y_true <= y_pred_p90)
    coverage = np.mean(within_interval) * 100
    
    return coverage


def calculate_skewness(y_pred_p10, y_pred_p50, y_pred_p90):
    """
    Calculate skewness of predicted distribution.
    
    Skewness = (P90 - P50) / (P50 - P10)
    - Positive skew (> 1): upside potential
    - Symmetric (≈ 1): balanced distribution
    - Negative skew (< 1): downside risk
    
    Parameters
    ----------
    y_pred_p10 : array-like
        Predicted P10 values
    y_pred_p50 : array-like
        Predicted P50 values
    y_pred_p90 : array-like
        Predicted P90 values
    
    Returns
    -------
    float
        Mean skewness across all samples
    """
    y_pred_p10 = np.array(y_pred_p10)
    y_pred_p50 = np.array(y_pred_p50)
    y_pred_p90 = np.array(y_pred_p90)
    
    # Calculate skewness per sample
    denominator = y_pred_p50 - y_pred_p10
    # Avoid division by zero
    denominator = np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)
    
    skewness = (y_pred_p90 - y_pred_p50) / denominator
    
    # Return mean skewness
    return np.mean(skewness)


def interpret_skewness(skewness):
    """
    Interpret skewness value.
    
    Parameters
    ----------
    skewness : float
        Skewness value
    
    Returns
    -------
    str
        Interpretation text
    """
    if skewness > 1.5:
        return "Strong upside potential - asymmetric gains likely"
    elif skewness > 1.1:
        return "Moderate upside potential - slightly positive skew"
    elif skewness > 0.9:
        return "Balanced distribution - symmetric outcomes"
    elif skewness > 0.5:
        return "Moderate downside risk - slightly negative skew"
    else:
        return "Strong downside risk - asymmetric losses likely"


def create_fan_chart(y_pred_p10, y_pred_p50, y_pred_p90, y_true=None, sample_indices=None):
    """
    Create fan chart showing P10-P90 range with P50 center line.
    
    Parameters
    ----------
    y_pred_p10 : array-like
        Predicted P10 values
    y_pred_p50 : array-like
        Predicted P50 values
    y_pred_p90 : array-like
        Predicted P90 values
    y_true : array-like, optional
        Actual R_multiple values to overlay
    sample_indices : array-like, optional
        Sample indices for x-axis (default: 0, 1, 2, ...)
    
    Returns
    -------
    plotly.graph_objects.Figure
        Fan chart figure
    """
    y_pred_p10 = np.array(y_pred_p10)
    y_pred_p50 = np.array(y_pred_p50)
    y_pred_p90 = np.array(y_pred_p90)
    
    if sample_indices is None:
        sample_indices = np.arange(len(y_pred_p10))
    
    fig = go.Figure()
    
    # Add P10-P90 shaded area
    fig.add_trace(go.Scatter(
        x=np.concatenate([sample_indices, sample_indices[::-1]]),
        y=np.concatenate([y_pred_p90, y_pred_p10[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 100, 200, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='P10-P90 Range',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Add P90 line
    fig.add_trace(go.Scatter(
        x=sample_indices,
        y=y_pred_p90,
        mode='lines',
        name='P90 (Upper)',
        line=dict(color='rgba(0, 100, 200, 0.5)', width=1, dash='dash'),
        showlegend=True,
        hovertemplate='Sample: %{x}<br>P90: %{y:.2f}R<extra></extra>'
    ))
    
    # Add P50 center line
    fig.add_trace(go.Scatter(
        x=sample_indices,
        y=y_pred_p50,
        mode='lines',
        name='P50 (Median)',
        line=dict(color='blue', width=2),
        showlegend=True,
        hovertemplate='Sample: %{x}<br>P50: %{y:.2f}R<extra></extra>'
    ))
    
    # Add P10 line
    fig.add_trace(go.Scatter(
        x=sample_indices,
        y=y_pred_p10,
        mode='lines',
        name='P10 (Lower)',
        line=dict(color='rgba(0, 100, 200, 0.5)', width=1, dash='dash'),
        showlegend=True,
        hovertemplate='Sample: %{x}<br>P10: %{y:.2f}R<extra></extra>'
    ))
    
    # Overlay actual R_multiple if provided
    if y_true is not None:
        y_true = np.array(y_true)
        
        # Determine color based on whether within interval
        within_interval = (y_true >= y_pred_p10) & (y_true <= y_pred_p90)
        colors = ['green' if w else 'red' for w in within_interval]
        
        fig.add_trace(go.Scatter(
            x=sample_indices,
            y=y_true,
            mode='markers',
            name='Actual R',
            marker=dict(
                size=6,
                color=colors,
                symbol='circle',
                line=dict(color='white', width=1)
            ),
            showlegend=True,
            hovertemplate='Sample: %{x}<br>Actual: %{y:.2f}R<extra></extra>'
        ))
    
    fig.update_layout(
        title='R_multiple Distribution Fan Chart',
        xaxis_title='Sample Index',
        yaxis_title='R_multiple',
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig


def create_comparison_histogram(y_pred_p50, y_true):
    """
    Create comparison histogram of predicted vs actual distribution.
    
    Parameters
    ----------
    y_pred_p50 : array-like
        Predicted P50 (median) values
    y_true : array-like
        Actual R_multiple values
    
    Returns
    -------
    plotly.graph_objects.Figure
        Histogram comparison figure
    """
    y_pred_p50 = np.array(y_pred_p50)
    y_true = np.array(y_true)
    
    fig = go.Figure()
    
    # Add predicted distribution histogram
    fig.add_trace(go.Histogram(
        x=y_pred_p50,
        name='Predicted (P50)',
        opacity=0.6,
        marker=dict(color='blue'),
        nbinsx=30,
        histnorm='probability density'
    ))
    
    # Add actual distribution histogram
    fig.add_trace(go.Histogram(
        x=y_true,
        name='Actual',
        opacity=0.6,
        marker=dict(color='green'),
        nbinsx=30,
        histnorm='probability density'
    ))
    
    fig.update_layout(
        title='Predicted vs Actual Distribution',
        xaxis_title='R_multiple',
        yaxis_title='Probability Density',
        barmode='overlay',
        height=400,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        template='plotly_white'
    )
    
    return fig


def create_distribution_analysis_section(predictions_dict):
    """
    Create complete distribution analysis section with all visualizations.
    
    Parameters
    ----------
    predictions_dict : dict
        Dictionary containing:
        - 'R_P10_conf': array of P10 conformal predictions
        - 'R_P50_raw': array of P50 predictions
        - 'R_P90_conf': array of P90 conformal predictions
        - 'R_actual': array of actual R_multiple values (optional)
        - 'sample_indices': array of sample indices (optional)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'fan_chart': plotly Figure
        - 'histogram': plotly Figure
        - 'coverage': float (percentage)
        - 'skewness': float
        - 'skewness_interpretation': str
    """
    y_pred_p10 = predictions_dict['R_P10_conf']
    y_pred_p50 = predictions_dict['R_P50_raw']
    y_pred_p90 = predictions_dict['R_P90_conf']
    y_true = predictions_dict.get('R_actual', None)
    sample_indices = predictions_dict.get('sample_indices', None)
    
    # Calculate metrics
    skewness = calculate_skewness(y_pred_p10, y_pred_p50, y_pred_p90)
    skewness_interpretation = interpret_skewness(skewness)
    
    coverage = None
    if y_true is not None:
        coverage = calculate_coverage(y_true, y_pred_p10, y_pred_p90)
    
    # Create visualizations
    fan_chart = create_fan_chart(
        y_pred_p10, y_pred_p50, y_pred_p90, 
        y_true=y_true, 
        sample_indices=sample_indices
    )
    
    histogram = None
    if y_true is not None:
        histogram = create_comparison_histogram(y_pred_p50, y_true)
    
    return {
        'fan_chart': fan_chart,
        'histogram': histogram,
        'coverage': coverage,
        'skewness': skewness,
        'skewness_interpretation': skewness_interpretation
    }


def create_empty_fan_chart():
    """
    Create empty fan chart (no data state).
    
    Returns
    -------
    plotly.graph_objects.Figure
        Empty fan chart figure
    """
    fig = go.Figure()
    
    # Add placeholder text
    fig.add_annotation(
        text="No distribution data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    
    fig.update_layout(
        title='R_multiple Distribution Fan Chart',
        xaxis_title='Sample Index',
        yaxis_title='R_multiple',
        height=500,
        template='plotly_white'
    )
    
    return fig


def create_empty_histogram():
    """
    Create empty histogram (no data state).
    
    Returns
    -------
    plotly.graph_objects.Figure
        Empty histogram figure
    """
    fig = go.Figure()
    
    # Add placeholder text
    fig.add_annotation(
        text="No distribution data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    
    fig.update_layout(
        title='Predicted vs Actual Distribution',
        xaxis_title='R_multiple',
        yaxis_title='Probability Density',
        height=400,
        template='plotly_white'
    )
    
    return fig
