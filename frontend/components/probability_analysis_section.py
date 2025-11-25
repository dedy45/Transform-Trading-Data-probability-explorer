"""
Reliability Diagram Component for Probability Analysis

This component creates visualizations for assessing probability calibration:
1. Reliability diagram (mean predicted vs observed frequency)
2. Perfect calibration reference line (y=x)
3. Brier score display with interpretation
4. Expected Calibration Error (ECE) with threshold
5. Histogram of predicted probabilities

**Feature: ml-prediction-engine**
**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
"""

import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Tuple

# Import calibration utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.models.calibration import (
    compute_reliability_diagram,
    compute_brier_score,
    compute_ece
)


def interpret_brier_score(brier_score: float) -> Tuple[str, str]:
    """
    Interpret Brier score with quality assessment.
    
    Parameters
    ----------
    brier_score : float
        Brier score value (0-1, lower is better)
    
    Returns
    -------
    tuple
        (interpretation_text, badge_color)
    """
    if brier_score < 0.10:
        return "Excellent calibration", "success"
    elif brier_score < 0.20:
        return "Good calibration", "success"
    elif brier_score < 0.30:
        return "Fair calibration", "warning"
    else:
        return "Poor calibration", "danger"


def interpret_ece(ece: float) -> Tuple[str, str]:
    """
    Interpret Expected Calibration Error with quality assessment.
    
    Parameters
    ----------
    ece : float
        ECE value (0-1, lower is better)
    
    Returns
    -------
    tuple
        (interpretation_text, badge_color)
    """
    if ece < 0.05:
        return "Excellent calibration", "success"
    elif ece < 0.10:
        return "Good calibration", "success"
    elif ece < 0.15:
        return "Fair calibration", "warning"
    else:
        return "Poor calibration", "danger"


def create_reliability_plot(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 10,
    show_histogram: bool = True
) -> go.Figure:
    """
    Create reliability diagram with optional histogram.
    
    Parameters
    ----------
    predicted_probs : np.ndarray
        Predicted probabilities (0-1)
    true_labels : np.ndarray
        True binary labels (0/1)
    n_bins : int, default=10
        Number of bins for reliability diagram
    show_histogram : bool, default=True
        Whether to show histogram subplot
    
    Returns
    -------
    plotly.graph_objects.Figure
        Reliability diagram figure
    """
    # Compute reliability diagram data
    reliability_data = compute_reliability_diagram(
        predicted_probs, true_labels, n_bins=n_bins
    )
    
    # Filter out NaN values
    mask = ~np.isnan(reliability_data['mean_predicted'])
    mean_predicted = reliability_data['mean_predicted'][mask]
    observed_freq = reliability_data['observed_frequency'][mask]
    counts = reliability_data['n_samples'][mask]
    
    # Create subplots if histogram is requested
    if show_histogram:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Reliability Diagram', 'Distribution of Predicted Probabilities'),
            vertical_spacing=0.12
        )
    else:
        fig = go.Figure()
    
    # Add perfect calibration line (y=x)
    if show_histogram:
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='gray', dash='dash', width=2),
                showlegend=True,
                hovertemplate='Perfect Calibration<br>x=y<extra></extra>'
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='gray', dash='dash', width=2),
                showlegend=True,
                hovertemplate='Perfect Calibration<br>x=y<extra></extra>'
            )
        )
    
    # Add actual calibration curve
    if show_histogram:
        fig.add_trace(
            go.Scatter(
                x=mean_predicted,
                y=observed_freq,
                mode='markers+lines',
                name='Actual Calibration',
                marker=dict(
                    size=12,
                    color=counts,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title='Sample<br>Count',
                        x=1.02,
                        len=0.5
                    ),
                    line=dict(color='white', width=1)
                ),
                line=dict(color='blue', width=2),
                showlegend=True,
                hovertemplate=(
                    '<b>Bin</b><br>'
                    'Mean Predicted: %{x:.3f}<br>'
                    'Observed Freq: %{y:.3f}<br>'
                    'Count: %{marker.color}<br>'
                    '<extra></extra>'
                )
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=mean_predicted,
                y=observed_freq,
                mode='markers+lines',
                name='Actual Calibration',
                marker=dict(
                    size=12,
                    color=counts,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title='Sample<br>Count',
                        x=1.15,
                        len=0.7
                    ),
                    line=dict(color='white', width=1)
                ),
                line=dict(color='blue', width=2),
                showlegend=True,
                hovertemplate=(
                    '<b>Bin</b><br>'
                    'Mean Predicted: %{x:.3f}<br>'
                    'Observed Freq: %{y:.3f}<br>'
                    'Count: %{marker.color}<br>'
                    '<extra></extra>'
                )
            )
        )
    
    # Update reliability diagram axes
    if show_histogram:
        fig.update_xaxes(
            title_text='Mean Predicted Probability',
            range=[0, 1],
            constrain='domain',
            row=1, col=1
        )
        fig.update_yaxes(
            title_text='Observed Frequency',
            range=[0, 1],
            scaleanchor='x',
            scaleratio=1,
            row=1, col=1
        )
    else:
        fig.update_xaxes(
            title_text='Mean Predicted Probability',
            range=[0, 1],
            constrain='domain'
        )
        fig.update_yaxes(
            title_text='Observed Frequency',
            range=[0, 1],
            scaleanchor='x',
            scaleratio=1
        )
    
    # Add histogram if requested
    if show_histogram:
        fig.add_trace(
            go.Histogram(
                x=predicted_probs,
                nbinsx=n_bins,
                name='Probability Distribution',
                marker=dict(
                    color='steelblue',
                    line=dict(color='white', width=1)
                ),
                showlegend=False,
                hovertemplate=(
                    '<b>Probability Range</b><br>'
                    '%{x}<br>'
                    'Count: %{y}<br>'
                    '<extra></extra>'
                )
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(
            title_text='Predicted Probability',
            range=[0, 1],
            row=2, col=1
        )
        fig.update_yaxes(
            title_text='Count',
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=700 if show_histogram else 500,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        hovermode='closest',
        template='plotly_white',
        margin=dict(l=60, r=120, t=80, b=60)
    )
    
    return fig


def create_probability_histogram(
    predicted_probs: np.ndarray,
    n_bins: int = 20
) -> go.Figure:
    """
    Create standalone histogram of predicted probabilities.
    
    Parameters
    ----------
    predicted_probs : np.ndarray
        Predicted probabilities (0-1)
    n_bins : int, default=20
        Number of bins for histogram
    
    Returns
    -------
    plotly.graph_objects.Figure
        Histogram figure
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=predicted_probs,
            nbinsx=n_bins,
            marker=dict(
                color='steelblue',
                line=dict(color='white', width=1)
            ),
            hovertemplate=(
                '<b>Probability Range</b><br>'
                '%{x}<br>'
                'Count: %{y}<br>'
                '<extra></extra>'
            )
        )
    )
    
    # Add mean line
    mean_prob = np.mean(predicted_probs)
    fig.add_vline(
        x=mean_prob,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_prob:.3f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title='Distribution of Predicted Probabilities',
        xaxis_title='Predicted Probability',
        yaxis_title='Count',
        xaxis=dict(range=[0, 1]),
        height=300,
        template='plotly_white',
        showlegend=False,
        margin=dict(l=60, r=60, t=60, b=60)
    )
    
    return fig


def create_probability_analysis_section(
    predicted_probs: Optional[np.ndarray] = None,
    true_labels: Optional[np.ndarray] = None,
    calibrated_probs: Optional[np.ndarray] = None,
    n_bins: int = 10
) -> dbc.Container:
    """
    Create complete probability analysis section with all components.
    
    This section includes:
    1. Reliability diagram with perfect calibration line
    2. Brier score with interpretation
    3. ECE with threshold assessment
    4. Histogram of predicted probabilities
    
    Parameters
    ----------
    predicted_probs : np.ndarray, optional
        Raw predicted probabilities (0-1)
    true_labels : np.ndarray, optional
        True binary labels (0/1)
    calibrated_probs : np.ndarray, optional
        Calibrated probabilities (0-1)
    n_bins : int, default=10
        Number of bins for reliability diagram
    
    Returns
    -------
    dash_bootstrap_components.Container
        Complete probability analysis section
    """
    # Check if data is available
    has_data = (
        predicted_probs is not None and
        true_labels is not None and
        len(predicted_probs) > 0 and
        len(true_labels) > 0
    )
    
    if has_data:
        # Use calibrated probs if available, otherwise use raw
        probs_to_analyze = (
            calibrated_probs if calibrated_probs is not None
            else predicted_probs
        )
        
        # Compute metrics
        brier_score = compute_brier_score(probs_to_analyze, true_labels)
        ece = compute_ece(probs_to_analyze, true_labels, n_bins=n_bins)
        
        # Get interpretations
        brier_interp, brier_color = interpret_brier_score(brier_score)
        ece_interp, ece_color = interpret_ece(ece)
        
        # Create reliability plot
        reliability_fig = create_reliability_plot(
            probs_to_analyze,
            true_labels,
            n_bins=n_bins,
            show_histogram=True
        )
        
        # Metrics cards
        metrics_cards = dbc.Row([
            # Brier Score Card
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6([
                            html.I(className="bi bi-bullseye me-2"),
                            "Brier Score"
                        ], className="text-muted mb-2"),
                        html.H2(f"{brier_score:.4f}", className="mb-2"),
                        dbc.Badge(
                            brier_interp,
                            color=brier_color,
                            className="mb-2"
                        ),
                        html.Hr(),
                        html.P([
                            html.Strong("Interpretation: "),
                            "Measures accuracy of probabilistic predictions. "
                            "Lower is better. "
                        ], className="small mb-1"),
                        html.P([
                            html.Strong("Threshold: "),
                            html.Span("< 0.20 = Good", className="text-success"),
                            " | ",
                            html.Span("< 0.30 = Fair", className="text-warning")
                        ], className="small mb-0")
                    ])
                ], className="h-100 shadow-sm")
            ], md=6, className="mb-3"),
            
            # ECE Card
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6([
                            html.I(className="bi bi-graph-up me-2"),
                            "Expected Calibration Error"
                        ], className="text-muted mb-2"),
                        html.H2(f"{ece:.4f}", className="mb-2"),
                        dbc.Badge(
                            ece_interp,
                            color=ece_color,
                            className="mb-2"
                        ),
                        html.Hr(),
                        html.P([
                            html.Strong("Interpretation: "),
                            "Measures calibration quality. "
                            "Lower is better. "
                        ], className="small mb-1"),
                        html.P([
                            html.Strong("Threshold: "),
                            html.Span("< 0.05 = Good", className="text-success"),
                            " | ",
                            html.Span("< 0.10 = Fair", className="text-warning")
                        ], className="small mb-0")
                    ])
                ], className="h-100 shadow-sm")
            ], md=6, className="mb-3")
        ])
        
        # Reliability diagram
        reliability_chart = dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="bi bi-graph-up-arrow me-2"),
                    "Reliability Diagram & Probability Distribution"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                dcc.Graph(
                    figure=reliability_fig,
                    config={'displayModeBar': True}
                ),
                html.Div([
                    html.P([
                        html.Strong("How to read: "),
                        "Points on the diagonal line indicate perfect calibration. "
                        "Points above the line mean the model is underconfident, "
                        "points below mean overconfident. "
                        "Marker size and color indicate the number of samples in each bin."
                    ], className="small text-muted mb-2"),
                    html.P([
                        html.Strong("Histogram: "),
                        "Shows the distribution of predicted probabilities. "
                        "A well-calibrated model should have predictions spread across "
                        "the full range [0, 1], not clustered in the middle."
                    ], className="small text-muted mb-0")
                ], className="mt-3 p-3 bg-light rounded")
            ])
        ], className="shadow-sm mb-3")
        
        content = [metrics_cards, reliability_chart]
        
    else:
        # No data state
        content = [
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                "No calibration data available. Train a model and run predictions to see probability analysis."
            ], color="info", className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-graph-up fs-1 text-muted mb-3"),
                        html.H5("Probability Analysis", className="text-muted mb-2"),
                        html.P(
                            "This section will display reliability diagrams, Brier score, "
                            "ECE, and probability distributions once you have prediction data.",
                            className="text-muted"
                        )
                    ], className="text-center py-5")
                ])
            ], className="shadow-sm")
        ]
    
    return dbc.Container([
        html.H4([
            html.I(className="bi bi-bar-chart-line me-2"),
            "Probability Analysis"
        ], className="mb-3"),
        html.P(
            "Assess the quality of probability calibration and prediction confidence.",
            className="text-muted mb-4"
        ),
        *content
    ], fluid=True, className="mb-4")


def create_empty_probability_analysis_section() -> dbc.Container:
    """
    Create empty probability analysis section (no data state).
    
    Returns
    -------
    dash_bootstrap_components.Container
        Empty probability analysis section
    """
    return create_probability_analysis_section(
        predicted_probs=None,
        true_labels=None,
        calibrated_probs=None
    )
