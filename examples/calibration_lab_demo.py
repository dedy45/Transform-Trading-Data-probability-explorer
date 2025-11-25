"""
Calibration Lab Demo
Demonstrates the Calibration Lab frontend components
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# Import backend calibration functions
from backend.models.calibration import (
    compute_reliability_diagram,
    compute_brier_score,
    compute_ece
)

# Import frontend components
from frontend.components.reliability_diagram import create_reliability_diagram
from frontend.components.calibration_histogram import create_calibration_histogram
from frontend.components.calibration_metrics_cards import create_calibration_metrics_cards
from frontend.components.calibration_table import create_calibration_table

# Import layout
from frontend.layouts.calibration_lab_layout import create_calibration_lab_layout


def generate_sample_data(n_samples=1000, calibration_quality='good'):
    """
    Generate sample probability predictions and outcomes
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    calibration_quality : str
        'perfect', 'good', 'fair', 'poor', 'overconfident', 'underconfident'
    
    Returns:
    --------
    predicted_probs : np.ndarray
        Predicted probabilities
    actual_outcomes : np.ndarray
        Actual binary outcomes
    """
    np.random.seed(42)
    
    if calibration_quality == 'perfect':
        # Perfect calibration: predicted = actual probability
        predicted_probs = np.random.uniform(0.1, 0.9, n_samples)
        actual_outcomes = (np.random.random(n_samples) < predicted_probs).astype(int)
    
    elif calibration_quality == 'good':
        # Good calibration with small noise
        predicted_probs = np.random.uniform(0.2, 0.8, n_samples)
        true_probs = predicted_probs + np.random.normal(0, 0.05, n_samples)
        true_probs = np.clip(true_probs, 0, 1)
        actual_outcomes = (np.random.random(n_samples) < true_probs).astype(int)
    
    elif calibration_quality == 'fair':
        # Fair calibration with moderate noise
        predicted_probs = np.random.uniform(0.2, 0.8, n_samples)
        true_probs = predicted_probs + np.random.normal(0, 0.1, n_samples)
        true_probs = np.clip(true_probs, 0, 1)
        actual_outcomes = (np.random.random(n_samples) < true_probs).astype(int)
    
    elif calibration_quality == 'poor':
        # Poor calibration with large noise
        predicted_probs = np.random.uniform(0.2, 0.8, n_samples)
        true_probs = predicted_probs + np.random.normal(0, 0.2, n_samples)
        true_probs = np.clip(true_probs, 0, 1)
        actual_outcomes = (np.random.random(n_samples) < true_probs).astype(int)
    
    elif calibration_quality == 'overconfident':
        # Overconfident: predicts higher than actual
        predicted_probs = np.random.uniform(0.4, 0.9, n_samples)
        true_probs = predicted_probs - 0.2
        true_probs = np.clip(true_probs, 0, 1)
        actual_outcomes = (np.random.random(n_samples) < true_probs).astype(int)
    
    elif calibration_quality == 'underconfident':
        # Underconfident: predicts lower than actual
        predicted_probs = np.random.uniform(0.1, 0.6, n_samples)
        true_probs = predicted_probs + 0.2
        true_probs = np.clip(true_probs, 0, 1)
        actual_outcomes = (np.random.random(n_samples) < true_probs).astype(int)
    
    else:
        raise ValueError(f"Unknown calibration_quality: {calibration_quality}")
    
    return predicted_probs, actual_outcomes


# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])

# Create layout
app.layout = dbc.Container([
    html.H1("Calibration Lab Demo", className="mb-4 mt-3"),
    
    # Demo controls
    dbc.Card([
        dbc.CardHeader(html.H5("Demo Data Generator", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Calibration Quality", className="fw-bold"),
                    dcc.Dropdown(
                        id='demo-quality-dropdown',
                        options=[
                            {'label': 'Perfect Calibration', 'value': 'perfect'},
                            {'label': 'Good Calibration', 'value': 'good'},
                            {'label': 'Fair Calibration', 'value': 'fair'},
                            {'label': 'Poor Calibration', 'value': 'poor'},
                            {'label': 'Overconfident Model', 'value': 'overconfident'},
                            {'label': 'Underconfident Model', 'value': 'underconfident'}
                        ],
                        value='good',
                        clearable=False
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Number of Samples", className="fw-bold"),
                    dcc.Slider(
                        id='demo-n-samples-slider',
                        min=100,
                        max=5000,
                        step=100,
                        value=1000,
                        marks={100: '100', 1000: '1K', 2500: '2.5K', 5000: '5K'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Number of Bins", className="fw-bold"),
                    dcc.Slider(
                        id='demo-n-bins-slider',
                        min=5,
                        max=20,
                        step=1,
                        value=10,
                        marks={5: '5', 10: '10', 15: '15', 20: '20'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=4)
            ], className="mb-3"),
            dbc.Button(
                [html.I(className="bi bi-play-fill me-2"), "Generate & Analyze"],
                id="demo-generate-btn",
                color="primary",
                size="lg"
            )
        ])
    ], className="mb-4"),
    
    # Calibration metrics cards
    html.Div(id='demo-metrics-cards'),
    
    # Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Reliability Diagram", className="mb-0")),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-demo-reliability",
                        children=[dcc.Graph(id='demo-reliability-diagram')]
                    )
                ])
            ])
        ], md=7),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Probability Distribution", className="mb-0")),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-demo-histogram",
                        children=[dcc.Graph(id='demo-histogram')]
                    )
                ])
            ])
        ], md=5)
    ], className="mb-3"),
    
    # Calibration table
    dbc.Card([
        dbc.CardHeader(html.H5("Calibration Details by Bin", className="mb-0")),
        dbc.CardBody([
            dcc.Loading(
                id="loading-demo-table",
                children=[html.Div(id='demo-calibration-table')]
            )
        ])
    ], className="mb-4")
    
], fluid=True)


@app.callback(
    [
        Output('demo-metrics-cards', 'children'),
        Output('demo-reliability-diagram', 'figure'),
        Output('demo-histogram', 'figure'),
        Output('demo-calibration-table', 'children')
    ],
    Input('demo-generate-btn', 'n_clicks'),
    [
        State('demo-quality-dropdown', 'value'),
        State('demo-n-samples-slider', 'value'),
        State('demo-n-bins-slider', 'value')
    ],
    prevent_initial_call=False
)
def update_calibration_demo(n_clicks, quality, n_samples, n_bins):
    """Update all calibration visualizations"""
    
    # Generate sample data
    predicted_probs, actual_outcomes = generate_sample_data(
        n_samples=int(n_samples),
        calibration_quality=quality
    )
    
    # Calculate calibration metrics
    reliability_data = compute_reliability_diagram(
        predicted_probs,
        actual_outcomes,
        n_bins=n_bins,
        strategy='quantile'
    )
    
    brier_score = compute_brier_score(predicted_probs, actual_outcomes)
    ece = compute_ece(predicted_probs, actual_outcomes, n_bins=n_bins, strategy='quantile')
    
    # Create components
    metrics_cards = create_calibration_metrics_cards(
        brier_score=brier_score,
        ece=ece,
        n_samples=len(predicted_probs),
        n_bins=n_bins
    )
    
    reliability_fig = create_reliability_diagram(reliability_data, conf_level=0.95)
    histogram_fig = create_calibration_histogram(predicted_probs, n_bins=20)
    calibration_table = create_calibration_table(reliability_data)
    
    return metrics_cards, reliability_fig, histogram_fig, calibration_table


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Calibration Lab Demo")
    print("="*60)
    print("\nStarting Dash server...")
    print("Open your browser to: http://127.0.0.1:8050")
    print("\nFeatures:")
    print("- Reliability diagram (calibration plot)")
    print("- Probability distribution histogram")
    print("- Calibration metrics (Brier Score, ECE)")
    print("- Detailed per-bin calibration table")
    print("\nTry different calibration qualities to see how metrics change!")
    print("="*60 + "\n")
    
    app.run(debug=True, port=8050)
