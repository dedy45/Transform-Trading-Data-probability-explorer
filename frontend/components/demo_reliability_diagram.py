"""
Demo script for Reliability Diagram Component

This script demonstrates the reliability diagram component with sample data.
"""

import numpy as np
import dash
from dash import html
import dash_bootstrap_components as dbc
from probability_analysis_section import (
    create_probability_analysis_section,
    create_empty_probability_analysis_section
)

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 1000

# Scenario 1: Well-calibrated probabilities
print("Generating well-calibrated sample data...")
raw_probs_good = np.random.beta(2, 2, n_samples)  # Centered distribution
true_labels_good = (np.random.random(n_samples) < raw_probs_good).astype(int)

# Add some calibration improvement
calibrated_probs_good = raw_probs_good * 0.95 + 0.025  # Slight adjustment

# Scenario 2: Poorly calibrated probabilities (overconfident)
print("Generating poorly-calibrated sample data...")
raw_probs_poor = np.random.beta(0.5, 0.5, n_samples)  # U-shaped distribution
true_labels_poor = (np.random.random(n_samples) < 0.5).astype(int)  # Random labels

# Scenario 3: Underconfident probabilities
print("Generating underconfident sample data...")
raw_probs_under = np.random.beta(5, 5, n_samples)  # Narrow distribution around 0.5
true_labels_under = (np.random.random(n_samples) < raw_probs_under * 1.2).astype(int)

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Reliability Diagram Component Demo", className="my-4"),
    
    html.Hr(),
    
    # Scenario 1: Well-calibrated
    html.H3("Scenario 1: Well-Calibrated Model", className="mt-4 mb-3"),
    html.P(
        "This model has good calibration with Brier score < 0.20 and ECE < 0.05.",
        className="text-muted"
    ),
    create_probability_analysis_section(
        predicted_probs=raw_probs_good,
        true_labels=true_labels_good,
        calibrated_probs=calibrated_probs_good,
        n_bins=10
    ),
    
    html.Hr(className="my-5"),
    
    # Scenario 2: Poorly calibrated
    html.H3("Scenario 2: Poorly-Calibrated Model (Overconfident)", className="mt-4 mb-3"),
    html.P(
        "This model is overconfident with poor calibration metrics.",
        className="text-muted"
    ),
    create_probability_analysis_section(
        predicted_probs=raw_probs_poor,
        true_labels=true_labels_poor,
        calibrated_probs=None,
        n_bins=10
    ),
    
    html.Hr(className="my-5"),
    
    # Scenario 3: Underconfident
    html.H3("Scenario 3: Underconfident Model", className="mt-4 mb-3"),
    html.P(
        "This model is underconfident with predictions clustered around 0.5.",
        className="text-muted"
    ),
    create_probability_analysis_section(
        predicted_probs=raw_probs_under,
        true_labels=true_labels_under,
        calibrated_probs=None,
        n_bins=10
    ),
    
    html.Hr(className="my-5"),
    
    # Scenario 4: No data
    html.H3("Scenario 4: No Data State", className="mt-4 mb-3"),
    html.P(
        "This shows the component when no prediction data is available.",
        className="text-muted"
    ),
    create_empty_probability_analysis_section(),
    
], fluid=True, className="py-4")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Reliability Diagram Component Demo")
    print("="*60)
    print("\nStarting Dash server...")
    print("Open your browser and navigate to: http://127.0.0.1:8050")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run_server(debug=True, port=8050)
