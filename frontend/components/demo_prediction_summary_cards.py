"""
Demo script for Prediction Summary Cards Component

This script demonstrates the prediction summary cards with sample data.
"""
import sys
sys.path.append('.')

from dash import Dash, html
import dash_bootstrap_components as dbc
from frontend.components.prediction_summary_cards import (
    create_prediction_summary_cards,
    create_empty_prediction_summary_cards
)


def main():
    """Run demo app"""
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
    
    app.layout = dbc.Container([
        html.H2("Prediction Summary Cards Demo", className="my-4"),
        
        # Example 1: Excellent Setup (A+)
        html.H4("Example 1: Excellent Setup (A+)", className="mt-4 mb-3"),
        create_prediction_summary_cards(
            prob_win_raw=0.68,
            prob_win_calibrated=0.72,
            R_P10_raw=-0.3,
            R_P50_raw=1.8,
            R_P90_raw=3.5,
            R_P10_conf=-0.5,
            R_P90_conf=3.8,
            quality_label='A+',
            recommendation='TRADE',
            skewness=1.35
        ),
        
        html.Hr(className="my-4"),
        
        # Example 2: Good Setup (A)
        html.H4("Example 2: Good Setup (A)", className="mt-4 mb-3"),
        create_prediction_summary_cards(
            prob_win_raw=0.58,
            prob_win_calibrated=0.60,
            R_P10_raw=-0.5,
            R_P50_raw=1.2,
            R_P90_raw=2.5,
            R_P10_conf=-0.7,
            R_P90_conf=2.8,
            quality_label='A',
            recommendation='TRADE',
            skewness=1.08
        ),
        
        html.Hr(className="my-4"),
        
        # Example 3: Fair Setup (B)
        html.H4("Example 3: Fair Setup (B)", className="mt-4 mb-3"),
        create_prediction_summary_cards(
            prob_win_raw=0.48,
            prob_win_calibrated=0.50,
            R_P10_raw=-0.8,
            R_P50_raw=0.7,
            R_P90_raw=1.8,
            R_P10_conf=-1.0,
            R_P90_conf=2.0,
            quality_label='B',
            recommendation='SKIP',
            skewness=0.92
        ),
        
        html.Hr(className="my-4"),
        
        # Example 4: Poor Setup (C)
        html.H4("Example 4: Poor Setup (C)", className="mt-4 mb-3"),
        create_prediction_summary_cards(
            prob_win_raw=0.38,
            prob_win_calibrated=0.42,
            R_P10_raw=-1.2,
            R_P50_raw=0.2,
            R_P90_raw=1.2,
            R_P10_conf=-1.5,
            R_P90_conf=1.5,
            quality_label='C',
            recommendation='SKIP',
            skewness=0.71
        ),
        
        html.Hr(className="my-4"),
        
        # Example 5: Empty State
        html.H4("Example 5: Empty State (No Prediction)", className="mt-4 mb-3"),
        create_empty_prediction_summary_cards(),
        
        html.Hr(className="my-4"),
        
        # Example 6: Negative Expected R
        html.H4("Example 6: Negative Expected R", className="mt-4 mb-3"),
        create_prediction_summary_cards(
            prob_win_raw=0.35,
            prob_win_calibrated=0.38,
            R_P10_raw=-2.0,
            R_P50_raw=-0.5,
            R_P90_raw=0.8,
            R_P10_conf=-2.3,
            R_P90_conf=1.0,
            quality_label='C',
            recommendation='SKIP',
            skewness=0.65
        ),
        
    ], fluid=True, className="py-4")
    
    print("\n" + "="*60)
    print("Prediction Summary Cards Demo")
    print("="*60)
    print("\nDemo app running at: http://127.0.0.1:8050")
    print("\nThis demo shows:")
    print("  1. Prob Win card with gauge chart and color gradient")
    print("  2. Expected R card with direction icon (↑/↓)")
    print("  3. Interval R card with bar visualization [P10_conf, P90_conf]")
    print("  4. Setup Quality card with color-coded badge (A+/A/B/C)")
    print("  5. Recommendation card with TRADE/SKIP label")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, port=8050)


if __name__ == '__main__':
    main()
