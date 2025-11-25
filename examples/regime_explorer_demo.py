"""
Regime Explorer Demo
Demonstrates the Regime Explorer frontend components
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from dash import Dash, html
import dash_bootstrap_components as dbc

# Import backend functions
from backend.calculators.regime_analysis import (
    compute_regime_probabilities,
    compute_regime_threshold_probs,
    create_regime_comparison_table
)

# Import frontend components
from frontend.components.regime_winrate_chart import create_regime_winrate_chart
from frontend.components.regime_r_multiple_chart import create_regime_r_multiple_chart
from frontend.components.regime_comparison_table import create_regime_comparison_table as create_regime_table_component
from frontend.components.regime_filter_controls import (
    create_regime_filter_controls,
    create_regime_summary_cards
)

# Import layout
from frontend.layouts.regime_explorer_layout import create_regime_explorer_layout


def generate_sample_data(n_trades=500):
    """Generate sample trade data with regime information"""
    np.random.seed(42)
    
    # Create regimes
    trend_regimes = np.random.choice([0, 1], size=n_trades, p=[0.4, 0.6])  # 0=ranging, 1=trending
    volatility_regimes = np.random.choice([0, 1, 2], size=n_trades, p=[0.3, 0.5, 0.2])  # 0=low, 1=med, 2=high
    
    # Generate outcomes based on regime
    # Trending regime has better win rate
    win_probs = np.where(trend_regimes == 1, 0.58, 0.45)
    trade_success = np.random.binomial(1, win_probs)
    
    # Generate R-multiples
    # Winners have positive R, losers have negative R
    r_multiples = np.where(
        trade_success == 1,
        np.random.gamma(2, 0.8, n_trades),  # Winners
        -np.random.gamma(1.5, 0.5, n_trades)  # Losers
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'Timestamp': pd.date_range('2024-01-01', periods=n_trades, freq='4h'),
        'trend_regime': trend_regimes,
        'volatility_regime': volatility_regimes,
        'trade_success': trade_success,
        'R_multiple': r_multiples,
        'y_hit_1R': (r_multiples >= 1.0).astype(int),
        'y_hit_2R': (r_multiples >= 2.0).astype(int)
    })
    
    return df


def main():
    """Run demo application"""
    print("=" * 60)
    print("Regime Explorer Demo")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample trade data...")
    df = generate_sample_data(n_trades=500)
    print(f"   Generated {len(df)} trades")
    print(f"   Trend regimes: {df['trend_regime'].unique()}")
    print(f"   Volatility regimes: {df['volatility_regime'].unique()}")
    
    # Calculate regime probabilities
    print("\n2. Calculating regime probabilities...")
    regime_probs = compute_regime_probabilities(
        df=df,
        regime_column='trend_regime',
        target_column='trade_success',
        conf_level=0.95,
        min_samples=5
    )
    print("\nRegime Probabilities:")
    print(regime_probs)
    
    # Calculate regime threshold probabilities
    print("\n3. Calculating regime threshold probabilities...")
    regime_thresholds = compute_regime_threshold_probs(
        df=df,
        regime_column='trend_regime',
        r_column='R_multiple',
        thresholds=[1.0, 2.0],
        conf_level=0.95,
        min_samples=5
    )
    print("\nRegime Threshold Probabilities:")
    print(regime_thresholds)
    
    # Create regime comparison table
    print("\n4. Creating regime comparison table...")
    regime_comparison = create_regime_comparison_table(
        df=df,
        regime_column='trend_regime',
        target_column='trade_success',
        r_column='R_multiple',
        conf_level=0.95,
        min_samples=5
    )
    print("\nRegime Comparison:")
    print(regime_comparison)
    
    # Create Dash app
    print("\n5. Creating Dash application...")
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
    
    # Create components
    winrate_chart = create_regime_winrate_chart(regime_probs)
    r_multiple_chart = create_regime_r_multiple_chart(regime_thresholds)
    comparison_table = create_regime_table_component(regime_comparison)
    summary_cards = create_regime_summary_cards(regime_comparison)
    filter_controls = create_regime_filter_controls(available_regimes=[0, 1])
    
    # Create layout
    app.layout = dbc.Container([
        html.H1("Regime Explorer Demo", className="my-4"),
        
        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            "This is a demo showing the Regime Explorer components with sample data. "
            "In the full application, this would be integrated with real trade data and interactive callbacks."
        ], color="info", className="mb-4"),
        
        # Summary Cards
        html.H3("Summary Cards", className="mt-4 mb-3"),
        summary_cards,
        
        # Filter Controls
        html.H3("Filter Controls", className="mt-4 mb-3"),
        filter_controls,
        
        # Charts
        html.H3("Win Rate Chart", className="mt-4 mb-3"),
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Div([winrate_chart], style={'height': '500px'})
                ])
            ])
        ], className="mb-4"),
        
        html.H3("R-Multiple Threshold Chart", className="mt-4 mb-3"),
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Div([r_multiple_chart], style={'height': '500px'})
                ])
            ])
        ], className="mb-4"),
        
        # Comparison Table
        html.H3("Comparison Table", className="mt-4 mb-3"),
        dbc.Card([
            dbc.CardBody([
                comparison_table
            ])
        ], className="mb-4"),
        
        # Full Layout
        html.H3("Complete Regime Explorer Layout", className="mt-5 mb-3"),
        dbc.Alert([
            html.I(className="bi bi-eye me-2"),
            "Below is the complete Regime Explorer layout as it would appear in the main application."
        ], color="secondary", className="mb-3"),
        create_regime_explorer_layout()
        
    ], fluid=True)
    
    print("\n6. Starting Dash server...")
    print("\n" + "=" * 60)
    print("Demo application running at: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    app.run(debug=True, port=8050)


if __name__ == '__main__':
    main()
