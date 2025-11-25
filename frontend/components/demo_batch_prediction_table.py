"""
Demo script for Batch Prediction Table Component

This script demonstrates the batch prediction table component with sample data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from frontend.components.batch_prediction_table import (
    create_batch_prediction_table,
    create_batch_prediction_interface,
    prepare_batch_table_data,
    apply_filters
)


def generate_sample_predictions(n_samples=50):
    """
    Generate sample prediction data for demonstration.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    
    Returns
    -------
    pd.DataFrame
        Sample predictions DataFrame
    """
    np.random.seed(42)
    
    # Generate random predictions
    prob_win_raw = np.random.beta(5, 3, n_samples)  # Skewed towards higher probabilities
    prob_win_calibrated = prob_win_raw * 0.9 + np.random.normal(0, 0.05, n_samples)
    prob_win_calibrated = np.clip(prob_win_calibrated, 0, 1)
    
    R_P50_raw = np.random.normal(1.0, 0.8, n_samples)
    R_P10_raw = R_P50_raw - np.abs(np.random.normal(0.5, 0.3, n_samples))
    R_P90_raw = R_P50_raw + np.abs(np.random.normal(1.0, 0.5, n_samples))
    
    # Conformal adjustment (slightly wider intervals)
    margin = np.random.uniform(0.1, 0.3, n_samples)
    R_P10_conf = R_P10_raw - margin
    R_P90_conf = R_P90_raw + margin
    
    # Categorize quality
    quality_labels = []
    recommendations = []
    
    for prob, r50 in zip(prob_win_calibrated, R_P50_raw):
        if prob > 0.65 and r50 > 1.5:
            quality = 'A+'
            rec = 'TRADE'
        elif prob > 0.55 and r50 > 1.0:
            quality = 'A'
            rec = 'TRADE'
        elif prob > 0.45 and r50 > 0.5:
            quality = 'B'
            rec = 'SKIP'
        else:
            quality = 'C'
            rec = 'SKIP'
        
        quality_labels.append(quality)
        recommendations.append(rec)
    
    # Create DataFrame
    df = pd.DataFrame({
        'setup_id': range(1, n_samples + 1),
        'prob_win_raw': prob_win_raw,
        'prob_win_calibrated': prob_win_calibrated,
        'R_P10_raw': R_P10_raw,
        'R_P50_raw': R_P50_raw,
        'R_P90_raw': R_P90_raw,
        'R_P10_conf': R_P10_conf,
        'R_P90_conf': R_P90_conf,
        'quality_label': quality_labels,
        'recommendation': recommendations
    })
    
    return df


def main():
    """
    Run demo application.
    """
    # Generate sample data
    predictions_df = generate_sample_predictions(n_samples=100)
    
    print("=" * 80)
    print("Batch Prediction Table Component Demo")
    print("=" * 80)
    print(f"\nGenerated {len(predictions_df)} sample predictions")
    print(f"\nQuality distribution:")
    print(predictions_df['quality_label'].value_counts())
    print(f"\nRecommendation distribution:")
    print(predictions_df['recommendation'].value_counts())
    
    # Test prepare_batch_table_data
    print("\n" + "=" * 80)
    print("Testing prepare_batch_table_data()")
    print("=" * 80)
    table_df = prepare_batch_table_data(predictions_df)
    print(f"\nTable columns: {list(table_df.columns)}")
    print(f"\nFirst 5 rows:")
    print(table_df.head())
    
    # Test apply_filters
    print("\n" + "=" * 80)
    print("Testing apply_filters()")
    print("=" * 80)
    
    # Filter by quality A+ and A only
    filtered_quality = apply_filters(table_df, quality_filter=['A+', 'A'])
    print(f"\nFiltered by quality ['A+', 'A']: {len(filtered_quality)} rows")
    
    # Filter by probability threshold
    filtered_prob = apply_filters(table_df, prob_win_threshold=60.0)
    print(f"Filtered by prob_win >= 60%: {len(filtered_prob)} rows")
    
    # Combined filters
    filtered_combined = apply_filters(
        table_df,
        quality_filter=['A+', 'A'],
        prob_win_threshold=60.0
    )
    print(f"Combined filters: {len(filtered_combined)} rows")
    
    # Create Dash app
    print("\n" + "=" * 80)
    print("Creating Dash Application")
    print("=" * 80)
    
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
    
    app.layout = dbc.Container([
        html.H1("Batch Prediction Table Demo", className="my-4"),
        
        dbc.Tabs([
            dbc.Tab(label="Full Interface", children=[
                html.Div(className="my-4", children=[
                    create_batch_prediction_interface(
                        predictions_df=predictions_df,
                        quality_filter=['A+', 'A'],
                        prob_win_threshold=50.0,
                        page_size=20
                    )
                ])
            ]),
            
            dbc.Tab(label="Table Only", children=[
                html.Div(className="my-4", children=[
                    create_batch_prediction_table(
                        predictions_df=predictions_df,
                        quality_filter=None,
                        prob_win_threshold=None,
                        page_size=20
                    )
                ])
            ]),
            
            dbc.Tab(label="Filtered (A+/A only)", children=[
                html.Div(className="my-4", children=[
                    create_batch_prediction_table(
                        predictions_df=predictions_df,
                        quality_filter=['A+', 'A'],
                        prob_win_threshold=None,
                        page_size=20
                    )
                ])
            ]),
            
            dbc.Tab(label="High Probability (>60%)", children=[
                html.Div(className="my-4", children=[
                    create_batch_prediction_table(
                        predictions_df=predictions_df,
                        quality_filter=None,
                        prob_win_threshold=60.0,
                        page_size=20
                    )
                ])
            ]),
            
            dbc.Tab(label="Empty State", children=[
                html.Div(className="my-4", children=[
                    create_batch_prediction_table(
                        predictions_df=None,
                        quality_filter=None,
                        prob_win_threshold=None,
                        page_size=20
                    )
                ])
            ])
        ])
    ], fluid=True)
    
    print("\nStarting Dash server...")
    print("Open http://127.0.0.1:8050 in your browser")
    print("Press Ctrl+C to stop the server")
    
    app.run_server(debug=True, port=8050)


if __name__ == '__main__':
    main()
