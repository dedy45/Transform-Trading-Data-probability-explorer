"""
Regime Comparison Table Component
Comprehensive table comparing all regimes side-by-side
"""
import dash_bootstrap_components as dbc
from dash import html, dash_table
import pandas as pd
import numpy as np


def create_regime_comparison_table(regime_comparison_df):
    """
    Create comprehensive comparison table with win rate, mean R, and sample size per regime
    
    Parameters:
    -----------
    regime_comparison_df : pd.DataFrame
        Results from create_regime_comparison_table with columns:
        - regime: Regime identifier
        - n_trades: Number of trades
        - win_rate: Win rate (proportion)
        - win_rate_ci_lower: Lower CI for win rate
        - win_rate_ci_upper: Upper CI for win rate
        - mean_r: Mean R-multiple
        - median_r: Median R-multiple
        - std_r: Standard deviation of R-multiple
        - reliable: Boolean for sample size adequacy
    
    Returns:
    --------
    dash_bootstrap_components or html.Div
        Interactive table with regime comparison
    
    Features:
    ---------
    - All key metrics per regime
    - Sortable columns
    - Color coding by performance
    - Reliability indicators
    - Formatted numbers with proper precision
    """
    if regime_comparison_df is None or regime_comparison_df.empty:
        return create_empty_regime_comparison_table()
    
    # Prepare data for display
    df = regime_comparison_df.copy()
    
    # Format columns for display
    table_data = []
    for _, row in df.iterrows():
        table_data.append({
            'Regime': str(row['regime']),
            'Trades': int(row['n_trades']),
            'Win Rate': f"{row['win_rate']:.1%}",
            'Win Rate CI': f"[{row['win_rate_ci_lower']:.1%}, {row['win_rate_ci_upper']:.1%}]",
            'Mean R': f"{row['mean_r']:.2f}",
            'Median R': f"{row['median_r']:.2f}",
            'Std R': f"{row['std_r']:.2f}",
            'Reliable': 'Yes' if row['reliable'] else 'No',
            # Hidden columns for conditional formatting
            '_win_rate_val': row['win_rate'],
            '_mean_r_val': row['mean_r'],
            '_reliable': row['reliable']
        })
    
    display_df = pd.DataFrame(table_data)
    
    # Create DataTable
    table = dash_table.DataTable(
        id='regime-comparison-detail-table',
        columns=[
            {'name': 'Regime', 'id': 'Regime', 'type': 'text'},
            {'name': 'Trades', 'id': 'Trades', 'type': 'numeric'},
            {'name': 'Win Rate', 'id': 'Win Rate', 'type': 'text'},
            {'name': 'Win Rate CI', 'id': 'Win Rate CI', 'type': 'text'},
            {'name': 'Mean R', 'id': 'Mean R', 'type': 'text'},
            {'name': 'Median R', 'id': 'Median R', 'type': 'text'},
            {'name': 'Std R', 'id': 'Std R', 'type': 'text'},
            {'name': 'Reliable', 'id': 'Reliable', 'type': 'text'}
        ],
        data=display_df.to_dict('records'),
        sort_action='native',
        sort_mode='single',
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '12px',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '13px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'border': '1px solid rgb(200, 200, 200)',
            'fontSize': '14px'
        },
        style_data={
            'border': '1px solid rgb(220, 220, 220)'
        },
        style_data_conditional=[
            # Highlight high win rate (>= 55%)
            {
                'if': {
                    'filter_query': '{_win_rate_val} >= 0.55',
                    'column_id': 'Win Rate'
                },
                'backgroundColor': 'rgba(50, 180, 50, 0.2)',
                'color': 'rgb(0, 100, 0)',
                'fontWeight': 'bold'
            },
            # Highlight low win rate (< 45%)
            {
                'if': {
                    'filter_query': '{_win_rate_val} < 0.45',
                    'column_id': 'Win Rate'
                },
                'backgroundColor': 'rgba(220, 50, 50, 0.2)',
                'color': 'rgb(150, 0, 0)',
                'fontWeight': 'bold'
            },
            # Highlight positive mean R (>= 0.5)
            {
                'if': {
                    'filter_query': '{_mean_r_val} >= 0.5',
                    'column_id': 'Mean R'
                },
                'backgroundColor': 'rgba(50, 180, 50, 0.15)',
                'color': 'rgb(0, 100, 0)',
                'fontWeight': 'bold'
            },
            # Highlight negative mean R (< 0)
            {
                'if': {
                    'filter_query': '{_mean_r_val} < 0',
                    'column_id': 'Mean R'
                },
                'backgroundColor': 'rgba(220, 50, 50, 0.15)',
                'color': 'rgb(150, 0, 0)',
                'fontWeight': 'bold'
            },
            # Highlight unreliable data
            {
                'if': {
                    'filter_query': '{Reliable} = "No"',
                    'column_id': 'Reliable'
                },
                'backgroundColor': 'rgba(255, 193, 7, 0.2)',
                'color': 'rgb(150, 100, 0)',
                'fontWeight': 'bold'
            },
            # Highlight reliable data
            {
                'if': {
                    'filter_query': '{Reliable} = "Yes"',
                    'column_id': 'Reliable'
                },
                'backgroundColor': 'rgba(70, 130, 180, 0.15)',
                'color': 'rgb(0, 70, 130)',
                'fontWeight': 'bold'
            },
            # Highlight regime column
            {
                'if': {'column_id': 'Regime'},
                'backgroundColor': 'rgba(240, 240, 240, 0.5)',
                'fontWeight': 'bold'
            }
        ],
        page_size=20,
        page_action='native'
    )
    
    return html.Div([
        table,
        html.Div([
            html.P([
                html.Strong("Table Guide: "),
                html.Span("Win Rate ≥ 55% ", className="text-success"),
                html.Span("and "),
                html.Span("Mean R ≥ 0.5 ", className="text-success"),
                html.Span("indicate strong regimes. "),
                html.Span("Win Rate < 45% ", className="text-danger"),
                html.Span("or "),
                html.Span("Mean R < 0 ", className="text-danger"),
                html.Span("indicate weak regimes. "),
                html.Span("Unreliable ", className="text-warning"),
                html.Span("means insufficient sample size.")
            ], className="small text-muted mb-0 mt-2")
        ])
    ])


def create_empty_regime_comparison_table():
    """Create empty comparison table placeholder"""
    return html.Div([
        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            "Calculate regime analysis to view comprehensive comparison table"
        ], color="info", className="mb-0")
    ])
