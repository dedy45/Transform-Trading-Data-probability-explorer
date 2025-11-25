"""
Calibration Table Component
Detailed table showing calibration statistics per bin
"""
import dash_bootstrap_components as dbc
from dash import html, dash_table
import pandas as pd
import numpy as np


def create_calibration_table(reliability_data):
    """
    Create detailed calibration table showing per-bin statistics
    
    Parameters:
    -----------
    reliability_data : dict
        Results from compute_reliability_diagram with keys:
        - bin_edges: Array of bin edges
        - mean_predicted: Mean predicted probability per bin
        - observed_frequency: Observed frequency per bin
        - n_samples: Number of samples per bin
        - bin_centers: Center point of each bin
    
    Returns:
    --------
    dash_bootstrap_components.Table or html.Div
        Interactive table with calibration details per bin
    
    Features:
    ---------
    - Bin range and center
    - Mean predicted probability
    - Observed frequency
    - Deviation (absolute difference)
    - Sample count
    - Calibration quality indicator
    - Sortable columns
    - Color coding by quality
    """
    if reliability_data is None or len(reliability_data.get('mean_predicted', [])) == 0:
        return html.Div([
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                "No calibration data available. Calculate calibration to view detailed table."
            ], color="info", className="mb-0")
        ])
    
    bin_edges = reliability_data['bin_edges']
    mean_predicted = reliability_data['mean_predicted']
    observed_frequency = reliability_data['observed_frequency']
    n_samples = reliability_data['n_samples']
    bin_centers = reliability_data['bin_centers']
    
    # Create DataFrame
    table_data = []
    for i in range(len(mean_predicted)):
        if not np.isnan(mean_predicted[i]):
            deviation = abs(mean_predicted[i] - observed_frequency[i])
            
            # Determine quality
            if deviation < 0.05:
                quality = "Excellent"
                quality_color = "success"
            elif deviation < 0.10:
                quality = "Good"
                quality_color = "info"
            elif deviation < 0.15:
                quality = "Fair"
                quality_color = "warning"
            else:
                quality = "Poor"
                quality_color = "danger"
            
            table_data.append({
                'Bin': i + 1,
                'Range': f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]",
                'Center': f"{bin_centers[i]:.3f}",
                'Mean Predicted': f"{mean_predicted[i]:.1%}",
                'Observed Freq': f"{observed_frequency[i]:.1%}",
                'Deviation': f"{deviation:.1%}",
                'Samples': int(n_samples[i]),
                'Quality': quality,
                'Quality_Color': quality_color
            })
    
    if len(table_data) == 0:
        return html.Div([
            dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                "No valid calibration bins. Adjust settings or check data."
            ], color="warning", className="mb-0")
        ])
    
    df = pd.DataFrame(table_data)
    
    # Create DataTable
    table = dash_table.DataTable(
        id='calibration-detail-table',
        columns=[
            {'name': 'Bin', 'id': 'Bin', 'type': 'numeric'},
            {'name': 'Range', 'id': 'Range', 'type': 'text'},
            {'name': 'Center', 'id': 'Center', 'type': 'numeric'},
            {'name': 'Mean Predicted', 'id': 'Mean Predicted', 'type': 'text'},
            {'name': 'Observed Freq', 'id': 'Observed Freq', 'type': 'text'},
            {'name': 'Deviation', 'id': 'Deviation', 'type': 'text'},
            {'name': 'Samples', 'id': 'Samples', 'type': 'numeric'},
            {'name': 'Quality', 'id': 'Quality', 'type': 'text'}
        ],
        data=df.to_dict('records'),
        sort_action='native',
        sort_mode='single',
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '13px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'border': '1px solid rgb(200, 200, 200)'
        },
        style_data={
            'border': '1px solid rgb(220, 220, 220)'
        },
        style_data_conditional=[
            # Color code by quality
            {
                'if': {
                    'filter_query': '{Quality} = "Excellent"',
                    'column_id': 'Quality'
                },
                'backgroundColor': 'rgba(50, 180, 50, 0.2)',
                'color': 'rgb(0, 100, 0)',
                'fontWeight': 'bold'
            },
            {
                'if': {
                    'filter_query': '{Quality} = "Good"',
                    'column_id': 'Quality'
                },
                'backgroundColor': 'rgba(70, 130, 180, 0.2)',
                'color': 'rgb(0, 70, 130)',
                'fontWeight': 'bold'
            },
            {
                'if': {
                    'filter_query': '{Quality} = "Fair"',
                    'column_id': 'Quality'
                },
                'backgroundColor': 'rgba(255, 193, 7, 0.2)',
                'color': 'rgb(150, 100, 0)',
                'fontWeight': 'bold'
            },
            {
                'if': {
                    'filter_query': '{Quality} = "Poor"',
                    'column_id': 'Quality'
                },
                'backgroundColor': 'rgba(220, 50, 50, 0.2)',
                'color': 'rgb(150, 0, 0)',
                'fontWeight': 'bold'
            },
            # Highlight high deviation
            {
                'if': {
                    'filter_query': '{Deviation} > "15%"',
                    'column_id': 'Deviation'
                },
                'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                'fontWeight': 'bold'
            }
        ],
        page_size=15,
        page_action='native'
    )
    
    return html.Div([
        table,
        html.Div([
            html.P([
                html.Strong("Table Guide: "),
                html.Span("Bins with deviation < 5% are excellent. ", className="text-success"),
                html.Span("Deviation 5-10% is good. ", className="text-info"),
                html.Span("Deviation 10-15% is fair. ", className="text-warning"),
                html.Span("Deviation > 15% indicates poor calibration.", className="text-danger")
            ], className="small text-muted mb-0 mt-2")
        ])
    ])


def create_empty_calibration_table():
    """Create empty calibration table placeholder"""
    return html.Div([
        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            "Calculate calibration to view detailed bin statistics"
        ], color="info", className="mb-0")
    ])
