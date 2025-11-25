"""
Batch Prediction Table Component

This component creates a table interface for displaying multiple ML predictions:
1. Table with columns: setup_id, prob_win, R_P50, interval, quality, recommendation
2. Sorting by prob_win, R_P50, quality
3. Filtering by quality (A+/A only) or prob_win threshold
4. Export button for batch results to CSV
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import pandas as pd
import numpy as np


def format_interval_text(p10, p90):
    """
    Format interval as text string.
    
    Parameters
    ----------
    p10 : float
        Lower bound (P10)
    p90 : float
        Upper bound (P90)
    
    Returns
    -------
    str
        Formatted interval text
    """
    if pd.isna(p10) or pd.isna(p90):
        return "—"
    return f"[{p10:.2f}R, {p90:.2f}R]"


def prepare_batch_table_data(predictions_df):
    """
    Prepare batch predictions DataFrame for table display.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with prediction results containing:
        - setup_id or index
        - prob_win_calibrated
        - R_P50_raw
        - R_P10_conf
        - R_P90_conf
        - quality_label
        - recommendation
    
    Returns
    -------
    pd.DataFrame
        Formatted DataFrame for table display
    """
    # Create a copy to avoid modifying original
    df = predictions_df.copy()
    
    # Ensure setup_id column exists
    if 'setup_id' not in df.columns:
        df['setup_id'] = df.index
    
    # Format probability as percentage
    if 'prob_win_calibrated' in df.columns:
        df['prob_win_pct'] = (df['prob_win_calibrated'] * 100).round(1)
    else:
        df['prob_win_pct'] = np.nan
    
    # Format R_P50
    if 'R_P50_raw' in df.columns:
        df['expected_R'] = df['R_P50_raw'].round(2)
    else:
        df['expected_R'] = np.nan
    
    # Format interval
    if 'R_P10_conf' in df.columns and 'R_P90_conf' in df.columns:
        df['interval'] = df.apply(
            lambda row: format_interval_text(row['R_P10_conf'], row['R_P90_conf']),
            axis=1
        )
        df['interval_width'] = (df['R_P90_conf'] - df['R_P10_conf']).round(2)
    else:
        df['interval'] = "—"
        df['interval_width'] = np.nan
    
    # Ensure quality_label and recommendation exist
    if 'quality_label' not in df.columns:
        df['quality_label'] = "—"
    if 'recommendation' not in df.columns:
        df['recommendation'] = "—"
    
    # Select and order columns for display
    display_columns = [
        'setup_id',
        'prob_win_pct',
        'expected_R',
        'interval',
        'interval_width',
        'quality_label',
        'recommendation'
    ]
    
    # Only include columns that exist
    display_columns = [col for col in display_columns if col in df.columns]
    
    return df[display_columns]


def apply_filters(df, quality_filter=None, prob_win_threshold=None):
    """
    Apply filters to batch predictions DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Batch predictions DataFrame
    quality_filter : list or None
        List of quality labels to include (e.g., ['A+', 'A'])
    prob_win_threshold : float or None
        Minimum probability threshold (0-100)
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Apply quality filter
    if quality_filter is not None and len(quality_filter) > 0:
        if 'quality_label' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['quality_label'].isin(quality_filter)]
    
    # Apply probability threshold
    if prob_win_threshold is not None:
        if 'prob_win_pct' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['prob_win_pct'] >= prob_win_threshold]
    
    return filtered_df


def get_quality_color(quality):
    """
    Get color for quality label.
    
    Parameters
    ----------
    quality : str
        Quality label (A+/A/B/C)
    
    Returns
    -------
    str
        Color name or hex code
    """
    quality_colors = {
        'A+': '#006400',  # dark green
        'A': '#32CD32',   # green
        'B': '#FFD700',   # yellow
        'C': '#DC143C'    # red
    }
    return quality_colors.get(quality, '#6c757d')


def get_recommendation_color(recommendation):
    """
    Get color for recommendation.
    
    Parameters
    ----------
    recommendation : str
        Recommendation (TRADE/SKIP)
    
    Returns
    -------
    str
        Color name
    """
    if recommendation == 'TRADE':
        return 'success'
    elif recommendation == 'SKIP':
        return 'danger'
    else:
        return 'secondary'


def create_batch_prediction_table(
    predictions_df,
    quality_filter=None,
    prob_win_threshold=None,
    page_size=20,
    table_id='batch-prediction-table'
):
    """
    Create batch prediction table with sorting and filtering.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with prediction results
    quality_filter : list or None
        List of quality labels to include
    prob_win_threshold : float or None
        Minimum probability threshold (0-100)
    page_size : int
        Number of rows per page
    table_id : str
        HTML id for the table
    
    Returns
    -------
    dash_bootstrap_components.Container
        Container with table and controls
    """
    # Prepare data
    if predictions_df is None or len(predictions_df) == 0:
        return create_empty_batch_table(table_id)
    
    table_df = prepare_batch_table_data(predictions_df)
    
    # Apply filters
    filtered_df = apply_filters(table_df, quality_filter, prob_win_threshold)
    
    # Define column configurations
    columns = [
        {
            'name': 'Setup ID',
            'id': 'setup_id',
            'type': 'numeric',
            'format': {'specifier': 'd'}
        },
        {
            'name': 'Win Prob (%)',
            'id': 'prob_win_pct',
            'type': 'numeric',
            'format': {'specifier': '.1f'}
        },
        {
            'name': 'Expected R',
            'id': 'expected_R',
            'type': 'numeric',
            'format': {'specifier': '.2f'}
        },
        {
            'name': 'Interval',
            'id': 'interval',
            'type': 'text'
        },
        {
            'name': 'Width',
            'id': 'interval_width',
            'type': 'numeric',
            'format': {'specifier': '.2f'}
        },
        {
            'name': 'Quality',
            'id': 'quality_label',
            'type': 'text'
        },
        {
            'name': 'Recommendation',
            'id': 'recommendation',
            'type': 'text'
        }
    ]
    
    # Filter columns to only those present in data
    columns = [col for col in columns if col['id'] in filtered_df.columns]
    
    # Create conditional styling
    style_data_conditional = [
        # Quality label colors
        {
            'if': {
                'filter_query': '{quality_label} = "A+"',
                'column_id': 'quality_label'
            },
            'backgroundColor': '#006400',
            'color': 'white',
            'fontWeight': 'bold'
        },
        {
            'if': {
                'filter_query': '{quality_label} = "A"',
                'column_id': 'quality_label'
            },
            'backgroundColor': '#32CD32',
            'color': 'white',
            'fontWeight': 'bold'
        },
        {
            'if': {
                'filter_query': '{quality_label} = "B"',
                'column_id': 'quality_label'
            },
            'backgroundColor': '#FFD700',
            'color': 'black',
            'fontWeight': 'bold'
        },
        {
            'if': {
                'filter_query': '{quality_label} = "C"',
                'column_id': 'quality_label'
            },
            'backgroundColor': '#DC143C',
            'color': 'white',
            'fontWeight': 'bold'
        },
        # Recommendation colors
        {
            'if': {
                'filter_query': '{recommendation} = "TRADE"',
                'column_id': 'recommendation'
            },
            'backgroundColor': '#d4edda',
            'color': '#155724',
            'fontWeight': 'bold'
        },
        {
            'if': {
                'filter_query': '{recommendation} = "SKIP"',
                'column_id': 'recommendation'
            },
            'backgroundColor': '#f8d7da',
            'color': '#721c24',
            'fontWeight': 'bold'
        },
        # Highlight high probability
        {
            'if': {
                'filter_query': '{prob_win_pct} >= 65',
                'column_id': 'prob_win_pct'
            },
            'backgroundColor': '#d4edda',
            'color': '#155724'
        },
        # Highlight positive expected R
        {
            'if': {
                'filter_query': '{expected_R} >= 1.5',
                'column_id': 'expected_R'
            },
            'backgroundColor': '#d4edda',
            'color': '#155724'
        }
    ]
    
    # Create DataTable
    table = dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=filtered_df.to_dict('records'),
        sort_action='native',
        sort_mode='multi',
        filter_action='native',
        page_action='native',
        page_current=0,
        page_size=page_size,
        style_table={
            'overflowX': 'auto',
            'minWidth': '100%'
        },
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontSize': '14px',
            'fontFamily': 'Arial, sans-serif'
        },
        style_header={
            'backgroundColor': '#f8f9fa',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'border': '1px solid #dee2e6'
        },
        style_data={
            'border': '1px solid #dee2e6'
        },
        style_data_conditional=style_data_conditional,
        export_format='csv',
        export_headers='display'
    )
    
    # Create summary statistics
    total_count = len(filtered_df)
    trade_count = len(filtered_df[filtered_df['recommendation'] == 'TRADE']) if 'recommendation' in filtered_df.columns else 0
    skip_count = len(filtered_df[filtered_df['recommendation'] == 'SKIP']) if 'recommendation' in filtered_df.columns else 0
    
    avg_prob = filtered_df['prob_win_pct'].mean() if 'prob_win_pct' in filtered_df.columns and len(filtered_df) > 0 else 0
    avg_expected_r = filtered_df['expected_R'].mean() if 'expected_R' in filtered_df.columns and len(filtered_df) > 0 else 0
    
    quality_counts = filtered_df['quality_label'].value_counts().to_dict() if 'quality_label' in filtered_df.columns else {}
    
    summary_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Total Setups", className="text-muted mb-2"),
                    html.H3(f"{total_count}", className="mb-0")
                ])
            ], className="shadow-sm")
        ], md=6, lg=3, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("TRADE Recommendations", className="text-muted mb-2"),
                    html.H3([
                        html.Span(f"{trade_count}", className="text-success"),
                        html.Small(f" ({trade_count/total_count*100:.1f}%)" if total_count > 0 else " (0%)", className="text-muted ms-2")
                    ], className="mb-0")
                ])
            ], className="shadow-sm")
        ], md=6, lg=3, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Avg Win Prob", className="text-muted mb-2"),
                    html.H3(f"{avg_prob:.1f}%", className="mb-0")
                ])
            ], className="shadow-sm")
        ], md=6, lg=3, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Avg Expected R", className="text-muted mb-2"),
                    html.H3(f"{avg_expected_r:.2f}R", className="mb-0")
                ])
            ], className="shadow-sm")
        ], md=6, lg=3, className="mb-3")
    ])
    
    # Create quality distribution badges
    quality_badges = html.Div([
        html.H6("Quality Distribution:", className="text-muted mb-2"),
        html.Div([
            dbc.Badge(
                f"A+: {quality_counts.get('A+', 0)}",
                color="success",
                className="me-2 mb-2",
                style={'backgroundColor': '#006400'}
            ),
            dbc.Badge(
                f"A: {quality_counts.get('A', 0)}",
                color="success",
                className="me-2 mb-2"
            ),
            dbc.Badge(
                f"B: {quality_counts.get('B', 0)}",
                color="warning",
                className="me-2 mb-2"
            ),
            dbc.Badge(
                f"C: {quality_counts.get('C', 0)}",
                color="danger",
                className="me-2 mb-2"
            )
        ])
    ], className="mb-3")
    
    return dbc.Container([
        # Summary statistics
        summary_cards,
        
        # Quality distribution
        quality_badges,
        
        # Table
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="bi bi-table me-2"),
                    "Batch Predictions"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                table
            ])
        ], className="shadow-sm")
    ], fluid=True)


def create_empty_batch_table(table_id='batch-prediction-table'):
    """
    Create empty batch prediction table (no data state).
    
    Parameters
    ----------
    table_id : str
        HTML id for the table
    
    Returns
    -------
    dash_bootstrap_components.Container
        Container with empty state message
    """
    return dbc.Container([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="bi bi-table fs-1 text-muted mb-3"),
                    html.H5("No Batch Predictions Available", className="text-muted mb-2"),
                    html.P(
                        "Upload a CSV file or run batch prediction to see results here.",
                        className="text-muted mb-0"
                    )
                ], className="text-center py-5")
            ])
        ], className="shadow-sm")
    ], fluid=True)


def create_filter_controls(
    quality_options=None,
    prob_win_threshold=50.0,
    quality_filter_id='batch-quality-filter',
    prob_threshold_id='batch-prob-threshold'
):
    """
    Create filter controls for batch prediction table.
    
    Parameters
    ----------
    quality_options : list or None
        Available quality options (default: ['A+', 'A', 'B', 'C'])
    prob_win_threshold : float
        Initial probability threshold (0-100)
    quality_filter_id : str
        HTML id for quality filter
    prob_threshold_id : str
        HTML id for probability threshold
    
    Returns
    -------
    dash_bootstrap_components.Row
        Row with filter controls
    """
    if quality_options is None:
        quality_options = ['A+', 'A', 'B', 'C']
    
    return dbc.Row([
        dbc.Col([
            html.Label("Filter by Quality:", className="fw-bold mb-2"),
            dcc.Checklist(
                id=quality_filter_id,
                options=[
                    {'label': ' A+ (Excellent)', 'value': 'A+'},
                    {'label': ' A (Good)', 'value': 'A'},
                    {'label': ' B (Fair)', 'value': 'B'},
                    {'label': ' C (Poor)', 'value': 'C'}
                ],
                value=['A+', 'A'],  # Default: show only A+ and A
                inline=True,
                className="mb-3"
            )
        ], md=12, lg=6, className="mb-3"),
        
        dbc.Col([
            html.Label("Minimum Win Probability (%):", className="fw-bold mb-2"),
            dcc.Slider(
                id=prob_threshold_id,
                min=0,
                max=100,
                step=5,
                value=prob_win_threshold,
                marks={i: f'{i}%' for i in range(0, 101, 20)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], md=12, lg=6, className="mb-3")
    ])


def create_export_button(button_id='export-batch-predictions-btn'):
    """
    Create export button for batch predictions.
    
    Parameters
    ----------
    button_id : str
        HTML id for export button
    
    Returns
    -------
    dash_bootstrap_components.Button
        Export button
    """
    return dbc.Button([
        html.I(className="bi bi-download me-2"),
        "Export to CSV"
    ], id=button_id, color="primary", className="me-2")


def create_batch_prediction_interface(
    predictions_df=None,
    quality_filter=None,
    prob_win_threshold=50.0,
    page_size=20
):
    """
    Create complete batch prediction interface with table, filters, and export.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame or None
        DataFrame with prediction results
    quality_filter : list or None
        List of quality labels to include
    prob_win_threshold : float
        Minimum probability threshold (0-100)
    page_size : int
        Number of rows per page
    
    Returns
    -------
    dash_bootstrap_components.Container
        Complete batch prediction interface
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H4([
                    html.I(className="bi bi-table me-2"),
                    "Batch Prediction Results"
                ], className="mb-3")
            ])
        ]),
        
        # Filter controls
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6([
                            html.I(className="bi bi-funnel me-2"),
                            "Filters"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        create_filter_controls(
                            prob_win_threshold=prob_win_threshold
                        )
                    ])
                ], className="shadow-sm mb-3")
            ])
        ]),
        
        # Export button
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_export_button(),
                    dbc.Button([
                        html.I(className="bi bi-arrow-clockwise me-2"),
                        "Refresh"
                    ], id='refresh-batch-predictions-btn', color="secondary", outline=True)
                ], className="mb-3")
            ])
        ]),
        
        # Table
        dbc.Row([
            dbc.Col([
                html.Div(id='batch-prediction-table-container', children=[
                    create_batch_prediction_table(
                        predictions_df,
                        quality_filter=quality_filter,
                        prob_win_threshold=prob_win_threshold,
                        page_size=page_size
                    )
                ])
            ])
        ])
    ], fluid=True)
