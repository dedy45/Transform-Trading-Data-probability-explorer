"""
Demo for Training Modal Component

This demo shows all states of the training modal:
1. Configuration form
2. Overwrite warning
3. Training progress
4. Training metrics
5. Error handling
"""

import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from training_modal import (
    create_training_modal,
    create_training_button
)
import time

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])

# Sample data
AVAILABLE_FEATURES = [
    'trend_strength_tf',
    'swing_position',
    'volatility_regime',
    'support_distance',
    'momentum_score',
    'time_of_day',
    'spread_ratio',
    'volume_profile',
    'rsi_14',
    'macd_signal',
    'bollinger_width',
    'atr_14'
]

SELECTED_FEATURES = AVAILABLE_FEATURES[:8]

SAMPLE_PROGRESS = {
    'current_component': 'classifier',
    'current_step': 'Training LightGBM classifier with 5-fold CV',
    'progress_percentage': 35.0,
    'component_status': {
        'preprocessing': 'completed',
        'classifier': 'in_progress',
        'calibration': 'pending',
        'quantile': 'pending',
        'conformal': 'pending'
    }
}

SAMPLE_METRICS = {
    'classifier': {
        'auc_train': 0.78,
        'auc_val': 0.75,
        'brier_score_train': 0.18,
        'brier_score_val': 0.20
    },
    'calibration': {
        'brier_before': 0.20,
        'brier_after': 0.18,
        'brier_improvement': 0.02,
        'ece_before': 0.08,
        'ece_after': 0.03
    },
    'quantile': {
        'mae_p10_train': 0.35,
        'mae_p10_val': 0.38,
        'mae_p50_train': 0.42,
        'mae_p50_val': 0.45,
        'mae_p90_train': 0.48,
        'mae_p90_val': 0.52
    },
    'conformal': {
        'target_coverage': 0.90,
        'actual_coverage_calib': 0.91,
        'actual_coverage_test': 0.89,
        'avg_interval_width': 3.2
    },
    'metadata': {
        'n_train': 7620,
        'n_calib': 2540,
        'n_test': 2540,
        'n_features': 8,
        'feature_columns': SELECTED_FEATURES,
        'training_time_seconds': 45.2,
        'timestamp': '2025-11-24 09:00:00'
    }
}

# App layout
app.layout = dbc.Container([
    html.H1([
        html.I(className="bi bi-gear-fill me-3"),
        "Training Modal Demo"
    ], className="my-4"),
    
    dbc.Card([
        dbc.CardHeader("Demo Controls"),
        dbc.CardBody([
            html.P("Click buttons below to see different modal states:", className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button([
                        html.I(className="bi bi-1-circle me-2"),
                        "Configuration Form"
                    ], id="demo-config-btn", color="primary", className="w-100 mb-2")
                ], md=6, lg=4),
                
                dbc.Col([
                    dbc.Button([
                        html.I(className="bi bi-2-circle me-2"),
                        "Overwrite Warning"
                    ], id="demo-warning-btn", color="warning", className="w-100 mb-2")
                ], md=6, lg=4),
                
                dbc.Col([
                    dbc.Button([
                        html.I(className="bi bi-3-circle me-2"),
                        "Training Progress"
                    ], id="demo-progress-btn", color="info", className="w-100 mb-2")
                ], md=6, lg=4),
                
                dbc.Col([
                    dbc.Button([
                        html.I(className="bi bi-4-circle me-2"),
                        "Training Metrics"
                    ], id="demo-metrics-btn", color="success", className="w-100 mb-2")
                ], md=6, lg=4),
                
                dbc.Col([
                    dbc.Button([
                        html.I(className="bi bi-5-circle me-2"),
                        "Error State"
                    ], id="demo-error-btn", color="danger", className="w-100 mb-2")
                ], md=6, lg=4),
                
                dbc.Col([
                    dbc.Button([
                        html.I(className="bi bi-x-circle me-2"),
                        "Close Modal"
                    ], id="demo-close-btn", color="secondary", className="w-100 mb-2")
                ], md=6, lg=4)
            ])
        ])
    ], className="mb-4"),
    
    # Training Modal (will be updated by callbacks)
    html.Div(id="modal-container"),
    
    # Store for modal state
    dcc.Store(id='modal-state-store', data={'is_open': False, 'state': 'config'})
    
], fluid=True, className="py-4")


@app.callback(
    Output('modal-container', 'children'),
    Output('modal-state-store', 'data'),
    Input('demo-config-btn', 'n_clicks'),
    Input('demo-warning-btn', 'n_clicks'),
    Input('demo-progress-btn', 'n_clicks'),
    Input('demo-metrics-btn', 'n_clicks'),
    Input('demo-error-btn', 'n_clicks'),
    Input('demo-close-btn', 'n_clicks'),
    State('modal-state-store', 'data'),
    prevent_initial_call=True
)
def update_modal(config_clicks, warning_clicks, progress_clicks, metrics_clicks, error_clicks, close_clicks, state):
    """Update modal based on button clicks."""
    ctx = callback_context
    
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Close modal
    if button_id == 'demo-close-btn':
        modal = create_training_modal(is_open=False)
        return modal, {'is_open': False, 'state': 'config'}
    
    # Configuration Form State
    if button_id == 'demo-config-btn':
        modal = create_training_modal(
            is_open=True,
            available_features=AVAILABLE_FEATURES,
            selected_features=SELECTED_FEATURES,
            models_exist=False,
            show_warning=False,
            show_progress=False,
            show_metrics=False
        )
        return modal, {'is_open': True, 'state': 'config'}
    
    # Warning State
    if button_id == 'demo-warning-btn':
        modal = create_training_modal(
            is_open=True,
            available_features=AVAILABLE_FEATURES,
            selected_features=SELECTED_FEATURES,
            models_exist=True,
            show_warning=True,
            show_progress=False,
            show_metrics=False
        )
        return modal, {'is_open': True, 'state': 'warning'}
    
    # Progress State
    if button_id == 'demo-progress-btn':
        modal = create_training_modal(
            is_open=True,
            show_progress=True,
            progress_data=SAMPLE_PROGRESS
        )
        return modal, {'is_open': True, 'state': 'progress'}
    
    # Metrics State
    if button_id == 'demo-metrics-btn':
        modal = create_training_modal(
            is_open=True,
            show_metrics=True,
            metrics_data=SAMPLE_METRICS
        )
        return modal, {'is_open': True, 'state': 'metrics'}
    
    # Error State
    if button_id == 'demo-error-btn':
        error_msg = """Training failed at component: classifier

Error: ValueError: Insufficient data for training
  File "backend/ml_engine/model_trainer.py", line 245, in _train_classifier
    raise ValueError("Insufficient data: need at least 1000 samples")

The system has automatically rolled back to the previous models."""
        
        modal = create_training_modal(
            is_open=True,
            error_message=error_msg
        )
        return modal, {'is_open': True, 'state': 'error'}
    
    return dash.no_update, dash.no_update


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Training Modal Demo")
    print("="*80)
    print("\nDemo showcases all training modal states:")
    print("1. Configuration Form - Set training parameters")
    print("2. Overwrite Warning - Confirm overwriting existing models")
    print("3. Training Progress - Real-time progress tracking")
    print("4. Training Metrics - Display results after completion")
    print("5. Error State - Show errors with rollback info")
    print("\nOpen browser at: http://127.0.0.1:8050")
    print("="*80 + "\n")
    
    app.run_server(debug=True, port=8050)
