"""
ML Prediction Engine Callbacks

Handles all callbacks for the ML Prediction Engine page including:
- Loading models when page opens
- Running single predictions
- Running batch predictions
- Training models
- Updating settings
- Exporting predictions and reports

**Feature: ml-prediction-engine**
**Validates: Requirements 0.1, 0.2, 0.3, 0.4, 16.1, 16.2**
"""

import os
import json
import base64
import io
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import dash
from dash import Input, Output, State, callback, html, no_update, ctx, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Import ML engine components
from backend.ml_engine.pipeline_prediction import PredictionPipeline
from backend.ml_engine.model_trainer import ModelTrainer, TrainingProgress

# Import frontend components
from frontend.components.prediction_summary_cards import create_prediction_summary_cards
from frontend.components.reliability_diagram import create_reliability_diagram
from frontend.components.r_distribution_fan_chart import create_distribution_analysis_section, create_fan_chart
from frontend.components.feature_importance_chart import create_feature_importance_section
from frontend.components.batch_prediction_table import create_batch_prediction_table
from frontend.components.training_modal import create_training_config_form
from frontend.components.ml_help_modal import get_help_content


def register_ml_prediction_callbacks(app):
    """Register all ML prediction engine callbacks"""
    
    # Callback 1: Load models when page opens
    @app.callback(
        [Output('ml-model-status-alert', 'children'),
         Output('ml-model-metadata-store', 'data')],
        [Input('main-tabs', 'active_tab')],
        prevent_initial_call=False  # Allow initial call to load models on app start
    )
    def load_models_on_page_open(active_tab):
        """Load ML models when ML Prediction Engine tab is opened"""
        
        # Only load when ML Prediction Engine tab is active
        if active_tab != 'ml-prediction-engine':
            raise PreventUpdate
        
        try:
            # Initialize pipeline
            pipeline = PredictionPipeline()
            model_dir = Path('data_processed/models')
            
            # Check if models exist
            if not model_dir.exists():
                alert = dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    html.Strong("Models not found. "),
                    "Please train models first using the 'Train Models' button."
                ], color="warning", dismissable=False)
                return alert, {'models_loaded': False}
            
            # Try to load models
            success = pipeline.load_models(model_dir)
            
            if success:
                # Load features from config to show what's configured
                from backend.ml_engine.feature_selector import load_features_from_config
                try:
                    config_features = load_features_from_config('config/ml_prediction_config.yaml')
                except Exception as e:
                    config_features = []
                
                # Get model metadata
                metadata = {
                    'models_loaded': True,
                    'feature_names': pipeline.feature_names,
                    'n_features': len(pipeline.feature_names) if pipeline.feature_names else 0,
                    'config_features': config_features,
                    'n_config_features': len(config_features),
                    'model_dir': str(model_dir),
                    'loaded_at': datetime.now().isoformat()
                }
                
                # Create alert with feature info
                alert_content = [
                    html.I(className="bi bi-check-circle me-2"),
                    html.Strong("Models loaded successfully. "),
                    f"Ready for prediction with {metadata['n_features']} features."
                ]
                
                # Add config features info if available
                if config_features:
                    alert_content.extend([
                        html.Br(),
                        html.Small([
                            html.I(className="bi bi-cpu me-1"),
                            f"Config features ({len(config_features)}): ",
                            ", ".join(config_features[:3]),
                            "..." if len(config_features) > 3 else ""
                        ], className="text-muted")
                    ])
                
                alert = dbc.Alert(alert_content, color="success", dismissable=True, duration=8000)
                
                return alert, metadata
            else:
                alert = dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    html.Strong("Failed to load models. "),
                    "Some model files may be missing or corrupted."
                ], color="danger", dismissable=False)
                return alert, {'models_loaded': False}
                
        except Exception as e:
            alert = dbc.Alert([
                html.I(className="bi bi-x-circle me-2"),
                html.Strong("Error loading models: "),
                str(e)
            ], color="danger", dismissable=False)
            return alert, {'models_loaded': False, 'error': str(e)}
    
    
    # Callback 1b: Display feature configuration info
    @app.callback(
        Output('ml-feature-config-info', 'children'),
        [Input('main-tabs', 'active_tab')],
        prevent_initial_call=True
    )
    def display_feature_config_info(active_tab):
        """Display configured features from Auto Feature Selection"""
        
        if active_tab != 'ml-prediction-engine':
            raise PreventUpdate
        
        try:
            from backend.ml_engine.feature_selector import load_features_from_config
            
            # Load features from config
            features = load_features_from_config('config/ml_prediction_config.yaml')
            
            if not features:
                return dbc.Alert([
                    html.I(className="bi bi-info-circle me-2"),
                    html.Strong("No features configured. "),
                    "Run Auto Feature Selection and click 'Use for ML' to configure features."
                ], color="info", dismissable=True)
            
            # Create feature info card
            return dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.I(className="bi bi-cpu me-2"),
                                html.Strong(f"Configured Features ({len(features)})")
                            ], className="mb-2"),
                            html.Div([
                                dbc.Badge(feature, color="primary", className="me-1 mb-1")
                                for feature in features
                            ])
                        ], md=10),
                        dbc.Col([
                            dbc.Button([
                                html.I(className="bi bi-arrow-clockwise me-1"),
                                "Reload"
                            ], id="ml-reload-features-btn", color="secondary", size="sm", outline=True)
                        ], md=2, className="text-end")
                    ])
                ])
            ], className="border-primary")
            
        except FileNotFoundError:
            return dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                html.Strong("Config file not found. "),
                "Please run Auto Feature Selection first."
            ], color="warning", dismissable=True)
        except Exception as e:
            return dbc.Alert([
                html.I(className="bi bi-x-circle me-2"),
                html.Strong("Error loading feature config: "),
                str(e)
            ], color="danger", dismissable=True)
    
    
    # Callback 1c: Reload feature config when button clicked
    @app.callback(
        Output('ml-feature-config-info', 'children', allow_duplicate=True),
        Input('ml-reload-features-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def reload_feature_config(n_clicks):
        """Reload feature configuration from config file"""
        
        if not n_clicks:
            raise PreventUpdate
        
        try:
            from backend.ml_engine.feature_selector import load_features_from_config
            
            # Load features from config
            features = load_features_from_config('config/ml_prediction_config.yaml')
            
            if not features:
                return dbc.Alert([
                    html.I(className="bi bi-info-circle me-2"),
                    html.Strong("No features configured. "),
                    "Run Auto Feature Selection and click 'Use for ML' to configure features."
                ], color="info", dismissable=True)
            
            # Create feature info card with success message
            return dbc.Card([
                dbc.CardBody([
                    dbc.Alert([
                        html.I(className="bi bi-check-circle me-2"),
                        html.Strong("Features reloaded successfully!")
                    ], color="success", dismissable=True, duration=3000, className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.I(className="bi bi-cpu me-2"),
                                html.Strong(f"Configured Features ({len(features)})")
                            ], className="mb-2"),
                            html.Div([
                                dbc.Badge(feature, color="primary", className="me-1 mb-1")
                                for feature in features
                            ])
                        ], md=10),
                        dbc.Col([
                            dbc.Button([
                                html.I(className="bi bi-arrow-clockwise me-1"),
                                "Reload"
                            ], id="ml-reload-features-btn", color="secondary", size="sm", outline=True)
                        ], md=2, className="text-end")
                    ])
                ])
            ], className="border-primary")
            
        except Exception as e:
            return dbc.Alert([
                html.I(className="bi bi-x-circle me-2"),
                html.Strong("Error reloading features: "),
                str(e)
            ], color="danger", dismissable=True)
    
    
    # Callback 2: Toggle upload container based on data source
    @app.callback(
        Output('ml-upload-container', 'style'),
        Input('ml-data-source', 'value')
    )
    def toggle_upload_container(data_source):
        """Show/hide upload container based on data source selection"""
        if data_source == 'upload':
            return {'display': 'block'}
        return {'display': 'none'}
    
    
    # Callback 3: Run single or batch prediction
    @app.callback(
        [Output('ml-prediction-summary-cards', 'children'),
         Output('ml-prediction-results-store', 'data')],
        [Input('ml-run-prediction-btn', 'n_clicks')],
        [State('ml-prediction-mode', 'value'),
         State('ml-data-source', 'value'),
         State('ml-upload-data', 'contents'),
         State('ml-upload-data', 'filename'),
         State('merged-data-store', 'data'),
         State('ml-model-metadata-store', 'data')],
        prevent_initial_call=True
    )
    def run_prediction(n_clicks, prediction_mode, data_source, upload_contents, 
                      upload_filename, merged_data, model_metadata):
        """Run single or batch prediction"""
        
        if not n_clicks:
            raise PreventUpdate
        
        # Debug logging
        print(f"\n{'='*80}")
        print(f"ML PREDICTION CALLBACK TRIGGERED")
        print(f"{'='*80}")
        print(f"Prediction mode: {prediction_mode}")
        print(f"Data source: {data_source}")
        print(f"Model metadata: {model_metadata}")
        print(f"Merged data available: {merged_data is not None}")
        if merged_data:
            print(f"Merged data keys: {merged_data.keys() if isinstance(merged_data, dict) else 'Not a dict'}")
        
        # Check if models are loaded
        if not model_metadata or not model_metadata.get('models_loaded'):
            error_card = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                html.Strong("Models not loaded. "),
                "Please train models first using the 'Train Models' button.",
                html.Br(),
                html.Small(f"Debug: model_metadata = {model_metadata}", className="text-muted")
            ], color="warning")
            print(f"ERROR: Models not loaded")
            return error_card, no_update
        
        try:
            # Load data based on source
            if data_source == 'merged':
                if not merged_data or 'data' not in merged_data:
                    error_card = dbc.Alert([
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        html.Strong("No merged data available. "),
                        "Please load data first from the Dashboard.",
                        html.Br(),
                        html.Small("Tip: Go to Trade Analysis Dashboard and load your CSV data.", className="text-muted")
                    ], color="warning")
                    print(f"ERROR: No merged data available")
                    return error_card, no_update
                
                print(f"Loading merged data...")
                df = pd.read_json(merged_data['data'], orient='split')
                print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
                
            elif data_source == 'upload':
                if not upload_contents:
                    error_card = dbc.Alert([
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        "Please upload a CSV file."
                    ], color="warning")
                    return error_card, no_update
                
                # Parse uploaded file
                content_type, content_string = upload_contents.split(',')
                decoded = base64.b64decode(content_string)
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            # Initialize pipeline
            print(f"Initializing pipeline...")
            pipeline = PredictionPipeline()
            print(f"Loading models from: {model_metadata['model_dir']}")
            pipeline.load_models(model_metadata['model_dir'])
            print(f"Models loaded successfully. Features: {pipeline.feature_names}")
            
            # Run prediction based on mode
            if prediction_mode == 'single':
                # Use first row for single prediction
                if len(df) == 0:
                    error_card = dbc.Alert([
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        "No data available for prediction."
                    ], color="warning")
                    print(f"ERROR: Empty dataframe")
                    return error_card, no_update
                
                print(f"Running single prediction...")
                sample = df.iloc[0]
                result = pipeline.predict_for_sample(sample)
                print(f"Prediction result: {result}")
                
                # Create summary cards
                print(f"Creating summary cards...")
                summary_cards = create_prediction_summary_cards(result)
                print(f"Summary cards created successfully")
                
                # Store results
                results_data = {
                    'mode': 'single',
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"SUCCESS: Single prediction completed")
                print(f"{'='*80}\n")
                return summary_cards, results_data
                
            else:  # batch mode
                # Run batch prediction
                print(f"Running batch prediction on {len(df)} samples...")
                results_df = pipeline.predict_for_batch(df)
                print(f"Batch prediction completed: {len(results_df)} results")
                
                # Create batch table
                print(f"Creating batch table...")
                batch_table = create_batch_prediction_table(results_df)
                print(f"Batch table created successfully")
                
                # Store results
                results_data = {
                    'mode': 'batch',
                    'results': results_df.to_json(orient='split', date_format='iso'),
                    'n_samples': len(results_df),
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"SUCCESS: Batch prediction completed")
                print(f"{'='*80}\n")
                return batch_table, results_data
                
        except Exception as e:
            error_card = dbc.Alert([
                html.I(className="bi bi-x-circle me-2"),
                html.Strong("Prediction error: "),
                str(e),
                html.Br(),
                html.Small(f"Error type: {type(e).__name__}", className="text-muted")
            ], color="danger")
            print(f"PREDICTION ERROR:")
            print(f"{'='*80}")
            print(traceback.format_exc())
            print(f"{'='*80}\n")
            return error_card, no_update
    
    
    # Callback 4: Update probability analysis content
    @app.callback(
        Output('ml-probability-analysis-content', 'children'),
        [Input('ml-probability-tabs', 'active_tab'),
         Input('ml-prediction-results-store', 'data')],
        [State('merged-data-store', 'data')]
    )
    def update_probability_analysis(active_tab, prediction_results, merged_data):
        """Update probability analysis visualizations"""
        
        if not prediction_results:
            return html.Div([
                html.P("Run a prediction to see probability analysis.", 
                      className="text-muted text-center mt-4")
            ])
        
        try:
            if active_tab == 'reliability':
                # Create reliability diagram
                # This requires batch predictions with actual outcomes
                return html.Div([
                    html.P("Reliability diagram requires batch predictions with actual outcomes.",
                          className="text-muted text-center mt-4"),
                    html.P("Run batch prediction on data with 'trade_success' column to see calibration analysis.",
                          className="text-muted text-center small")
                ])
                
            elif active_tab == 'prob-dist':
                # Probability distribution histogram
                return html.Div([
                    html.P("Probability distribution visualization coming soon.",
                          className="text-muted text-center mt-4")
                ])
                
            elif active_tab == 'calib-metrics':
                # Calibration metrics
                return html.Div([
                    html.P("Calibration metrics display coming soon.",
                          className="text-muted text-center mt-4")
                ])
                
        except Exception as e:
            return dbc.Alert([
                html.I(className="bi bi-x-circle me-2"),
                f"Error: {str(e)}"
            ], color="danger")
    
    
    # Callback 5: Update distribution analysis content
    @app.callback(
        Output('ml-distribution-analysis-content', 'children'),
        [Input('ml-distribution-tabs', 'active_tab'),
         Input('ml-prediction-results-store', 'data')]
    )
    def update_distribution_analysis(active_tab, prediction_results):
        """Update distribution analysis visualizations"""
        
        if not prediction_results:
            return html.Div([
                html.P("Run a prediction to see distribution analysis.",
                      className="text-muted text-center mt-4")
            ])
        
        try:
            if active_tab == 'fan-chart':
                # Create R_multiple fan chart
                if prediction_results.get('mode') == 'batch':
                    results_df = pd.read_json(prediction_results['results'], orient='split')
                    
                    # Extract predictions for fan chart
                    if all(col in results_df.columns for col in ['R_P10_conf', 'R_P50_raw', 'R_P90_conf']):
                        fig = create_fan_chart(
                            y_pred_p10=results_df['R_P10_conf'].values,
                            y_pred_p50=results_df['R_P50_raw'].values,
                            y_pred_p90=results_df['R_P90_conf'].values,
                            y_true=results_df['R_multiple'].values if 'R_multiple' in results_df.columns else None
                        )
                        return dcc.Graph(figure=fig)
                    else:
                        return dbc.Alert("Required prediction columns not found in results.", color="warning")
                else:
                    # Single prediction - show as point
                    result = prediction_results.get('result', {})
                    return html.Div([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("R_multiple Distribution", className="mb-3"),
                                html.Div([
                                    dbc.Row([
                                        dbc.Col([
                                            html.P("P10 (Conformal):", className="mb-1"),
                                            html.H4(f"{result.get('R_P10_conf', 0):.2f}R", 
                                                   className="text-danger")
                                        ], md=4),
                                        dbc.Col([
                                            html.P("P50 (Expected):", className="mb-1"),
                                            html.H4(f"{result.get('R_P50_raw', 0):.2f}R",
                                                   className="text-primary")
                                        ], md=4),
                                        dbc.Col([
                                            html.P("P90 (Conformal):", className="mb-1"),
                                            html.H4(f"{result.get('R_P90_conf', 0):.2f}R",
                                                   className="text-success")
                                        ], md=4)
                                    ])
                                ])
                            ])
                        ])
                    ])
                    
            elif active_tab == 'dist-comp':
                return html.Div([
                    html.P("Distribution comparison visualization coming soon.",
                          className="text-muted text-center mt-4")
                ])
                
            elif active_tab == 'coverage':
                return html.Div([
                    html.P("Coverage analysis visualization coming soon.",
                          className="text-muted text-center mt-4")
                ])
                
        except Exception as e:
            return dbc.Alert([
                html.I(className="bi bi-x-circle me-2"),
                f"Error: {str(e)}"
            ], color="danger")
    
    
    # Callback 6: Toggle training modal
    @app.callback(
        [Output('ml-training-modal', 'is_open'),
         Output('ml-training-modal-content', 'children')],
        [Input('ml-train-models-btn', 'n_clicks'),
         Input('ml-training-cancel-btn', 'n_clicks'),
         Input('ml-training-start-btn', 'n_clicks')],
        [State('ml-training-modal', 'is_open'),
         State('merged-data-store', 'data')],
        prevent_initial_call=True
    )
    def toggle_training_modal(train_clicks, cancel_clicks, start_clicks, 
                             is_open, merged_data):
        """Toggle training modal and initialize content"""
        
        triggered_id = ctx.triggered_id
        
        if triggered_id == 'ml-train-models-btn':
            # Open modal and show configuration form
            if not merged_data or 'data' not in merged_data:
                content = dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    "No data available. Please load data first."
                ], color="warning")
                return True, content
            
            # Get available features from merged data
            df = pd.read_json(merged_data['data'], orient='split')
            
            # Filter out trade metadata columns (same as Auto Feature Selection)
            trade_metadata_cols = [
                # Trade identifiers
                'Ticket_id', 'ticket_id', 'ticket', 'trade_id',
                # Trade execution details
                'Symbol', 'symbol', 'Type', 'type', 'OpenPrice', 'open_price',
                'ClosePrice', 'close_price', 'Volume', 'volume',
                # EA-specific parameters
                'Timeframe', 'timeframe', 'UseFibo50Filter', 'FiboBasePrice', 
                'FiboRange', 'MagicNumber', 'magic_number', 'StrategyType', 
                'strategy_type', 'ConsecutiveSLCount', 'TPHitsToday', 'SLHitsToday',
                # Session info
                'SessionHour', 'SessionMinute', 'SessionDayOfWeek', 'entry_session',
                # Trade results (DATA LEAKAGE)
                'gross_profit', 'net_profit', 'R_multiple', 'ExitReason', 'exit_reason',
                'MFEPips', 'MAEPips', 'MAE_R', 'MFE_R', 'max_drawdown_k', 'max_runup_k',
                'future_return_k', 'equity_at_entry', 'equity_after_trade',
                # Trade timing
                'exit_time', 'holding_bars', 'holding_minutes', 'K_bars',
                # Price levels
                'entry_price', 'sl_price', 'tp_price', 'sl_distance', 
                'money_risk', 'risk_percent', 'MaxSLTP',
                # Timestamps
                'Timestamp', 'timestamp', 'entry_time', 'exit_time'
            ]
            
            # Target-related columns
            target_related_cols = [
                'trade_success', 'y_win', 'y_hit_1R', 'y_hit_2R', 
                'y_future_win_k', 'win', 'success', 'target',
                'Morang_y_win', 'label', 'outcome'
            ]
            
            # Create lowercase mapping for case-insensitive comparison
            trade_metadata_lower = [c.lower() for c in trade_metadata_cols]
            target_related_lower = [c.lower() for c in target_related_cols]
            
            # Filter features
            available_features = []
            for col in df.columns:
                # Skip if in metadata or target lists (case-insensitive)
                if (col in trade_metadata_cols or col.lower() in trade_metadata_lower or
                    col in target_related_cols or col.lower() in target_related_lower):
                    continue
                # Only include numeric columns
                if df[col].dtype in [np.int64, np.float64]:
                    available_features.append(col)
            
            # Load features from config if available
            try:
                from backend.ml_engine.feature_selector import load_features_from_config
                config_features = load_features_from_config('config/ml_prediction_config.yaml')
                # Use config features if they exist in available features
                selected_features = [f for f in config_features if f in available_features]
                if not selected_features:
                    selected_features = available_features[:8] if len(available_features) >= 8 else available_features
            except:
                selected_features = available_features[:8] if len(available_features) >= 8 else available_features
            
            # Create configuration form
            content = create_training_config_form(
                available_features=available_features,
                selected_features=available_features[:8] if len(available_features) >= 8 else available_features
            )
            
            return True, content
            
        elif triggered_id in ['ml-training-cancel-btn', 'ml-training-start-btn']:
            return False, no_update
        
        return is_open, no_update
    
    
    # Callback 7: Start training process
    @app.callback(
        [Output('ml-training-status-store', 'data'),
         Output('ml-model-status-alert', 'children', allow_duplicate=True),
         Output('ml-model-metadata-store', 'data', allow_duplicate=True)],
        [Input('ml-training-start-btn', 'n_clicks')],
        [State('training-feature-select', 'value'),
         State('training-train-ratio', 'value'),
         State('training-calib-ratio', 'value'),
         State('training-cv-folds', 'value'),
         State('training-n-estimators', 'value'),
         State('training-learning-rate', 'value'),
         State('training-max-depth', 'value'),
         State('merged-data-store', 'data')],
        prevent_initial_call=True
    )
    def start_training(n_clicks, selected_features, train_ratio, calib_ratio,
                      cv_folds, n_estimators, learning_rate, max_depth, merged_data):
        """Start model training process"""
        
        if not n_clicks:
            raise PreventUpdate
        
        try:
            # Validate inputs
            if not selected_features or len(selected_features) < 5:
                alert = dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    "Please select at least 5 features."
                ], color="warning")
                return no_update, alert, no_update
            
            if not merged_data or 'data' not in merged_data:
                alert = dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    "No data available for training."
                ], color="warning")
                return no_update, alert, no_update
            
            # Load data
            df = pd.read_json(merged_data['data'], orient='split')
            
            # Check required columns
            if 'trade_success' not in df.columns or 'R_multiple' not in df.columns:
                alert = dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    "Data must contain 'trade_success' and 'R_multiple' columns."
                ], color="warning")
                return no_update, alert, no_update
            
            # Update config with user-selected parameters
            import yaml
            config_path = 'config/ml_prediction_config.yaml'
            
            try:
                # Load existing config
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Update data split ratios
                config['data_split']['train_ratio'] = train_ratio
                config['data_split']['calib_ratio'] = calib_ratio
                config['data_split']['test_ratio'] = 1.0 - train_ratio - calib_ratio
                
                # Update CV folds
                config['preprocessing']['cv_folds'] = cv_folds
                
                # Update model hyperparameters if provided
                if n_estimators:
                    config['model_hyperparameters']['classifier']['n_estimators'] = n_estimators
                    config['model_hyperparameters']['quantile']['n_estimators'] = n_estimators
                if learning_rate:
                    config['model_hyperparameters']['classifier']['learning_rate'] = learning_rate
                    config['model_hyperparameters']['quantile']['learning_rate'] = learning_rate
                if max_depth:
                    config['model_hyperparameters']['classifier']['max_depth'] = max_depth
                    config['model_hyperparameters']['quantile']['max_depth'] = max_depth
                
                # Save updated config
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                print(f"✓ Config updated: train={train_ratio}, calib={calib_ratio}, cv_folds={cv_folds}")
                
            except Exception as e:
                print(f"Warning: Could not update config: {e}")
                # Continue anyway with existing config
            
            # Initialize trainer
            trainer = ModelTrainer()
            
            # Start training (ModelTrainer reads parameters from config file)
            metrics = trainer.train_all_components(
                data=df,
                feature_columns=selected_features,
                target_column='R_multiple',
                win_column='trade_success',
                overwrite=True  # Allow overwriting existing models
            )
            
            # Training completed successfully - now reload models into pipeline
            model_dir = Path('data_processed/models')
            pipeline = PredictionPipeline()
            pipeline.load_models(model_dir)
            
            # Update model metadata store
            model_metadata = {
                'models_loaded': True,
                'feature_names': selected_features,
                'n_features': len(selected_features),
                'model_dir': str(model_dir),
                'loaded_at': datetime.now().isoformat(),
                'training_metrics': {
                    'auc': metrics['classifier'].get('auc_val', metrics['classifier']['auc_train']),
                    'brier': metrics['calibration']['brier_after'],
                    'mae_p50': metrics['quantile'].get('mae_p50_val', metrics['quantile']['mae_p50_train'])
                }
            }
            
            # Training completed successfully
            alert = dbc.Alert([
                html.I(className="bi bi-check-circle me-2"),
                html.Strong("Training completed successfully! "),
                f"AUC: {metrics['classifier'].get('auc_val', metrics['classifier']['auc_train']):.3f}, ",
                f"Brier: {metrics['calibration']['brier_after']:.3f}. ",
                html.Br(),
                "Models loaded and ready for prediction."
            ], color="success", dismissable=True, duration=10000)
            
            training_status = {
                'status': 'completed',
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            return training_status, alert, model_metadata
            
        except Exception as e:
            alert = dbc.Alert([
                html.I(className="bi bi-x-circle me-2"),
                html.Strong("Training failed: "),
                str(e)
            ], color="danger")
            print(f"Training error: {traceback.format_exc()}")
            
            training_status = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            return training_status, alert, no_update
    
    
    # Callback 8: Export predictions to CSV
    @app.callback(
        Output('ml-download-predictions', 'data'),
        [Input('ml-export-predictions-btn', 'n_clicks')],
        [State('ml-prediction-results-store', 'data')],
        prevent_initial_call=True
    )
    def export_predictions(n_clicks, prediction_results):
        """Export predictions to CSV file"""
        
        if not n_clicks or not prediction_results:
            raise PreventUpdate
        
        try:
            if prediction_results.get('mode') == 'batch':
                # Export batch results
                results_df = pd.read_json(prediction_results['results'], orient='split')
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'ml_predictions_{timestamp}.csv'
                
                return dcc.send_data_frame(results_df.to_csv, filename, index=False)
                
            elif prediction_results.get('mode') == 'single':
                # Export single result
                result = prediction_results.get('result', {})
                results_df = pd.DataFrame([result])
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'ml_prediction_single_{timestamp}.csv'
                
                return dcc.send_data_frame(results_df.to_csv, filename, index=False)
                
        except Exception as e:
            print(f"Export error: {str(e)}")
            raise PreventUpdate
    
    
    # Callback 9: Export report to PDF (placeholder)
    @app.callback(
        Output('ml-download-report', 'data'),
        [Input('ml-export-report-btn', 'n_clicks')],
        [State('ml-prediction-results-store', 'data')],
        prevent_initial_call=True
    )
    def export_report(n_clicks, prediction_results):
        """Export prediction report to PDF (placeholder)"""
        
        if not n_clicks or not prediction_results:
            raise PreventUpdate
        
        # TODO: Implement PDF report generation
        # For now, just raise PreventUpdate
        print("PDF export not yet implemented")
        raise PreventUpdate
    
    
    # Callback 10: Navigate back to dashboard
    @app.callback(
        Output('main-tabs', 'active_tab', allow_duplicate=True),
        [Input('ml-back-to-dashboard-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def navigate_back_to_dashboard(n_clicks):
        """Navigate back to Trade Analysis Dashboard"""
        if n_clicks:
            return 'trade-analysis'
        raise PreventUpdate
    
    
    # Callback 11: Toggle help modal
    @app.callback(
        Output('ml-help-modal', 'is_open'),
        [Input('ml-help-btn', 'n_clicks'),
         Input('ml-help-close-btn', 'n_clicks')],
        [State('ml-help-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_help_modal(open_clicks, close_clicks, is_open):
        """Toggle help modal open/close"""
        if open_clicks or close_clicks:
            return not is_open
        return is_open
    
    
    # Callback 12: Render help content based on active tab
    @app.callback(
        Output('ml-help-content', 'children'),
        Input('ml-help-tabs', 'active_tab')
    )
    def render_help_content(active_tab):
        """Render help content for the selected tab"""
        return get_help_content(active_tab)


    # Callback: Send ML predictions to What-If Scenarios
    @app.callback(
        [Output('ml-predictions-store', 'data'),
         Output('main-tabs', 'active_tab', allow_duplicate=True)],
        [Input('ml-add-whatif-scenario-btn', 'n_clicks')],
        [State('ml-prediction-results-store', 'data')],
        prevent_initial_call=True
    )
    def send_predictions_to_whatif(n_clicks, prediction_results):
        """
        Send ML predictions to What-If Scenarios page.
        
        This callback transfers ML prediction results to the What-If Scenarios
        page so users can create scenarios based on ML predictions.
        
        Validates: Requirements 15.4
        """
        if not n_clicks or not prediction_results:
            raise PreventUpdate
        
        try:
            # Convert prediction results to format expected by What-If Scenarios
            # prediction_results should be a list of dicts or a dict with 'predictions' key
            if isinstance(prediction_results, dict):
                if 'predictions' in prediction_results:
                    predictions_data = prediction_results['predictions']
                else:
                    predictions_data = [prediction_results]
            else:
                predictions_data = prediction_results
            
            # Navigate to What-If Scenarios tab
            return predictions_data, 'what-if-scenarios'
            
        except Exception as e:
            print(f"Error sending predictions to What-If Scenarios: {e}")
            traceback.print_exc()
            raise PreventUpdate
