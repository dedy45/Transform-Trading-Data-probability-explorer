"""
ML Settings Modal Callbacks

Handles all callbacks for the ML Prediction Engine settings modal including:
- Opening/closing the modal
- Tab switching
- Saving settings to config file
- Validating feature selection (minimum 5 features)
- Select All / Unselect All feature buttons
"""

from dash import Input, Output, State, callback, html, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from frontend.components.ml_settings_modal import (
    get_settings_content,
    save_settings_to_config
)


def register_ml_settings_callbacks(app):
    """Register all ML settings modal callbacks"""
    
    @app.callback(
        Output('ml-settings-modal', 'is_open'),
        [Input('ml-settings-btn', 'n_clicks'),
         Input('ml-settings-cancel-btn', 'n_clicks'),
         Input('ml-settings-save-btn', 'n_clicks')],
        [State('ml-settings-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_settings_modal(open_clicks, cancel_clicks, save_clicks, is_open):
        """Toggle settings modal open/close"""
        if open_clicks or cancel_clicks or save_clicks:
            return not is_open
        return is_open
    
    
    @app.callback(
        Output('ml-settings-content', 'children'),
        Input('ml-settings-tabs', 'active_tab'),
        prevent_initial_call=False
    )
    def update_settings_content(active_tab):
        """Update settings content based on active tab"""
        if not active_tab:
            raise PreventUpdate
        
        return get_settings_content(active_tab)
    
    
    @app.callback(
        Output('ml-settings-feature-checklist', 'value'),
        [Input('ml-settings-select-all-btn', 'n_clicks'),
         Input('ml-settings-unselect-all-btn', 'n_clicks')],
        [State('ml-settings-feature-checklist', 'options')],
        prevent_initial_call=True
    )
    def handle_select_all_unselect_all(select_all_clicks, unselect_all_clicks, options):
        """Handle Select All and Unselect All buttons"""
        from dash import ctx
        
        if not ctx.triggered_id:
            raise PreventUpdate
        
        if ctx.triggered_id == 'ml-settings-select-all-btn':
            # Select all features
            return [opt['value'] for opt in options]
        elif ctx.triggered_id == 'ml-settings-unselect-all-btn':
            # Unselect all features
            return []
        
        raise PreventUpdate
    
    
    @app.callback(
        Output('ml-settings-feature-count', 'children'),
        Input('ml-settings-feature-checklist', 'value'),
        prevent_initial_call=True
    )
    def update_feature_count(selected_features):
        """Update feature count display"""
        if not selected_features:
            return "Selected: 0 features (minimum 5 required)"
        
        count = len(selected_features)
        
        if count < 5:
            return html.Span(
                f"Selected: {count} features (minimum 5 required)",
                className="text-danger fw-bold"
            )
        else:
            return html.Span(
                f"Selected: {count} features",
                className="text-success fw-bold"
            )
    
    
    @app.callback(
        [Output('ml-settings-modal', 'is_open', allow_duplicate=True),
         Output('ml-model-status-alert', 'children', allow_duplicate=True)],
        Input('ml-settings-save-btn', 'n_clicks'),
        [State('ml-settings-feature-checklist', 'value'),
         State('ml-settings-scaling', 'value'),
         State('ml-settings-missing', 'value'),
         State('ml-settings-classifier-n-estimators', 'value'),
         State('ml-settings-classifier-learning-rate', 'value'),
         State('ml-settings-classifier-max-depth', 'value'),
         State('ml-settings-classifier-min-child-samples', 'value'),
         State('ml-settings-classifier-subsample', 'value'),
         State('ml-settings-classifier-colsample', 'value'),
         State('ml-settings-quantile-n-estimators', 'value'),
         State('ml-settings-quantile-learning-rate', 'value'),
         State('ml-settings-quantile-max-depth', 'value'),
         State('ml-settings-threshold-aplus-prob', 'value'),
         State('ml-settings-threshold-aplus-r', 'value'),
         State('ml-settings-threshold-a-prob', 'value'),
         State('ml-settings-threshold-a-r', 'value'),
         State('ml-settings-threshold-b-prob', 'value'),
         State('ml-settings-threshold-b-r', 'value'),
         State('ml-settings-trade-quality-min', 'value'),
         State('ml-settings-n-bins', 'value'),
         State('ml-settings-n-top-features', 'value'),
         State('ml-settings-chart-height', 'value'),
         State('ml-settings-chart-template', 'value'),
         State('ml-settings-color-aplus', 'value'),
         State('ml-settings-color-a', 'value'),
         State('ml-settings-color-b', 'value'),
         State('ml-settings-color-c', 'value')],
        prevent_initial_call=True
    )
    def save_settings(n_clicks, 
                     # Features
                     selected_features, scaling, missing,
                     # Classifier hyperparameters
                     clf_n_est, clf_lr, clf_depth, clf_min_child, clf_subsample, clf_colsample,
                     # Quantile hyperparameters
                     qnt_n_est, qnt_lr, qnt_depth,
                     # Thresholds
                     aplus_prob, aplus_r, a_prob, a_r, b_prob, b_r, trade_min,
                     # Display
                     n_bins, n_top_features, chart_height, chart_template,
                     color_aplus, color_a, color_b, color_c):
        """Save all settings to config file"""
        
        if not n_clicks:
            raise PreventUpdate
        
        # Validate feature selection
        if not selected_features or len(selected_features) < 5:
            alert = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                "Error: Please select at least 5 features"
            ], color="danger", dismissable=True, duration=4000)
            return no_update, alert
        
        # Validate threshold ordering (A+ >= A >= B)
        if aplus_prob < a_prob or a_prob < b_prob:
            alert = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                "Error: Threshold ordering must be A+ >= A >= B for probability"
            ], color="danger", dismissable=True, duration=4000)
            return no_update, alert
        
        if aplus_r < a_r or a_r < b_r:
            alert = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                "Error: Threshold ordering must be A+ >= A >= B for R_P50"
            ], color="danger", dismissable=True, duration=4000)
            return no_update, alert
        
        # Build settings dictionary
        settings_dict = {
            'features': {
                'selected': selected_features,
                'scaling': scaling,
                'handle_missing': missing
            },
            'model_hyperparameters': {
                'classifier': {
                    'n_estimators': int(clf_n_est),
                    'learning_rate': float(clf_lr),
                    'max_depth': int(clf_depth),
                    'min_child_samples': int(clf_min_child),
                    'subsample': float(clf_subsample),
                    'colsample_bytree': float(clf_colsample),
                    'random_state': 42
                },
                'quantile': {
                    'n_estimators': int(qnt_n_est),
                    'learning_rate': float(qnt_lr),
                    'max_depth': int(qnt_depth),
                    'min_child_samples': 20,
                    'random_state': 42
                }
            },
            'thresholds': {
                'quality_A_plus': {
                    'prob_win_min': float(aplus_prob),
                    'R_P50_min': float(aplus_r)
                },
                'quality_A': {
                    'prob_win_min': float(a_prob),
                    'R_P50_min': float(a_r)
                },
                'quality_B': {
                    'prob_win_min': float(b_prob),
                    'R_P50_min': float(b_r)
                },
                'trade_quality_min': trade_min
            },
            'display': {
                'n_bins_reliability': int(n_bins),
                'n_top_features': int(n_top_features),
                'chart_height': int(chart_height),
                'chart_template': chart_template,
                'color_scheme': {
                    'A_plus': color_aplus,
                    'A': color_a,
                    'B': color_b,
                    'C': color_c
                }
            }
        }
        
        # Save to config file
        success, message = save_settings_to_config(settings_dict)
        
        if success:
            alert = dbc.Alert([
                html.I(className="bi bi-check-circle me-2"),
                message,
                html.Br(),
                html.Small("Note: You may need to retrain models for changes to take effect.", 
                          className="text-muted")
            ], color="success", dismissable=True, duration=5000)
            return False, alert  # Close modal and show success
        else:
            alert = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                message
            ], color="danger", dismissable=True, duration=4000)
            return no_update, alert  # Keep modal open and show error
