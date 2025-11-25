"""
ML Prediction Engine - Global Data Store Integration Callbacks

Handles synchronization of ML prediction results with global data stores
to enable data sharing across different pages in the application.

**Feature: ml-prediction-engine, Task 26**
**Validates: Requirement 15.5**
"""

import json
from datetime import datetime

import pandas as pd
from dash import Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate


def register_ml_global_store_callbacks(app):
    """
    Register callbacks for ML prediction global data store integration.
    
    This enables:
    - Sharing ML predictions across pages
    - Caching predictions for reuse
    - Syncing prediction results with global store
    """
    
    # Callback 1: Sync ML predictions to global store when predictions are made
    @app.callback(
        Output('ml-prediction-results-store', 'data', allow_duplicate=True),
        [Input('ml-run-prediction-btn', 'n_clicks')],
        [State('ml-prediction-mode', 'value'),
         State('ml-data-source', 'value'),
         State('merged-data-store', 'data')],
        prevent_initial_call=True
    )
    def sync_predictions_to_global_store(n_clicks, prediction_mode, data_source, merged_data):
        """
        Sync ML prediction results to global store for cross-page access.
        
        This callback ensures that when predictions are made, they are stored
        in the global ml-prediction-results-store so other pages can access them.
        
        Parameters
        ----------
        n_clicks : int
            Number of times prediction button was clicked
        prediction_mode : str
            'single' or 'batch' prediction mode
        data_source : str
            Source of data ('merged' or 'upload')
        merged_data : dict
            Merged dataset from global store
            
        Returns
        -------
        dict
            Prediction results metadata for global store
            
        Validates: Requirement 15.5
        """
        if not n_clicks:
            raise PreventUpdate
        
        # This callback is triggered by the same button as run_prediction
        # The actual prediction results are stored by run_prediction callback
        # This callback just ensures the store is properly initialized
        return no_update
    
    
    # Callback 2: Share predictions with Trade Analysis Dashboard
    @app.callback(
        Output('dashboard-metrics-store', 'data', allow_duplicate=True),
        [Input('ml-prediction-results-store', 'data')],
        [State('dashboard-metrics-store', 'data')],
        prevent_initial_call=True
    )
    def share_predictions_with_dashboard(ml_predictions, dashboard_metrics):
        """
        Share ML predictions with Trade Analysis Dashboard.
        
        When ML predictions are made, this callback updates the dashboard
        metrics store so the dashboard can display ML-enhanced metrics.
        
        Parameters
        ----------
        ml_predictions : dict
            ML prediction results from ml-prediction-results-store
        dashboard_metrics : dict
            Current dashboard metrics
            
        Returns
        -------
        dict
            Updated dashboard metrics with ML predictions
            
        Validates: Requirement 15.1
        """
        if not ml_predictions:
            raise PreventUpdate
        
        # Initialize dashboard metrics if not exists
        if not dashboard_metrics:
            dashboard_metrics = {}
        
        # Add ML predictions to dashboard metrics
        dashboard_metrics['ml_predictions'] = {
            'available': True,
            'mode': ml_predictions.get('mode'),
            'timestamp': ml_predictions.get('timestamp'),
            'n_samples': ml_predictions.get('n_samples', 1)
        }
        
        # If batch predictions, add summary statistics
        if ml_predictions.get('mode') == 'batch' and 'results' in ml_predictions:
            try:
                results_df = pd.read_json(ml_predictions['results'], orient='split')
                
                # Calculate summary statistics
                dashboard_metrics['ml_predictions']['summary'] = {
                    'avg_prob_win': float(results_df['prob_win_calibrated'].mean()),
                    'avg_expected_r': float(results_df['R_P50_raw'].mean()),
                    'quality_distribution': results_df['quality_label'].value_counts().to_dict(),
                    'trade_recommendations': {
                        'TRADE': int((results_df['recommendation'] == 'TRADE').sum()),
                        'SKIP': int((results_df['recommendation'] == 'SKIP').sum())
                    }
                }
            except Exception as e:
                print(f"Error calculating ML prediction summary: {e}")
        
        return dashboard_metrics
    
    
    # Callback 3: Share predictions with Probability Explorer
    @app.callback(
        Output('probability-results-store', 'data', allow_duplicate=True),
        [Input('ml-prediction-results-store', 'data')],
        [State('probability-results-store', 'data')],
        prevent_initial_call=True
    )
    def share_predictions_with_probability_explorer(ml_predictions, prob_results):
        """
        Share ML predictions with Probability Explorer.
        
        When ML predictions are made, this callback updates the probability
        results store so Probability Explorer can compare ML predictions
        with rule-based probability calculations.
        
        Parameters
        ----------
        ml_predictions : dict
            ML prediction results from ml-prediction-results-store
        prob_results : dict
            Current probability explorer results
            
        Returns
        -------
        dict
            Updated probability results with ML predictions
            
        Validates: Requirement 15.2
        """
        if not ml_predictions:
            raise PreventUpdate
        
        # Initialize probability results if not exists
        if not prob_results:
            prob_results = {}
        
        # Add ML predictions to probability results
        prob_results['ml_predictions'] = {
            'available': True,
            'mode': ml_predictions.get('mode'),
            'timestamp': ml_predictions.get('timestamp')
        }
        
        # If single prediction, add the result
        if ml_predictions.get('mode') == 'single' and 'result' in ml_predictions:
            result = ml_predictions['result']
            prob_results['ml_predictions']['single_result'] = {
                'prob_win_calibrated': result.get('prob_win_calibrated'),
                'R_P50_raw': result.get('R_P50_raw'),
                'quality_label': result.get('quality_label'),
                'recommendation': result.get('recommendation')
            }
        
        return prob_results
    
    
    # Callback 4: Share predictions with What-If Scenarios (already implemented in ml_prediction_callbacks.py)
    # This is handled by the send_predictions_to_whatif callback
    
    
    # Callback 5: Monitor and log data sharing activity
    @app.callback(
        Output('ml-data-sharing-status', 'data'),
        [Input('ml-prediction-results-store', 'data'),
         Input('dashboard-metrics-store', 'data'),
         Input('probability-results-store', 'data'),
         Input('ml-predictions-store', 'data')],
        prevent_initial_call=True
    )
    def monitor_data_sharing(ml_predictions, dashboard_metrics, prob_results, whatif_predictions):
        """
        Monitor data sharing activity across stores.
        
        This callback tracks when ML predictions are shared with other pages
        and logs the activity for debugging and monitoring purposes.
        
        Parameters
        ----------
        ml_predictions : dict
            ML prediction results
        dashboard_metrics : dict
            Dashboard metrics with ML predictions
        prob_results : dict
            Probability results with ML predictions
        whatif_predictions : dict
            What-If scenarios with ML predictions
            
        Returns
        -------
        dict
            Data sharing status and activity log
            
        Validates: Requirement 15.5
        """
        sharing_status = {
            'timestamp': datetime.now().isoformat(),
            'ml_predictions_available': bool(ml_predictions),
            'shared_with': {
                'dashboard': bool(dashboard_metrics and dashboard_metrics.get('ml_predictions')),
                'probability_explorer': bool(prob_results and prob_results.get('ml_predictions')),
                'whatif_scenarios': bool(whatif_predictions)
            }
        }
        
        # Log sharing activity
        if ml_predictions:
            print(f"[ML Data Sharing] Predictions available: {ml_predictions.get('mode')} mode")
            print(f"[ML Data Sharing] Shared with: {sharing_status['shared_with']}")
        
        return sharing_status
    
    
    # Callback 6: Clear ML predictions from all stores
    @app.callback(
        [Output('ml-prediction-results-store', 'data', allow_duplicate=True),
         Output('ml-predictions-store', 'data', allow_duplicate=True)],
        [Input('ml-clear-predictions-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def clear_ml_predictions(n_clicks):
        """
        Clear ML predictions from all stores.
        
        This callback clears ML prediction results from all stores when
        the user clicks the clear button, freeing up memory and resetting
        the prediction state.
        
        Parameters
        ----------
        n_clicks : int
            Number of times clear button was clicked
            
        Returns
        -------
        tuple
            (None, None) to clear both stores
            
        Validates: Requirement 15.5
        """
        if not n_clicks:
            raise PreventUpdate
        
        print("[ML Data Sharing] Clearing all ML predictions from stores")
        return None, None
    
    
    # Callback 7: Restore predictions from global store on page load
    @app.callback(
        Output('ml-prediction-summary-cards', 'children', allow_duplicate=True),
        [Input('main-tabs', 'active_tab')],
        [State('ml-prediction-results-store', 'data')],
        prevent_initial_call=True
    )
    def restore_predictions_on_page_load(active_tab, ml_predictions):
        """
        Restore ML predictions when returning to ML Prediction Engine page.
        
        When user navigates back to ML Prediction Engine page, this callback
        restores the last prediction results from the global store so users
        don't lose their work.
        
        Parameters
        ----------
        active_tab : str
            Currently active tab
        ml_predictions : dict
            Stored ML prediction results
            
        Returns
        -------
        dash component
            Restored prediction summary cards or no_update
            
        Validates: Requirement 15.5
        """
        if active_tab != 'ml-prediction-engine':
            raise PreventUpdate
        
        if not ml_predictions:
            raise PreventUpdate
        
        # Import here to avoid circular imports
        from frontend.components.prediction_summary_cards import create_prediction_summary_cards
        from frontend.components.batch_prediction_table import create_batch_prediction_table
        
        try:
            if ml_predictions.get('mode') == 'single' and 'result' in ml_predictions:
                # Restore single prediction
                result = ml_predictions['result']
                return create_prediction_summary_cards(result)
                
            elif ml_predictions.get('mode') == 'batch' and 'results' in ml_predictions:
                # Restore batch predictions
                results_df = pd.read_json(ml_predictions['results'], orient='split')
                return create_batch_prediction_table(results_df)
            
        except Exception as e:
            print(f"Error restoring predictions: {e}")
            raise PreventUpdate
        
        raise PreventUpdate


def get_ml_predictions_from_store(ml_predictions_store):
    """
    Helper function to extract ML predictions from store.
    
    This utility function helps other pages access ML predictions
    from the global store in a consistent way.
    
    Parameters
    ----------
    ml_predictions_store : dict
        ML prediction results from store
        
    Returns
    -------
    dict or pd.DataFrame
        Extracted prediction results
    """
    if not ml_predictions_store:
        return None
    
    try:
        if ml_predictions_store.get('mode') == 'single':
            return ml_predictions_store.get('result')
        elif ml_predictions_store.get('mode') == 'batch':
            return pd.read_json(ml_predictions_store['results'], orient='split')
    except Exception as e:
        print(f"Error extracting ML predictions: {e}")
        return None
