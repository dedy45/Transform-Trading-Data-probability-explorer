"""
Cache Control Callbacks

Callbacks for managing server-side data cache:
- Display cache status
- Reload data functionality
- Clear cache functionality
- Show cache information
"""

from dash import Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import html

from backend.utils.data_cache import (
    has_cached_data,
    get_cache_info,
    clear_all_data,
    get_merged_data,
    get_trade_data
)
from frontend.components.cache_control_panel import (
    create_cache_status_badge,
    create_cache_info_alert
)


def register_cache_control_callbacks(app):
    """
    Register all cache control callbacks.
    
    Args:
        app: Dash application instance
    """
    
    # Callback 1: Update cache status display
    @app.callback(
        [
            Output('cache-status-indicator', 'children'),
            Output('reload-data-btn', 'disabled'),
            Output('clear-cache-btn', 'disabled'),
            Output('show-cache-info-btn', 'disabled')
        ],
        [
            Input('merged-data-store', 'data'),
            Input('cache-reload-trigger', 'data'),
            Input('cache-clear-trigger', 'data')
        ],
        prevent_initial_call=False
    )
    def update_cache_status(merged_data, reload_trigger, clear_trigger):
        """Update cache status indicator and button states."""
        try:
            # Check if data is cached on server
            has_data = has_cached_data()
            
            if has_data:
                # Get cache info
                cache_info = get_cache_info()
                summary = cache_info.get('summary', {})
                
                # Get merged data info
                merged_info = summary.get('merged_data', {})
                
                # Create status badge
                status_badge = create_cache_status_badge(True, merged_info)
                
                # Enable buttons
                return status_badge, False, False, False
            else:
                # No data cached
                status_badge = create_cache_status_badge(False)
                
                # Disable buttons
                return status_badge, True, True, True
                
        except Exception as e:
            print(f"Error updating cache status: {e}")
            return no_update, no_update, no_update, no_update
    
    # Callback 2: Show cache info
    @app.callback(
        Output('cache-info-display', 'children'),
        [Input('show-cache-info-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def show_cache_info(n_clicks):
        """Display detailed cache information."""
        if not n_clicks:
            raise PreventUpdate
        
        try:
            cache_info = get_cache_info()
            
            if not cache_info.get('has_data'):
                return html.Small("No data cached", className="text-muted")
            
            # Create info alert
            info_alert = create_cache_info_alert(cache_info)
            return info_alert
            
        except Exception as e:
            print(f"Error showing cache info: {e}")
            return dbc.Alert(f"Error: {str(e)}", color="danger")
    
    # Callback 3: Open reload confirmation modal
    @app.callback(
        Output('reload-confirmation-modal', 'is_open'),
        [
            Input('reload-data-btn', 'n_clicks'),
            Input('reload-confirm-btn', 'n_clicks'),
            Input('reload-cancel-btn', 'n_clicks')
        ],
        [State('reload-confirmation-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_reload_modal(reload_click, confirm_click, cancel_click, is_open):
        """Toggle reload confirmation modal."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'reload-data-btn':
            return True
        elif trigger_id in ['reload-confirm-btn', 'reload-cancel-btn']:
            return False
        
        return is_open
    
    # Callback 4: REMOVED - upload-trade-csv component no longer exists
    # Data loading is now handled by global data loader in app.py
    
    # Callback 5: Handle reload status message (separate callback to avoid duplicates)
    # REMOVED - This conflicts with trade_analysis_dashboard_callbacks.py
    # The trade analysis dashboard handles upload status messages
    # @app.callback(
    #     Output('upload-status', 'children'),
    #     [Input('reload-confirm-btn', 'n_clicks')],
    #     prevent_initial_call=True
    # )
    # def handle_reload_confirm_status(n_clicks):
    #     """Show status message when reload is confirmed."""
    #     if not n_clicks:
    #         raise PreventUpdate
    #     return dbc.Alert([
    #         html.I(className="fas fa-info-circle me-2"),
    #         "Ready to load new data. Please upload CSV files."
    #     ], color="info")
    
    # Callback 5: Handle reload trigger (separate callback to avoid duplicates)
    @app.callback(
        Output('cache-reload-trigger', 'data'),
        [Input('reload-confirm-btn', 'n_clicks')],
        [State('cache-reload-trigger', 'data')],
        prevent_initial_call=True
    )
    def handle_reload_confirm_trigger(n_clicks, current_trigger):
        """Increment reload trigger when reload is confirmed."""
        if not n_clicks:
            raise PreventUpdate
        return (current_trigger or 0) + 1
    
    # Callback 5: Open clear cache confirmation modal
    @app.callback(
        Output('clear-cache-modal', 'is_open'),
        [
            Input('clear-cache-btn', 'n_clicks'),
            Input('clear-confirm-btn', 'n_clicks'),
            Input('clear-cancel-btn', 'n_clicks')
        ],
        [State('clear-cache-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_clear_modal(clear_click, confirm_click, cancel_click, is_open):
        """Toggle clear cache confirmation modal."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'clear-cache-btn':
            return True
        elif trigger_id in ['clear-confirm-btn', 'clear-cancel-btn']:
            return False
        
        return is_open
    
    # Callback 6: Handle clear cache confirmation (avoid duplicate outputs)
    @app.callback(
        [
            Output('merged-data-store', 'clear_data'),
            Output('global-data-store', 'clear_data')
        ],
        [Input('clear-confirm-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def handle_clear_confirm_clear_stores(n_clicks):
        """Clear browser stores when cache is cleared."""
        if not n_clicks:
            raise PreventUpdate
        return True, True
    
    # Callback 7: Handle clear cache trigger (separate callback to avoid duplicates)
    @app.callback(
        Output('cache-clear-trigger', 'data'),
        [Input('clear-confirm-btn', 'n_clicks')],
        [State('cache-clear-trigger', 'data')],
        prevent_initial_call=True
    )
    def handle_clear_confirm_trigger(n_clicks, current_trigger):
        """Increment clear cache trigger and clear server cache."""
        if not n_clicks:
            raise PreventUpdate
        
        try:
            # Clear server-side cache
            clear_all_data()
            print("[OK] Cache cleared successfully")
            
            # Increment trigger
            return (current_trigger or 0) + 1
            
        except Exception as e:
            print(f"Error clearing cache: {e}")
            raise PreventUpdate
    
    # Callback 8: Auto-restore data from server cache on page load
    # REMOVED - This conflicts with trade_analysis_dashboard_callbacks.py
    # The trade analysis dashboard handles auto-restore status messages
    # @app.callback(
    #     [
    #         # Output('merged-data-store', 'data'),  # REMOVED - Conflicts with trade analysis dashboard
    #         Output('auto-restore-status', 'children')
    #     ],
    #     [Input('url', 'pathname')],
    #     prevent_initial_call=False
    # )
    

    # def auto_restore_from_cache(pathname):
    #     """Automatically restore data from server cache on page load."""
    #     try:
    #         # Check if data exists in server cache
    #         if has_cached_data():
    #             # Get merged data from server cache
    #             merged_df = get_merged_data()
    #             
    #             if merged_df is not None and not merged_df.empty:
    #                 # Convert to dict for browser storage
    #                 data_dict = merged_df.to_dict('records')
    #                 
    #                 status_msg = dbc.Alert([
    #                     html.I(className="fas fa-check-circle me-2"),
    #                     f"Data restored from server cache: {len(merged_df):,} trades"
    #                 ], color="success", dismissable=True)
    #                 
    #                 print(f"[OK] Auto-restored {len(merged_df)} trades from server cache")
    #                 
    #                 # return data_dict, status_msg  # REMOVED data_dict output
    #                 return status_msg
    #         
    #         # No data to restore
    #         return no_update
    #         
    #     except Exception as e:
    #         print(f"Error auto-restoring from cache: {e}")
    #         return no_update
    
    print("[OK] Cache control callbacks registered")


print("[OK] Cache control callbacks module loaded")
