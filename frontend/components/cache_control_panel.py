"""
Cache Control Panel Component

Provides UI controls for managing server-side data cache:
- Cache status display
- Reload data button
- Clear cache button
- Cache info display
"""

import dash_bootstrap_components as dbc
from dash import html, dcc


def create_cache_control_panel():
    """
    Create cache control panel with status and action buttons.
    
    Returns:
        dbc.Card: Cache control panel component
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-database me-2"),
                "Data Cache Control"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            # Cache Status
            dbc.Row([
                dbc.Col([
                    html.Div(id="cache-status-indicator", children=[
                        dbc.Badge("No Data Loaded", color="secondary", className="me-2"),
                        html.Small("Load data to begin analysis", className="text-muted")
                    ])
                ], width=12)
            ], className="mb-3"),
            
            # Cache Info
            dbc.Row([
                dbc.Col([
                    html.Div(id="cache-info-display", children=[
                        html.Small("Cache info will appear here after loading data", 
                                 className="text-muted")
                    ])
                ], width=12)
            ], className="mb-3"),
            
            # Action Buttons
            dbc.Row([
                dbc.Col([
                    dbc.Button([
                        html.I(className="fas fa-sync-alt me-2"),
                        "Reload Data"
                    ], 
                    id="reload-data-btn",
                    color="primary",
                    size="sm",
                    className="w-100",
                    disabled=True)
                ], width=4),
                dbc.Col([
                    dbc.Button([
                        html.I(className="fas fa-trash-alt me-2"),
                        "Clear Cache"
                    ], 
                    id="clear-cache-btn",
                    color="warning",
                    size="sm",
                    className="w-100",
                    disabled=True)
                ], width=4),
                dbc.Col([
                    dbc.Button([
                        html.I(className="fas fa-info-circle me-2"),
                        "Cache Info"
                    ], 
                    id="show-cache-info-btn",
                    color="info",
                    size="sm",
                    className="w-100",
                    disabled=True)
                ], width=4),
            ]),
            
            # Hidden stores for cache management
            dcc.Store(id="cache-reload-trigger", data=0),
            dcc.Store(id="cache-clear-trigger", data=0),
        ])
    ], className="mb-3")


def create_cache_status_badge(has_data: bool, data_info: dict = None):
    """
    Create cache status badge based on current state.
    
    Args:
        has_data: Whether data is cached
        data_info: Dictionary with cache information
        
    Returns:
        html.Div: Status badge and info
    """
    if not has_data:
        return html.Div([
            dbc.Badge("No Data Cached", color="secondary", className="me-2"),
            html.Small("Upload CSV to load data", className="text-muted")
        ])
    
    # Extract info
    rows = data_info.get('rows', 0) if data_info else 0
    memory_mb = data_info.get('memory_mb', 0) if data_info else 0
    timestamp = data_info.get('timestamp', 'unknown') if data_info else 'unknown'
    
    return html.Div([
        dbc.Badge("Data Cached", color="success", className="me-2"),
        html.Small([
            f"{rows:,} trades | ",
            f"{memory_mb:.1f} MB | ",
            f"Loaded: {timestamp[:19] if timestamp != 'unknown' else 'unknown'}"
        ], className="text-muted")
    ])


def create_cache_info_alert(cache_summary: dict):
    """
    Create detailed cache information alert.
    
    Args:
        cache_summary: Dictionary with cache summary
        
    Returns:
        dbc.Alert: Cache information alert
    """
    if not cache_summary:
        return dbc.Alert("No cache data available", color="info", className="mb-0")
    
    total_memory = cache_summary.get('total_memory_mb', 0)
    cached_keys = cache_summary.get('cached_keys', [])
    summary = cache_summary.get('summary', {})
    
    info_items = []
    for key, data in summary.items():
        info_items.append(html.Li([
            html.Strong(f"{key}: "),
            f"{data.get('rows', 0):,} rows × {data.get('cols', 0)} cols ",
            f"({data.get('memory_mb', 0):.2f} MB)"
        ]))
    
    return dbc.Alert([
        html.H6("Cached Data Summary", className="alert-heading"),
        html.Hr(),
        html.Ul(info_items, className="mb-2"),
        html.Hr(),
        html.P([
            html.Strong("Total Memory: "),
            f"{total_memory:.2f} MB"
        ], className="mb-0")
    ], color="info", className="mb-0")


def create_reload_confirmation_modal():
    """
    Create modal for reload confirmation.
    
    Returns:
        dbc.Modal: Reload confirmation modal
    """
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Reload Data")),
        dbc.ModalBody([
            html.P("Are you sure you want to reload the data?"),
            html.P([
                "This will:",
                html.Ul([
                    html.Li("Keep current data in cache"),
                    html.Li("Allow you to upload new CSV files"),
                    html.Li("Replace cached data with new data")
                ])
            ]),
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "Current analysis results will be cleared. Make sure to export any important results first."
            ], color="warning")
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="reload-cancel-btn", color="secondary", className="me-2"),
            dbc.Button("Reload Data", id="reload-confirm-btn", color="primary")
        ])
    ], id="reload-confirmation-modal", is_open=False)


def create_clear_cache_modal():
    """
    Create modal for clear cache confirmation.
    
    Returns:
        dbc.Modal: Clear cache confirmation modal
    """
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Clear Cache")),
        dbc.ModalBody([
            html.P("Are you sure you want to clear all cached data?"),
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                html.Strong("Warning: "),
                "This will permanently remove all loaded data from server memory. ",
                "You will need to upload CSV files again."
            ], color="danger")
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="clear-cancel-btn", color="secondary", className="me-2"),
            dbc.Button("Clear Cache", id="clear-confirm-btn", color="danger")
        ])
    ], id="clear-cache-modal", is_open=False)


print("[OK] Cache control panel component loaded")
