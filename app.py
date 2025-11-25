"""
Trading Probability Explorer - Main Application

This is the main entry point for the Trading Probability Explorer application.
It sets up the Dash app structure with multiple analysis tabs, global data stores,
and comprehensive error handling.
"""
import sys
import os
import base64
import io
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import traceback

# Check for production mode flag
if '--production' in sys.argv or '--prod' in sys.argv:
    from config.prod_config import DASH_HOST, DASH_PORT, DASH_DEBUG, DASH_AUTO_OPEN_BROWSER, DATA_STORAGE_TYPE
else:
    from config.config import DASH_HOST, DASH_PORT, DASH_DEBUG, DASH_AUTO_OPEN_BROWSER
    try:
        from config.config import DATA_STORAGE_TYPE
    except ImportError:
        DATA_STORAGE_TYPE = 'session'  # Default to session storage

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Analisis Probabilitas Trading"
)

app.title = "Analisis Probabilitas Trading"
server = app.server  # Expose server for deployment

# Function to create initial tab content
def create_initial_tab_content():
    """Create initial tab content (Trade Analysis Dashboard)"""
    try:
        from frontend.layouts.trade_analysis_dashboard_layout import create_trade_analysis_dashboard_layout
        return create_trade_analysis_dashboard_layout()
    except Exception as e:
        print(f"Error loading initial tab: {e}")
        return html.Div([
            html.H4("Error Loading Dashboard", className="text-center mt-5 text-danger"),
            html.P(str(e), className="text-center text-muted")
        ])

# Dynamic layout function
def serve_layout():
    """Generate layout dynamically on each page load"""
    import os
    # FIX: Path dataraw sekarang benar - di folder yang sama dengan app.py
    dataraw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataraw'))
    trade_files = []
    feature_files = []
    try:
        if os.path.isdir(dataraw_dir):
            for name in os.listdir(dataraw_dir):
                path = os.path.join(dataraw_dir, name)
                if name.lower().endswith('.csv') and os.path.isfile(path):
                    if ('ea_' in name.lower()) or ('bt_' in name.lower()) or ('trade' in name.lower()) or ('swinghl' in name.lower()) or ('mscandle' in name.lower()):
                        trade_files.append(path)
                    if ('feature' in name.lower()) or ('market_features' in name.lower()):
                        feature_files.append(path)
            # Scan market-fitur subfolder for feature files
            market_fitur_dir = os.path.join(dataraw_dir, 'market-fitur')
            if os.path.isdir(market_fitur_dir):
                for name in os.listdir(market_fitur_dir):
                    path = os.path.join(market_fitur_dir, name)
                    if name.lower().endswith('.csv') and os.path.isfile(path):
                        feature_files.append(path)
    except Exception as e:
        print(f"Error scanning dataraw directory: {e}")
    trade_options = [{'label': os.path.basename(p), 'value': p} for p in sorted(trade_files)]
    feature_options = [{'label': os.path.basename(p), 'value': p} for p in sorted(feature_files)]
    print(f"[INFO] Found {len(trade_files)} trade files and {len(feature_files)} feature files in {dataraw_dir}")
    return dbc.Container([
        # Header section
        dbc.Row([
            dbc.Col([
                html.H1("Analisis Probabilitas Trading", className="text-center my-4"),
                html.P(
                    "Analisis probabilitas komprehensif untuk optimasi trading",
                    className="text-center text-muted mb-4"
                )
            ])
        ]),
        
        # Error alert (hidden by default)
        dbc.Row([
            dbc.Col([
                dbc.Alert(
                    id="global-error-alert",
                    children="",
                    color="danger",
                    is_open=False,
                    dismissable=True,
                    className="mb-3"
                )
            ])
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Global: Pilih Trade CSV (dataraw)", className="fw-bold"),
                                dcc.Dropdown(
                                    id='global-trade-file-dropdown',
                                    options=trade_options,
                                    value=(trade_options[0]['value'] if trade_options else None),
                                    placeholder='Pilih file trade dari dataraw',
                                    clearable=True
                                ),
                                html.Div(id="global-data-status-trade", className="mt-2")
                            ], md=4),
                            dbc.Col([
                                html.Label("Global: Pilih Feature CSV (dataraw)", className="fw-bold"),
                                dcc.Dropdown(
                                    id='global-feature-file-dropdown',
                                    options=feature_options,
                                    value=(feature_options[0]['value'] if feature_options else None),
                                    placeholder='Pilih file feature dari dataraw',
                                    clearable=True
                                ),
                                html.Div(id="global-data-status-feature", className="mt-2")
                            ], md=4),
                            dbc.Col([
                                html.Label("Global: Muat Data", className="fw-bold"),
                                dbc.Button([
                                    html.I(className="bi bi-database me-2"),
                                    "Muat Data Terpilih"
                                ], id="load-selected-btn-global", color="info", outline=True, className="me-2"),
                                html.Div(id="global-data-status-sample", className="mt-2")
                            ], md=4)
                        ])
                    ])
                ])
            ])
        ], className="mb-3"),
        
        # Panel Control
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6([
                            html.I(className="bi bi-sliders me-2"),
                            "Panel Kontrol"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="cache-status-indicator", children=[
                                    dbc.Badge("Belum Ada Data", color="secondary", className="me-2"),
                                    html.Small("Muat data untuk memulai analisis", className="text-muted")
                                ])
                            ], md=5),
                            dbc.Col([
                                dbc.ButtonGroup([
                                    dbc.Button([
                                        html.I(className="bi bi-arrow-clockwise me-2"),
                                        "Muat Ulang"
                                    ], id="reload-data-btn", color="primary", size="sm", disabled=True),
                                    dbc.Button([
                                        html.I(className="bi bi-trash me-2"),
                                        "Hapus Cache"
                                    ], id="clear-cache-btn", color="warning", size="sm", disabled=True),
                                    dbc.Button([
                                        html.I(className="bi bi-info-circle me-2"),
                                        "Info"
                                    ], id="show-cache-info-btn", color="info", size="sm", disabled=True)
                                ], className="float-end")
                            ], md=7)
                        ]),
                        html.Div(id="cache-info-display", className="mt-2")
                    ])
                ])
            ])
        ], className="mb-3"),
        
        # Loading indicator for tab switching
        dcc.Loading(
            id="tab-loading",
            type="default",
            children=[
                # Tabs for different analysis modules
                dbc.Tabs([
                    dbc.Tab(
                        label="Dashboard Analisis Trading", 
                        tab_id="trade-analysis",
                        tab_style={"marginLeft": "auto"}
                    ),
                    dbc.Tab(label="Eksplorasi Probabilitas", tab_id="probability-explorer"),
                    dbc.Tab(label="Auto Feature Selection", tab_id="auto-feature-selection"),
                    dbc.Tab(label="Analisis Sekuensial", tab_id="sequential-analysis"),
                    dbc.Tab(label="Lab Kalibrasi", tab_id="calibration-lab"),
                    dbc.Tab(label="Eksplorasi Regime", tab_id="regime-explorer"),
                    dbc.Tab(label="Skenario What-If", tab_id="what-if-scenarios"),
                    dbc.Tab(label="ML Prediction Engine", tab_id="ml-prediction-engine"),
                ], id="main-tabs", active_tab="trade-analysis"),
            ]
        ),
        
        # Auto-restore status message
        html.Div(id="auto-restore-status", className="mt-2"),
        
        # Content area for active tab with loading state
        dcc.Loading(
            id="content-loading",
            type="circle",
            children=[
                html.Div(id="tab-content", className="mt-4", children=[
                    create_initial_tab_content()  # Load initial content directly
                ])
            ]
        ),
    
    # URL component for page load detection
    dcc.Location(id='url', refresh=False),
    
    # Global data stores - using DATA_STORAGE_TYPE to persist data
    # 'session' = data persists during browser session (recommended)
    # 'local' = data persists even after browser close
    # 'memory' = data lost on page refresh (not recommended)
    dcc.Store(id="global-data-store", storage_type=DATA_STORAGE_TYPE),
    dcc.Store(id="feature-data-store", storage_type=DATA_STORAGE_TYPE),
    dcc.Store(id="trade-data-store", storage_type=DATA_STORAGE_TYPE),
    dcc.Store(id="merged-data-store", storage_type=DATA_STORAGE_TYPE),
    
    # Filter and scenario stores - persist during session
    dcc.Store(id="filter-state-store", storage_type=DATA_STORAGE_TYPE),
    dcc.Store(id="scenario-state-store", storage_type=DATA_STORAGE_TYPE),
    
    # Cache management stores
    dcc.Store(id="cache-reload-trigger", data=0),
    dcc.Store(id="cache-clear-trigger", data=0),
    
    # Analysis results cache - persist during session
    dcc.Store(id="probability-results-store", storage_type=DATA_STORAGE_TYPE),
    dcc.Store(id="dashboard-metrics-store", storage_type=DATA_STORAGE_TYPE),
    
    # New integration stores - persist during session
    dcc.Store(id="expectancy-results-store", storage_type=DATA_STORAGE_TYPE),
    dcc.Store(id="mae-mfe-results-store", storage_type=DATA_STORAGE_TYPE),
    dcc.Store(id="monte-carlo-results-store", storage_type=DATA_STORAGE_TYPE),
    dcc.Store(id="composite-score-results-store", storage_type=DATA_STORAGE_TYPE),
    
    # ML Prediction Engine stores - persist during session
    dcc.Store(id="ml-prediction-results-store", storage_type=DATA_STORAGE_TYPE),
    dcc.Store(id="ml-predictions-store", storage_type=DATA_STORAGE_TYPE),  # For What-If integration
    
    # Error state store - memory only (temporary)
    dcc.Store(id="error-state-store", storage_type='memory'),
    
    # Cache Control Modals
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Muat Ulang Data")),
        dbc.ModalBody([
            html.P("Apakah Anda yakin ingin memuat ulang data?"),
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                "Hasil analisis saat ini akan dihapus."
            ], color="warning")
        ]),
        dbc.ModalFooter([
            dbc.Button("Batal", id="reload-cancel-btn", color="secondary", className="me-2"),
            dbc.Button("Muat Ulang Data", id="reload-confirm-btn", color="primary")
        ])
    ], id="reload-confirmation-modal", is_open=False),
    
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Hapus Cache")),
        dbc.ModalBody([
            html.P("Apakah Anda yakin ingin menghapus semua data cache?"),
            dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                html.Strong("Peringatan: "),
                "Ini akan menghapus semua data yang dimuat dari memori server secara permanen."
            ], color="danger")
        ]),
        dbc.ModalFooter([
            dbc.Button("Batal", id="clear-cancel-btn", color="secondary", className="me-2"),
            dbc.Button("Hapus Cache", id="clear-confirm-btn", color="danger")
        ])
    ], id="clear-cache-modal", is_open=False),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(className="mt-5"),
            html.P(
                "Analisis Probabilitas Trading v1.0 | Dibangun dengan Dash & Plotly",
                className="text-center text-muted small mb-4"
            )
        ])
    ])
    
], fluid=True)

# Set dynamic layout function
app.layout = serve_layout


# Tab content callback with error handling (simplified for tab switching only)
@app.callback(
    [Output('tab-content', 'children'),
     Output('global-error-alert', 'children'),
     Output('global-error-alert', 'is_open')],
    [Input('main-tabs', 'active_tab')],
    prevent_initial_call=True  # Only trigger on tab changes, not initial load
)
def render_tab_content(active_tab):
    """
    Render content for active tab with comprehensive error handling.
    
    Parameters
    ----------
    active_tab : str
        ID of the active tab
        
    Returns
    -------
    tuple
        (tab_content, error_message, error_is_open)
    """
    print(f"\n=== Rendering tab: {active_tab} ===")
    
    # Handle None case (shouldn't happen but just in case)
    if active_tab is None:
        active_tab = 'trade-analysis'
        
    # Skip if initial call (content already loaded)
    if ctx.triggered_id is None:
        print("Initial call skipped - content already loaded")
        return dash.no_update, "", False
    
    try:
        if active_tab == 'trade-analysis':
            print("Loading Trade Analysis Dashboard...")
            from frontend.layouts.trade_analysis_dashboard_layout import create_trade_analysis_dashboard_layout
            return create_trade_analysis_dashboard_layout(), "", False
            
        elif active_tab == 'probability-explorer':
            from frontend.layouts.probability_explorer_layout import create_probability_explorer_layout
            return create_probability_explorer_layout(), "", False
            
        elif active_tab == 'sequential-analysis':
            from frontend.layouts.sequential_analysis_layout import create_sequential_analysis_layout
            return create_sequential_analysis_layout(), "", False
            
        elif active_tab == 'calibration-lab':
            from frontend.layouts.calibration_lab_layout import create_calibration_lab_layout
            return create_calibration_lab_layout(), "", False
            
        elif active_tab == 'regime-explorer':
            from frontend.layouts.regime_explorer_layout import create_regime_explorer_layout
            return create_regime_explorer_layout(), "", False
            
        elif active_tab == 'what-if-scenarios':
            from frontend.layouts.whatif_scenarios_layout import create_whatif_scenarios_layout
            return create_whatif_scenarios_layout(), "", False
            
        elif active_tab == 'auto-feature-selection':
            from frontend.layouts.auto_feature_selection_layout import create_auto_feature_selection_layout
            return create_auto_feature_selection_layout(), "", False
            
        elif active_tab == 'ml-prediction-engine':
            from frontend.layouts.ml_prediction_engine_layout import create_ml_prediction_engine_layout
            return create_ml_prediction_engine_layout(), "", False
        
        # Default fallback
        return html.Div([
            html.H4("Selamat Datang di Analisis Probabilitas Trading", className="text-center mt-5"),
            html.P("Pilih tab di atas untuk memulai analisis", className="text-center text-muted")
        ]), "", False
        
    except ImportError as e:
        error_msg = f"Gagal memuat tab '{active_tab}': {str(e)}"
        print(f"Import Error: {error_msg}")
        return html.Div([
            html.H4("Modul Tidak Tersedia", className="text-center mt-5 text-danger"),
            html.P(f"Modul '{active_tab}' belum diimplementasikan atau ada dependensi yang hilang.", 
                   className="text-center")
        ]), error_msg, True
        
    except Exception as e:
        error_msg = f"Error memuat tab '{active_tab}': {str(e)}"
        print(f"Error: {error_msg}")
        print(traceback.format_exc())
        return html.Div([
            html.H4("Error Memuat Tab", className="text-center mt-5 text-danger"),
            html.P("Terjadi error yang tidak terduga. Silakan periksa console untuk detail.", 
                   className="text-center")
        ]), error_msg, True


# Register all callbacks with error handling
def register_all_callbacks():
    """
    Register all application callbacks with comprehensive error handling.
    
    This function attempts to register callbacks for all modules.
    If a module is not available, it logs a warning but continues.
    """
    callbacks_registered = []
    callbacks_failed = []
    
    # Do not clear existing callbacks to preserve tab switching callback
    
    # Trade Analysis Dashboard callbacks
    try:
        from frontend.callbacks.trade_analysis_dashboard_callbacks import register_trade_analysis_callbacks
        register_trade_analysis_callbacks(app)
        callbacks_registered.append("Trade Analysis Dashboard")
    except Exception as e:
        callbacks_failed.append(("Trade Analysis Dashboard", str(e)))
        print(f"Warning: Failed to register Trade Analysis Dashboard callbacks: {e}")
    
    # Probability Explorer callbacks
    try:
        from frontend.callbacks.probability_explorer_callbacks import register_probability_explorer_callbacks
        register_probability_explorer_callbacks(app)
        callbacks_registered.append("Probability Explorer")
    except Exception as e:
        callbacks_failed.append(("Probability Explorer", str(e)))
        print(f"Warning: Failed to register Probability Explorer callbacks: {e}")
    
    # What-If Scenarios callbacks
    try:
        from frontend.callbacks.whatif_scenarios_callbacks import register_whatif_scenarios_callbacks
        register_whatif_scenarios_callbacks(app)
        callbacks_registered.append("What-If Scenarios")
    except Exception as e:
        callbacks_failed.append(("What-If Scenarios", str(e)))
        print(f"Warning: Failed to register What-If Scenarios callbacks: {e}")
    
    # Interactive Filter callbacks
    try:
        from frontend.callbacks.interactive_filter_callbacks import register_interactive_filter_callbacks
        register_interactive_filter_callbacks(app)
        callbacks_registered.append("Interactive Filter")
    except Exception as e:
        callbacks_failed.append(("Interactive Filter", str(e)))
        print(f"Warning: Failed to register Interactive Filter callbacks: {e}")
    
    # Sequential Analysis callbacks
    try:
        from frontend.callbacks.sequential_analysis_callbacks import register_sequential_analysis_callbacks
        register_sequential_analysis_callbacks(app)
        callbacks_registered.append("Sequential Analysis")
    except Exception as e:
        callbacks_failed.append(("Sequential Analysis", str(e)))
        print(f"Warning: Failed to register Sequential Analysis callbacks: {e}")
    
    # Calibration Lab callbacks
    try:
        from frontend.callbacks.calibration_lab_callbacks import register_calibration_lab_callbacks
        register_calibration_lab_callbacks(app)
        callbacks_registered.append("Calibration Lab")
    except Exception as e:
        callbacks_failed.append(("Calibration Lab", str(e)))
        print(f"Warning: Failed to register Calibration Lab callbacks: {e}")
    
    # Regime Explorer callbacks
    try:
        from frontend.callbacks.regime_explorer_callbacks import register_regime_explorer_callbacks
        register_regime_explorer_callbacks(app)
        callbacks_registered.append("Regime Explorer")
    except Exception as e:
        callbacks_failed.append(("Regime Explorer", str(e)))
        print(f"Warning: Failed to register Regime Explorer callbacks: {e}")
    
    # Cache Control callbacks
    try:
        from frontend.callbacks.cache_control_callbacks import register_cache_control_callbacks
        register_cache_control_callbacks(app)
        callbacks_registered.append("Cache Control")
    except Exception as e:
        callbacks_failed.append(("Cache Control", str(e)))
        print(f"Warning: Failed to register Cache Control callbacks: {e}")
    
    # Placeholder callbacks (for components without data)
    try:
        from frontend.callbacks.placeholder_callbacks import register_placeholder_callbacks
        register_placeholder_callbacks(app)
        callbacks_registered.append("Placeholder Callbacks")
    except Exception as e:
        callbacks_failed.append(("Placeholder Callbacks", str(e)))
        print(f"Warning: Failed to register Placeholder callbacks: {e}")
    
    # New integration callbacks
    try:
        from frontend.callbacks.expectancy_callbacks import register_expectancy_callbacks
        register_expectancy_callbacks(app)
        callbacks_registered.append("Expectancy Analysis")
    except Exception as e:
        callbacks_failed.append(("Expectancy Analysis", str(e)))
        print(f"Warning: Failed to register Expectancy callbacks: {e}")
    
    try:
        from frontend.callbacks.mae_mfe_callbacks import register_mae_mfe_callbacks
        register_mae_mfe_callbacks(app)
        callbacks_registered.append("MAE/MFE Optimizer")
    except Exception as e:
        callbacks_failed.append(("MAE/MFE Optimizer", str(e)))
        print(f"Warning: Failed to register MAE/MFE callbacks: {e}")
    
    try:
        from frontend.callbacks.monte_carlo_callbacks import register_monte_carlo_callbacks
        register_monte_carlo_callbacks(app)
        callbacks_registered.append("Monte Carlo Simulation")
    except Exception as e:
        callbacks_failed.append(("Monte Carlo Simulation", str(e)))
        print(f"Warning: Failed to register Monte Carlo callbacks: {e}")
    
    try:
        from frontend.callbacks.composite_score_callbacks import register_composite_score_callbacks
        register_composite_score_callbacks(app)
        callbacks_registered.append("Composite Score Analysis")
    except Exception as e:
        callbacks_failed.append(("Composite Score Analysis", str(e)))
        print(f"Warning: Failed to register Composite Score callbacks: {e}")
    
    try:
        from frontend.callbacks.auto_feature_selection_callbacks import register_auto_feature_selection_callbacks
        register_auto_feature_selection_callbacks(app)
        callbacks_registered.append("Auto Feature Selection")
    except Exception as e:
        callbacks_failed.append(("Auto Feature Selection", str(e)))
        print(f"Warning: Failed to register Auto Feature Selection callbacks: {e}")
    
    try:
        from frontend.callbacks.ml_settings_callbacks import register_ml_settings_callbacks
        register_ml_settings_callbacks(app)
        callbacks_registered.append("ML Settings")
    except Exception as e:
        callbacks_failed.append(("ML Settings", str(e)))
        print(f"Warning: Failed to register ML Settings callbacks: {e}")
    
    try:
        from frontend.callbacks.ml_prediction_callbacks import register_ml_prediction_callbacks
        register_ml_prediction_callbacks(app)
        callbacks_registered.append("ML Prediction Engine")
    except Exception as e:
        callbacks_failed.append(("ML Prediction Engine", str(e)))
        print(f"Warning: Failed to register ML Prediction Engine callbacks: {e}")
    
    try:
        from frontend.callbacks.ml_global_store_callbacks import register_ml_global_store_callbacks
        register_ml_global_store_callbacks(app)
        callbacks_registered.append("ML Global Store Integration")
    except Exception as e:
        callbacks_failed.append(("ML Global Store Integration", str(e)))
        print(f"Warning: Failed to register ML Global Store callbacks: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("CALLBACK REGISTRATION SUMMARY")
    print("="*60)
    print(f"Successfully registered: {len(callbacks_registered)} modules")
    for module in callbacks_registered:
        print(f"  [OK] {module}")
    
    if callbacks_failed:
        print(f"\nFailed to register: {len(callbacks_failed)} modules")
        for module, error in callbacks_failed:
            print(f"  [FAIL] {module}: {error}")
    
    print("="*60 + "\n")

# Register all callbacks
register_all_callbacks()


 

@app.callback(
    [Output('global-data-status-sample', 'children'),
     Output('global-data-status-trade', 'children'),
     Output('global-data-status-feature', 'children'),
     Output('global-data-store', 'data'),
     Output('trade-data-store', 'data'),
     Output('merged-data-store', 'data')],
    [Input('load-selected-btn-global', 'n_clicks')],
    [State('global-trade-file-dropdown', 'value'),
     State('global-feature-file-dropdown', 'value')],
    prevent_initial_call=True,
    running=[(Output('load-selected-btn-global', 'disabled'), True, False)]
)
def handle_global_data_controls(load_clicks, trade_path, feature_path):
    """Handle global data loading with timeout protection"""
    from backend.utils.timeout_utils import timeout_context, TIMEOUT_CSV_LOAD, TIMEOUT_DATA_MERGE
    
    try:
        from dash import html
        import dash_bootstrap_components as dbc
        from backend.utils.data_cache import store_trade_data, store_merged_data
        from backend.utils.data_preprocessor import load_trade_csv, load_feature_csv, merge_datasets, filter_valid_trades, create_target_columns
        from backend.utils.smart_data_merger import smart_merge_datasets
        
        sample_status = dash.no_update
        trade_status = dash.no_update
        feature_status = dash.no_update
        trades_df = None
        features_df = None
        
        # Only load data when button is clicked
        if load_clicks:
            if trade_path and os.path.exists(trade_path):
                try:
                    # Load with timeout protection (60 seconds)
                    with timeout_context(TIMEOUT_CSV_LOAD):
                        trades_df = load_trade_csv(trade_path)
                except TimeoutError:
                    raise Exception(f"Timeout loading trade CSV (>{TIMEOUT_CSV_LOAD}s): {os.path.basename(trade_path)}")
                except Exception:
                    try:
                        with timeout_context(TIMEOUT_CSV_LOAD):
                            trades_df = pd.read_csv(trade_path, sep='\t')
                    except TimeoutError:
                        raise Exception(f"Timeout loading trade CSV (>{TIMEOUT_CSV_LOAD}s): {os.path.basename(trade_path)}")
                    except Exception:
                        with timeout_context(TIMEOUT_CSV_LOAD):
                            trades_df = pd.read_csv(trade_path, sep=';')
                if 'Timestamp' in trades_df.columns:
                    trades_df['Timestamp'] = pd.to_datetime(trades_df['Timestamp'], errors='coerce')
                if 'entry_time' in trades_df.columns:
                    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'], errors='coerce')
                if 'trade_success' not in trades_df.columns:
                    if 'R_multiple' in trades_df.columns:
                        trades_df['trade_success'] = (trades_df['R_multiple'] > 0).astype(int)
                    elif 'net_profit' in trades_df.columns:
                        trades_df['trade_success'] = (trades_df['net_profit'] > 0).astype(int)
                trade_status = dbc.Alert([
                    html.I(className="bi bi-check-circle me-2"),
                    f"Loaded trade CSV: {os.path.basename(trade_path)} ({len(trades_df)} rows)"
                ], color="success")
            if feature_path and os.path.exists(feature_path):
                try:
                    features_df = load_feature_csv(feature_path)
                except Exception:
                    try:
                        features_df = pd.read_csv(feature_path, sep=';')
                    except Exception:
                        features_df = pd.read_csv(feature_path, sep=',')
                ts_candidates = ['timestamp', 'Timestamp', 'time', 'datetime']
                for c in ts_candidates:
                    if c in features_df.columns:
                        features_df = features_df.rename(columns={c: 'timestamp'})
                        break
                if 'timestamp' in features_df.columns:
                    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'], errors='coerce')
                feature_status = dbc.Alert([
                    html.I(className="bi bi-check-circle me-2"),
                    f"Loaded feature CSV: {os.path.basename(feature_path)} ({len(features_df)} rows)"
                ], color="success")
        
        # Check if no data was loaded
        if trades_df is None and features_df is None:
            return sample_status, trade_status, feature_status, dash.no_update, dash.no_update, dash.no_update
        if trades_df is not None:
            store_trade_data(trades_df)
        merged_df = None
        if trades_df is not None and features_df is not None:
            # Use smart merge for flexible column handling with timeout
            try:
                with timeout_context(TIMEOUT_DATA_MERGE):
                    merged_df, merge_stats = smart_merge_datasets(
                        features_df, trades_df,
                        merge_tolerance='1min',
                        verbose=True
                    )
                    print(f"[INFO] Smart merge completed: {merge_stats['matched_trades']}/{merge_stats['total_trades']} trades matched")
            except TimeoutError:
                raise Exception(f"Timeout merging datasets (>{TIMEOUT_DATA_MERGE//60} minutes). Data terlalu besar.")
            except Exception as e:
                print(f"[WARNING] Smart merge failed: {e}. Falling back to standard merge.")
                try:
                    with timeout_context(TIMEOUT_DATA_MERGE):
                        merged_df = merge_datasets(features_df, trades_df)
                except TimeoutError:
                    raise Exception(f"Timeout merging datasets (>{TIMEOUT_DATA_MERGE//60} minutes). Data terlalu besar.")
        elif trades_df is not None and features_df is None:
            merged_df = trades_df
        else:
            merged_df = None
        if merged_df is not None:
            try:
                merged_df = filter_valid_trades(merged_df)
                merged_df = create_target_columns(merged_df)
            except Exception:
                pass
            store_merged_data(merged_df)
        gd = {'loaded': True, 'trade_path': trade_path, 'feature_path': feature_path}
        tr = ({'data': trades_df.to_json(date_format='iso', orient='split')} if trades_df is not None else dash.no_update)
        mr = ({'data': merged_df.to_json(date_format='iso', orient='split')} if merged_df is not None else dash.no_update)
        return sample_status, trade_status, feature_status, gd, tr, mr
    except Exception as e:
        from dash import html
        import dash_bootstrap_components as dbc
        err = dbc.Alert(str(e), color="danger")
        return err, err, err, dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    print("\n" + "="*60)
    print("STARTING TRADING PROBABILITY EXPLORER")
    print("="*60)
    print(f"Host: {DASH_HOST}")
    print(f"Port: {DASH_PORT}")
    print(f"Debug Mode: {DASH_DEBUG}")
    print(f"URL: http://{DASH_HOST}:{DASH_PORT}")
    print("="*60 + "\n")
    
    try:
        app.run(
            host=DASH_HOST,
            port=DASH_PORT,
            debug=DASH_DEBUG
        )
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR STARTING APPLICATION")
        print("="*60)
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        print("="*60 + "\n")
        raise
