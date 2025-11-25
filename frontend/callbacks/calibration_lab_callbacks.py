"""
Calibration Lab Callbacks

Callbacks for Calibration Lab tab including reliability diagram,
calibration metrics, and probability distribution analysis.

Enhanced with:
- Interactive controls for probability column, target variable, bins, strategy
- Comprehensive info panel with analysis parameters and data summary
- Auto-generated insights with calibration quality assessment
- Export functionality for calibration results
- Help modal with interpretation guide
"""

from dash import Input, Output, State, callback_context, html, dcc
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from backend.models.calibration import (
    compute_reliability_diagram,
    compute_brier_score,
    compute_ece,
    create_calibration_bins
)
from backend.utils.data_cache import get_merged_data, get_trade_data
from backend.utils.probability_generator import (
    auto_generate_all_probabilities,
    get_probability_info
)
from frontend.components.calibration_metrics_cards import create_calibration_metrics_cards
from frontend.components.reliability_diagram import create_reliability_diagram
from frontend.components.calibration_histogram import create_calibration_histogram
from frontend.components.calibration_table import create_calibration_table
import io
import base64


def register_calibration_lab_callbacks(app):
    """
    Register all Calibration Lab callbacks.
    
    Args:
        app: Dash application instance
    """
    
    print("\n" + "="*80)
    print("[CALIBRATION CALLBACKS] Registering callbacks...")
    print("="*80)
    
    # Callback to dynamically populate probability column dropdown
    @app.callback(
        Output('calib-prob-column-dropdown', 'options'),
        [
            Input('merged-data-store', 'data'),
            Input('trade-data-store', 'data'),
            Input('main-tabs', 'active_tab')
        ],
        prevent_initial_call=False
    )
    def update_prob_column_options(merged_data, trade_data, active_tab):
        """
        Dynamically update probability column dropdown based on available data.
        """
        # Only update when on calibration tab
        if active_tab != 'calibration-lab':
            raise PreventUpdate
        
        print(f"\n[CALIBRATION DROPDOWN] Updating probability column options...")
        
        # Get data
        data_src = merged_data if merged_data else trade_data
        if not data_src:
            cached = get_merged_data()
            if cached is None:
                cached = get_trade_data()
            if cached is None:
                print(f"[CALIBRATION DROPDOWN] No data available")
                return [
                    {'label': '⚠️ Muat data terlebih dahulu', 'value': None, 'disabled': True}
                ]
            data_src = cached
        
        try:
            # Load DataFrame
            if isinstance(data_src, dict) and 'data' in data_src:
                df = pd.read_json(io.StringIO(data_src['data']), orient='split')
            elif isinstance(data_src, pd.DataFrame):
                df = data_src.copy()
            else:
                df = pd.DataFrame(data_src)
            
            # Find probability columns
            prob_cols = [col for col in df.columns if 'prob' in col.lower()]
            
            print(f"[CALIBRATION DROPDOWN] Found {len(prob_cols)} probability columns: {prob_cols}")
            
            if not prob_cols:
                return [
                    {'label': '⚠️ Tidak ada kolom probabilitas - akan di-generate otomatis saat klik "Hitung Kalibrasi"', 'value': 'auto_generate', 'disabled': False}
                ]
            
            # Create options with nice labels
            options = []
            for col in prob_cols:
                # Create readable label
                if 'model_prob' in col.lower():
                    label = f"🎯 {col} (Model Prediction)"
                elif 'global' in col.lower():
                    label = f"🌍 {col} (Global)"
                elif 'session' in col.lower():
                    label = f"📅 {col} (Session)"
                elif 'trend' in col.lower():
                    label = f"📈 {col} (Trend)"
                elif 'vol' in col.lower():
                    label = f"📊 {col} (Volatility)"
                elif 'composite' in col.lower():
                    label = f"⭐ {col} (Composite)"
                else:
                    label = f"📌 {col}"
                
                options.append({'label': label, 'value': col})
            
            print(f"[CALIBRATION DROPDOWN] Created {len(options)} dropdown options")
            return options
            
        except Exception as e:
            print(f"[CALIBRATION DROPDOWN] Error: {e}")
            return [
                {'label': f'⚠️ Error: {str(e)}', 'value': None, 'disabled': True}
            ]
    
    @app.callback(
        [
            Output('reliability-diagram', 'figure'),
            Output('calibration-histogram', 'figure'),
            Output('calibration-metrics-cards', 'children'),
            Output('calibration-table', 'children'),
            Output('calib-insights-content', 'children'),
            Output('calib-info-panel', 'children'),
            Output('calibration-results-store', 'data'),
            Output('predicted-probs-store', 'data')
        ],
        [
            Input('calib-calculate-btn', 'n_clicks'),
            Input('main-tabs', 'active_tab')
        ],
        [
            State('merged-data-store', 'data'),
            State('trade-data-store', 'data'),
            State('calib-prob-column-dropdown', 'value'),
            State('calib-target-variable-dropdown', 'value'),
            State('calib-n-bins-slider', 'value'),
            State('calib-strategy-dropdown', 'value')
        ],
        prevent_initial_call=True  # IMPORTANT: Prevent initial call
    )
    def update_calibration_analysis(n_clicks, active_tab, merged_data, trade_data, 
                                   prob_column, target_variable, n_bins, strategy):
        """
        Update all calibration visualizations with comprehensive analysis.
        
        Enhanced features:
        - Interactive controls for probability column and target variable
        - Binning strategy selection (quantile vs uniform)
        - Comprehensive info panel with analysis parameters
        - Auto-generated insights with quality assessment
        - Export-ready data stores
        - Enhanced validation and error handling
        - Dynamic column detection
        """
        print(f"\n{'='*80}")
        print(f"[CALIBRATION] 🔥 Callback triggered!")
        print(f"[CALIBRATION] n_clicks: {n_clicks}")
        print(f"[CALIBRATION] active_tab: {active_tab}")
        
        # Get callback context
        ctx = callback_context
        
        # Log trigger info - ALWAYS log this first
        if ctx.triggered:
            trigger_info = ctx.triggered[0]
            trigger_id = trigger_info['prop_id'].split('.')[0]
            trigger_prop = trigger_info['prop_id'].split('.')[-1]
            trigger_value = trigger_info['value']
            print(f"[CALIBRATION] Trigger ID: {trigger_id}")
            print(f"[CALIBRATION] Trigger prop: {trigger_prop}")
            print(f"[CALIBRATION] Trigger value: {trigger_value}")
        else:
            trigger_id = None
            print(f"[CALIBRATION] ⚠️ No trigger (should not happen with prevent_initial_call=True)")
            raise PreventUpdate
        
        # CRITICAL: Only process if button was clicked
        if trigger_id != 'calib-calculate-btn':
            print(f"[CALIBRATION] ⏭️ Not a button click (trigger={trigger_id}), skipping")
            print(f"{'='*80}\n")
            raise PreventUpdate
        
        # If we reach here, button was definitely clicked
        print(f"[CALIBRATION] ✅ BUTTON CLICKED! Processing analysis...")
        print(f"[CALIBRATION] Click count: {n_clicks}")
        
        # Get data with enhanced validation
        print(f"[CALIBRATION] Checking data sources...")
        print(f"  - merged_data exists: {merged_data is not None}")
        print(f"  - trade_data exists: {trade_data is not None}")
        
        data_src = merged_data if merged_data else trade_data
        if not data_src:
            print(f"[CALIBRATION] No data in stores, checking cache...")
            cached = get_merged_data()
            if cached is None:
                cached = get_trade_data()
            if cached is None:
                print(f"[CALIBRATION] ERROR: No data available!")
                print(f"{'='*80}\n")
                return (
                    create_empty_figure("⚠️ Data belum dimuat! Klik 'Muat Data Terpilih' di panel atas."),
                    create_empty_figure("⚠️ Data belum dimuat!"),
                    create_calibration_metrics_cards(),
                    create_empty_calibration_table(),
                    dbc.Alert([
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        html.Strong("Data Belum Dimuat! "),
                        html.Br(),
                        "Silakan muat data trade atau merged data terlebih dahulu menggunakan panel kontrol di atas.",
                        html.Br(),
                        html.Br(),
                        html.Small("Langkah: Pilih file CSV → Klik 'Muat Data Terpilih' → Kembali ke Lab Kalibrasi → Klik 'Hitung Kalibrasi'")
                    ], color="warning"),
                    create_empty_info_panel(),
                    None,
                    None
                )
            data_src = cached
            print(f"[CALIBRATION] Data loaded from cache")
        
        try:
            # Load DataFrame with enhanced error handling
            print(f"[CALIBRATION] Loading DataFrame...")
            if isinstance(data_src, dict) and 'data' in data_src:
                df = pd.read_json(io.StringIO(data_src['data']), orient='split')
                print(f"[CALIBRATION] Loaded from JSON dict")
            elif isinstance(data_src, pd.DataFrame):
                df = data_src.copy()
                print(f"[CALIBRATION] Loaded from DataFrame")
            else:
                try:
                    df = pd.DataFrame(data_src)
                    print(f"[CALIBRATION] Converted to DataFrame")
                except Exception as e:
                    print(f"[CALIBRATION] ERROR: Cannot convert to DataFrame: {e}")
                    print(f"{'='*80}\n")
                    return (
                        create_empty_figure(f"❌ Error loading data: {str(e)}"),
                        create_empty_figure(f"❌ Error loading data"),
                        create_calibration_metrics_cards(),
                        create_empty_calibration_table(),
                        dbc.Alert([
                            html.I(className="bi bi-x-circle me-2"),
                            html.Strong("Error Loading Data! "),
                            f"Format data tidak valid: {str(e)}"
                        ], color="danger"),
                        create_empty_info_panel(),
                        None,
                        None
                    )
            
            print(f"[CALIBRATION] DataFrame shape: {df.shape}")
            print(f"[CALIBRATION] DataFrame columns: {df.columns.tolist()[:20]}...")
            
            # Check if probability columns exist, if not, generate them
            prob_info = get_probability_info(df)
            if not prob_info['has_probability_columns']:
                print(f"[CALIBRATION] ⚠️ No probability columns found!")
                print(f"[CALIBRATION] 🔧 Auto-generating probability columns...")
                
                # Determine target column
                if target_variable and target_variable in df.columns:
                    target_for_gen = target_variable
                elif 'trade_success' in df.columns:
                    target_for_gen = 'trade_success'
                elif 'R_multiple' in df.columns:
                    df['trade_success'] = (df['R_multiple'] > 0).astype(int)
                    target_for_gen = 'trade_success'
                else:
                    print(f"[CALIBRATION] ❌ Cannot generate probabilities: no valid target column")
                    return (
                        create_empty_figure("❌ Tidak ada kolom probabilitas dan tidak bisa generate otomatis"),
                        create_empty_figure("❌ Tidak ada kolom probabilitas"),
                        create_calibration_metrics_cards(),
                        create_empty_calibration_table(),
                        dbc.Alert([
                            html.I(className="bi bi-exclamation-triangle me-2"),
                            html.Strong("Tidak Ada Kolom Probabilitas! "),
                            html.Br(),
                            html.Br(),
                            "Data tidak memiliki kolom probabilitas dan tidak bisa di-generate otomatis.",
                            html.Br(),
                            "Data harus memiliki salah satu: trade_success, R_multiple, atau net_profit"
                        ], color="danger"),
                        create_empty_info_panel(),
                        None,
                        None
                    )
                
                # Generate probabilities
                try:
                    df = auto_generate_all_probabilities(df, target_col=target_for_gen, include_model=True)
                    print(f"[CALIBRATION] ✅ Probability columns generated successfully!")
                    
                    # Update prob_info
                    prob_info = get_probability_info(df)
                    print(f"[CALIBRATION] Available probability columns: {prob_info['probability_columns']}")
                    
                except Exception as e:
                    print(f"[CALIBRATION] ❌ Error generating probabilities: {e}")
                    import traceback
                    traceback.print_exc()
                    return (
                        create_empty_figure(f"❌ Error generating probabilities: {str(e)}"),
                        create_empty_figure(f"❌ Error"),
                        create_calibration_metrics_cards(),
                        create_empty_calibration_table(),
                        dbc.Alert([
                            html.I(className="bi bi-x-circle me-2"),
                            html.Strong("Error Generating Probabilities! "),
                            html.Br(),
                            f"Error: {str(e)}"
                        ], color="danger"),
                        create_empty_info_panel(),
                        None,
                        None
                    )
            else:
                print(f"[CALIBRATION] ✅ Found {prob_info['count']} probability columns: {prob_info['probability_columns']}")
            
            if df.empty:
                print(f"[CALIBRATION] ERROR: DataFrame is empty!")
                print(f"{'='*80}\n")
                return (
                    create_empty_figure("❌ Data kosong!"),
                    create_empty_figure("❌ Data kosong!"),
                    create_calibration_metrics_cards(),
                    create_empty_calibration_table(),
                    dbc.Alert([
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        html.Strong("Data Kosong! "),
                        "DataFrame tidak memiliki data."
                    ], color="warning"),
                    create_empty_info_panel(),
                    None,
                    None
                )
            
            # Set default values with enhanced detection
            print(f"[CALIBRATION] Setting default values...")
            print(f"[CALIBRATION] prob_column from dropdown: {prob_column}")
            
            # Handle 'auto_generate' value from dropdown
            if prob_column == 'auto_generate' or not prob_column:
                print(f"[CALIBRATION] Need to find or generate probability column...")
                # Try to find probability columns
                prob_cols = [col for col in df.columns if 'prob' in col.lower()]
                print(f"[CALIBRATION] Found probability columns: {prob_cols}")
                if prob_cols:
                    prob_column = prob_cols[0]
                    print(f"[CALIBRATION] Auto-selected prob column: {prob_column}")
                else:
                    prob_column = None
                    print(f"[CALIBRATION] No probability columns found, will need to generate")
            
            if not target_variable:
                target_variable = 'trade_success'
            
            if not n_bins:
                n_bins = 10
            
            if not strategy:
                strategy = 'quantile'
            
            print(f"[CALIBRATION] Parameters:")
            print(f"  - prob_column: {prob_column}")
            print(f"  - target_variable: {target_variable}")
            print(f"  - n_bins: {n_bins}")
            print(f"  - strategy: {strategy}")
            
            # Validate probability column exists with better error message
            if not prob_column or prob_column not in df.columns:
                available_prob_cols = [col for col in df.columns if 'prob' in col.lower()]
                available_cols_str = ', '.join(df.columns.tolist()[:15])
                if len(df.columns) > 15:
                    available_cols_str += f", ... ({len(df.columns)} total)"
                
                print(f"[CALIBRATION] ERROR: Probability column '{prob_column}' not found!")
                print(f"[CALIBRATION] Available probability columns: {available_prob_cols}")
                print(f"{'='*80}\n")
                
                return (
                    create_empty_figure("❌ Kolom probabilitas tidak ditemukan"),
                    create_empty_figure("❌ Kolom probabilitas tidak ditemukan"),
                    create_calibration_metrics_cards(),
                    create_empty_calibration_table(),
                    dbc.Alert([
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        html.Strong("Kolom Probabilitas Tidak Ditemukan! "),
                        html.Br(),
                        html.Br(),
                        f"Kolom yang dicari: '{prob_column}'",
                        html.Br(),
                        html.Br(),
                        html.Strong("Kolom probabilitas yang tersedia: "),
                        html.Br(),
                        ', '.join(available_prob_cols) if available_prob_cols else "Tidak ada kolom probabilitas!",
                        html.Br(),
                        html.Br(),
                        html.Small([
                            html.Strong("Semua kolom di data: "),
                            html.Br(),
                            available_cols_str
                        ])
                    ], color="warning"),
                    create_empty_info_panel(),
                    None,
                    None
                )
            
            # Prepare target variable with enhanced fallback
            print(f"[CALIBRATION] Validating target variable...")
            if target_variable not in df.columns:
                print(f"[CALIBRATION] Target '{target_variable}' not found, trying fallbacks...")
                if 'trade_success' in df.columns:
                    target_variable = 'trade_success'
                    print(f"[CALIBRATION] Using 'trade_success'")
                elif 'R_multiple' in df.columns:
                    df['trade_success'] = (df['R_multiple'] > 0).astype(int)
                    target_variable = 'trade_success'
                    print(f"[CALIBRATION] Created 'trade_success' from R_multiple")
                elif 'net_profit' in df.columns:
                    df['trade_success'] = (df['net_profit'] > 0).astype(int)
                    target_variable = 'trade_success'
                    print(f"[CALIBRATION] Created 'trade_success' from net_profit")
                elif 'y_hit_1R' in df.columns:
                    target_variable = 'y_hit_1R'
                    print(f"[CALIBRATION] Using 'y_hit_1R'")
                elif 'y_hit_2R' in df.columns:
                    target_variable = 'y_hit_2R'
                    print(f"[CALIBRATION] Using 'y_hit_2R'")
                else:
                    available_targets = [col for col in df.columns if any(x in col.lower() for x in ['success', 'target', 'y_', 'label'])]
                    print(f"[CALIBRATION] ERROR: No valid target variable found!")
                    print(f"[CALIBRATION] Available target-like columns: {available_targets}")
                    print(f"{'='*80}\n")
                    
                    return (
                        create_empty_figure("❌ Target variable tidak ditemukan"),
                        create_empty_figure("❌ Target variable tidak ditemukan"),
                        create_calibration_metrics_cards(),
                        create_empty_calibration_table(),
                        dbc.Alert([
                            html.I(className="bi bi-exclamation-triangle me-2"),
                            html.Strong("Target Variable Tidak Ditemukan! "),
                            html.Br(),
                            html.Br(),
                            f"Kolom yang dicari: '{target_variable}'",
                            html.Br(),
                            html.Br(),
                            html.Strong("Kolom yang tersedia: "),
                            ', '.join(available_targets) if available_targets else "Tidak ada kolom target yang cocok!",
                            html.Br(),
                            html.Br(),
                            html.Small("Data harus memiliki kolom seperti: trade_success, R_multiple, net_profit, y_hit_1R, atau y_hit_2R")
                        ], color="warning"),
                        create_empty_info_panel(),
                        None,
                        None
                    )
            
            print(f"[CALIBRATION] Target variable: {target_variable}")
            
            # Clean data with detailed logging
            print(f"[CALIBRATION] Cleaning data...")
            print(f"  - Before cleaning: {len(df)} rows")
            df_clean = df[[prob_column, target_variable]].copy()
            
            # Count NaN values
            nan_prob = df_clean[prob_column].isna().sum()
            nan_target = df_clean[target_variable].isna().sum()
            print(f"  - NaN in prob_column: {nan_prob}")
            print(f"  - NaN in target_variable: {nan_target}")
            
            df_clean = df_clean.dropna()
            print(f"  - After cleaning: {len(df_clean)} rows")
            
            if len(df_clean) < 10:
                print(f"[CALIBRATION] ERROR: Not enough data after cleaning!")
                print(f"{'='*80}\n")
                return (
                    create_empty_figure("❌ Data tidak cukup (minimum 10 samples)"),
                    create_empty_figure("❌ Data tidak cukup"),
                    create_calibration_metrics_cards(),
                    create_empty_calibration_table(),
                    dbc.Alert([
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        html.Strong("Data Tidak Cukup! "),
                        html.Br(),
                        html.Br(),
                        f"Data awal: {len(df)} rows",
                        html.Br(),
                        f"Setelah cleaning: {len(df_clean)} rows",
                        html.Br(),
                        f"NaN di {prob_column}: {nan_prob}",
                        html.Br(),
                        f"NaN di {target_variable}: {nan_target}",
                        html.Br(),
                        html.Br(),
                        html.Strong("Minimum 10 samples diperlukan untuk analisis kalibrasi.")
                    ], color="warning"),
                    create_empty_info_panel(),
                    None,
                    None
                )
            
            predicted_probs = df_clean[prob_column].values
            actual_outcomes = df_clean[target_variable].values
            
            print(f"[CALIBRATION] Data statistics:")
            print(f"  - Predicted probs range: [{predicted_probs.min():.4f}, {predicted_probs.max():.4f}]")
            print(f"  - Predicted probs mean: {predicted_probs.mean():.4f}")
            print(f"  - Actual outcomes unique: {np.unique(actual_outcomes)}")
            
            # Ensure binary outcomes
            if not np.all(np.isin(actual_outcomes, [0, 1])):
                print(f"[CALIBRATION] Converting outcomes to binary (0/1)")
                actual_outcomes = (actual_outcomes > 0).astype(int)
            
            # Ensure probabilities are in [0, 1]
            if predicted_probs.max() > 1:
                print(f"[CALIBRATION] Normalizing probabilities (dividing by 100)")
                predicted_probs = predicted_probs / 100.0
            predicted_probs = np.clip(predicted_probs, 0, 1)
            
            print(f"  - After normalization: [{predicted_probs.min():.4f}, {predicted_probs.max():.4f}]")
            
            # Compute calibration metrics
            print(f"[CALIBRATION] Computing calibration metrics...")
            reliability_result = compute_reliability_diagram(
                predicted_probs, 
                actual_outcomes, 
                n_bins=n_bins,
                strategy=strategy
            )
            print(f"[CALIBRATION] Reliability diagram computed")
            
            brier_score = compute_brier_score(predicted_probs, actual_outcomes)
            print(f"[CALIBRATION] Brier score: {brier_score:.6f}")
            
            ece = compute_ece(predicted_probs, actual_outcomes, n_bins=n_bins, strategy=strategy)
            print(f"[CALIBRATION] ECE: {ece:.6f}")
            
            # Create visualizations
            print(f"[CALIBRATION] Creating visualizations...")
            reliability_fig = create_reliability_diagram(reliability_result)
            histogram_fig = create_calibration_histogram(predicted_probs, n_bins=20)
            metrics_cards = create_calibration_metrics_cards(brier_score, ece, len(df_clean), n_bins)
            calibration_table = create_calibration_table(reliability_result)
            print(f"[CALIBRATION] Visualizations created")
            
            # Generate insights
            print(f"[CALIBRATION] Generating insights...")
            insights = generate_calibration_insights(
                brier_score, ece, reliability_result, 
                predicted_probs, actual_outcomes, n_bins, strategy
            )
            
            # Generate info panel
            info_panel = generate_info_panel(
                prob_column, target_variable, n_bins, strategy,
                len(df_clean), brier_score, ece
            )
            print(f"[CALIBRATION] Insights and info panel generated")
            
            # Prepare export data
            export_data = prepare_export_data(
                reliability_result, brier_score, ece, 
                prob_column, target_variable, n_bins, strategy
            )
            
            probs_data = {
                'predicted_probs': predicted_probs.tolist(),
                'actual_outcomes': actual_outcomes.tolist()
            }
            
            print(f"[CALIBRATION] ✅ SUCCESS! Analysis complete")
            print(f"{'='*80}\n")
            
            return (
                reliability_fig,
                histogram_fig,
                metrics_cards,
                calibration_table,
                insights,
                info_panel,
                export_data,
                probs_data
            )
            
        except Exception as e:
            print(f"Error in calibration analysis: {e}")
            import traceback
            traceback.print_exc()
            return (
                create_empty_figure(f"Error: {str(e)}"),
                create_empty_figure(f"Error: {str(e)}"),
                create_calibration_metrics_cards(),
                create_empty_calibration_table(),
                create_error_alert(f"Error dalam analisis: {str(e)}"),
                create_empty_info_panel(),
                None,
                None
            )


    # Export callback
    @app.callback(
        Output('calib-download-data', 'data'),
        Input('calib-export-btn', 'n_clicks'),
        State('calibration-results-store', 'data'),
        prevent_initial_call=True
    )
    def export_calibration_results(n_clicks, results_data):
        """Export calibration results to CSV."""
        print(f"\n{'='*60}")
        print(f"[CALIBRATION EXPORT] Callback triggered!")
        print(f"[CALIBRATION EXPORT] n_clicks: {n_clicks}")
        print(f"[CALIBRATION EXPORT] results_data exists: {results_data is not None}")
        print(f"{'='*60}\n")
        
        if not n_clicks:
            print("[CALIBRATION EXPORT] No clicks, preventing update")
            raise PreventUpdate
        
        if not results_data:
            print("[CALIBRATION EXPORT] ERROR: No results data!")
            raise PreventUpdate
        
        try:
            from datetime import datetime
            
            print(f"[CALIBRATION EXPORT] Starting export...")
            
            # Create comprehensive export
            export_lines = []
            
            # Header
            export_lines.append("Calibration Analysis Results")
            export_lines.append(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            export_lines.append("")
            
            # Analysis Parameters
            export_lines.append("=== ANALYSIS PARAMETERS ===")
            export_lines.append(f"Probability Column: {results_data.get('prob_column', 'N/A')}")
            export_lines.append(f"Target Variable: {results_data.get('target_variable', 'N/A')}")
            export_lines.append(f"Number of Bins: {results_data.get('n_bins', 'N/A')}")
            export_lines.append(f"Binning Strategy: {results_data.get('strategy', 'N/A')}")
            export_lines.append(f"Total Samples: {results_data.get('n_samples', 'N/A')}")
            export_lines.append("")
            
            # Overall Metrics
            export_lines.append("=== OVERALL CALIBRATION METRICS ===")
            export_lines.append(f"Brier Score: {results_data.get('brier_score', 0):.6f}")
            export_lines.append(f"Expected Calibration Error (ECE): {results_data.get('ece', 0):.6f}")
            
            # Quality assessment
            brier = results_data.get('brier_score', 0)
            if brier < 0.05:
                quality = "Excellent"
            elif brier < 0.10:
                quality = "Good"
            elif brier < 0.15:
                quality = "Fair"
            else:
                quality = "Poor"
            export_lines.append(f"Overall Quality: {quality}")
            export_lines.append("")
            
            # Per-Bin Details
            export_lines.append("=== PER-BIN CALIBRATION DETAILS ===")
            export_lines.append("Bin,Range,Center,Mean Predicted,Observed Frequency,Deviation,Sample Count,Quality")
            
            reliability = results_data.get('reliability_result', {})
            bin_edges = reliability.get('bin_edges', [])
            mean_predicted = reliability.get('mean_predicted', [])
            observed_frequency = reliability.get('observed_frequency', [])
            n_samples = reliability.get('n_samples', [])
            bin_centers = reliability.get('bin_centers', [])
            
            # Convert to numpy arrays for safe handling
            bin_edges = np.array(bin_edges)
            mean_predicted = np.array(mean_predicted)
            observed_frequency = np.array(observed_frequency)
            n_samples = np.array(n_samples)
            bin_centers = np.array(bin_centers)
            
            for i in range(len(mean_predicted)):
                # Check if value is valid (not NaN and not None)
                if mean_predicted[i] is not None and not np.isnan(mean_predicted[i]):
                    deviation = abs(mean_predicted[i] - observed_frequency[i])
                    if deviation < 0.05:
                        bin_quality = "Excellent"
                    elif deviation < 0.10:
                        bin_quality = "Good"
                    elif deviation < 0.15:
                        bin_quality = "Fair"
                    else:
                        bin_quality = "Poor"
                    
                    export_lines.append(
                        f"{i + 1},"
                        f"[{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}],"
                        f"{bin_centers[i]:.3f},"
                        f"{mean_predicted[i]:.6f},"
                        f"{observed_frequency[i]:.6f},"
                        f"{deviation:.6f},"
                        f"{int(n_samples[i])},"
                        f"{bin_quality}"
                    )
            
            # Create CSV content
            csv_content = "\n".join(export_lines)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prob_col = results_data.get('prob_column', 'calibration')
            filename = f"calibration_{prob_col}_{timestamp}.csv"
            
            print(f"\n[CALIBRATION EXPORT] SUCCESS!")
            print(f"  - Generated CSV with {len(export_lines)} lines")
            print(f"  - Filename: {filename}")
            print(f"  - CSV size: {len(csv_content)} characters")
            print(f"  - Returning download data...\n")
            
            return dict(content=csv_content, filename=filename)
            
        except Exception as e:
            print(f"[CALIBRATION EXPORT] ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise PreventUpdate
    
    # Help modal callback
    @app.callback(
        Output('calib-help-modal', 'is_open'),
        [Input('calib-help-btn', 'n_clicks'), Input('calib-help-close', 'n_clicks')],
        [State('calib-help-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_help_modal(n1, n2, is_open):
        """Toggle help modal visibility."""
        if n1 or n2:
            return not is_open
        return is_open


def generate_calibration_insights(brier_score, ece, reliability_result, 
                                  predicted_probs, actual_outcomes, n_bins, strategy):
    """
    Generate comprehensive calibration insights and recommendations.
    
    Returns:
        dash_bootstrap_components layout with insights
    """
    insights = []
    
    # Overall calibration quality
    if brier_score < 0.05 and ece < 0.05:
        quality = "EXCELLENT"
        quality_color = "success"
        quality_icon = "bi-check-circle-fill"
        quality_msg = "Model memiliki kalibrasi yang sangat baik! Prediksi probabilitas sangat akurat."
    elif brier_score < 0.10 and ece < 0.10:
        quality = "GOOD"
        quality_color = "info"
        quality_icon = "bi-check-circle"
        quality_msg = "Model memiliki kalibrasi yang baik. Prediksi probabilitas cukup dapat diandalkan."
    elif brier_score < 0.15 and ece < 0.15:
        quality = "FAIR"
        quality_color = "warning"
        quality_icon = "bi-exclamation-circle"
        quality_msg = "Model memiliki kalibrasi yang cukup. Ada ruang untuk perbaikan."
    else:
        quality = "POOR"
        quality_color = "danger"
        quality_icon = "bi-x-circle-fill"
        quality_msg = "Model memiliki kalibrasi yang buruk. Perlu perbaikan signifikan."
    
    insights.append(
        dbc.Alert([
            html.H5([
                html.I(className=f"{quality_icon} me-2"),
                f"Kualitas Kalibrasi: {quality}"
            ], className="alert-heading"),
            html.P(quality_msg, className="mb-0")
        ], color=quality_color, className="mb-3")
    )
    
    # Detailed analysis
    analysis_points = []
    
    # Brier Score analysis
    if brier_score < 0.05:
        analysis_points.append(f"✓ Brier Score sangat rendah ({brier_score:.4f}) - prediksi sangat akurat")
    elif brier_score < 0.10:
        analysis_points.append(f"✓ Brier Score rendah ({brier_score:.4f}) - prediksi akurat")
    elif brier_score < 0.15:
        analysis_points.append(f"⚠ Brier Score sedang ({brier_score:.4f}) - ada ruang perbaikan")
    else:
        analysis_points.append(f"✗ Brier Score tinggi ({brier_score:.4f}) - prediksi kurang akurat")
    
    # ECE analysis
    if ece < 0.05:
        analysis_points.append(f"✓ ECE sangat rendah ({ece:.4f}) - kalibrasi sangat baik")
    elif ece < 0.10:
        analysis_points.append(f"✓ ECE rendah ({ece:.4f}) - kalibrasi baik")
    elif ece < 0.15:
        analysis_points.append(f"⚠ ECE sedang ({ece:.4f}) - kalibrasi perlu ditingkatkan")
    else:
        analysis_points.append(f"✗ ECE tinggi ({ece:.4f}) - kalibrasi buruk")
    
    # Bin-level analysis
    mean_predicted = reliability_result.get('mean_predicted', [])
    observed_frequency = reliability_result.get('observed_frequency', [])
    n_samples = reliability_result.get('n_samples', [])
    
    valid_bins = sum(1 for mp in mean_predicted if not np.isnan(mp))
    analysis_points.append(f"📊 {valid_bins} dari {n_bins} bins memiliki data")
    
    # Count bins by quality
    excellent_bins = 0
    good_bins = 0
    fair_bins = 0
    poor_bins = 0
    
    for i in range(len(mean_predicted)):
        if not np.isnan(mean_predicted[i]):
            deviation = abs(mean_predicted[i] - observed_frequency[i])
            if deviation < 0.05:
                excellent_bins += 1
            elif deviation < 0.10:
                good_bins += 1
            elif deviation < 0.15:
                fair_bins += 1
            else:
                poor_bins += 1
    
    if excellent_bins > 0:
        analysis_points.append(f"✓ {excellent_bins} bins dengan kalibrasi excellent (deviasi < 5%)")
    if good_bins > 0:
        analysis_points.append(f"✓ {good_bins} bins dengan kalibrasi good (deviasi 5-10%)")
    if fair_bins > 0:
        analysis_points.append(f"⚠ {fair_bins} bins dengan kalibrasi fair (deviasi 10-15%)")
    if poor_bins > 0:
        analysis_points.append(f"✗ {poor_bins} bins dengan kalibrasi poor (deviasi > 15%)")
    
    # Probability distribution analysis
    mean_prob = np.mean(predicted_probs)
    std_prob = np.std(predicted_probs)
    
    if std_prob < 0.1:
        analysis_points.append(f"⚠ Prediksi terlalu konservatif (std={std_prob:.3f}) - model kurang confident")
    elif std_prob > 0.3:
        analysis_points.append(f"✓ Prediksi memiliki variasi baik (std={std_prob:.3f}) - model confident")
    else:
        analysis_points.append(f"✓ Prediksi memiliki variasi sedang (std={std_prob:.3f})")
    
    if mean_prob < 0.4 or mean_prob > 0.6:
        analysis_points.append(f"⚠ Mean probability {mean_prob:.1%} - mungkin ada bias dalam prediksi")
    
    insights.append(
        dbc.Card([
            dbc.CardHeader(html.H6("Analisis Detail", className="mb-0")),
            dbc.CardBody([
                html.Ul([html.Li(point) for point in analysis_points], className="mb-0")
            ])
        ], className="mb-3")
    )
    
    # Recommendations
    recommendations = []
    
    if brier_score > 0.10 or ece > 0.10:
        recommendations.append("Pertimbangkan untuk melakukan recalibration menggunakan Platt scaling atau isotonic regression")
    
    if poor_bins > 0:
        recommendations.append(f"Fokus perbaikan pada {poor_bins} bins dengan kalibrasi buruk")
    
    if std_prob < 0.1:
        recommendations.append("Model terlalu konservatif - review feature engineering dan model selection")
    
    if strategy == 'uniform' and valid_bins < n_bins * 0.7:
        recommendations.append("Coba gunakan 'quantile' binning strategy untuk distribusi yang lebih merata")
    
    if len(predicted_probs) < 100:
        recommendations.append("Kumpulkan lebih banyak data untuk analisis kalibrasi yang lebih robust")
    
    # Check for over/under confidence
    over_confident_bins = sum(1 for i in range(len(mean_predicted)) 
                             if not np.isnan(mean_predicted[i]) and 
                             mean_predicted[i] > observed_frequency[i] + 0.1)
    under_confident_bins = sum(1 for i in range(len(mean_predicted)) 
                              if not np.isnan(mean_predicted[i]) and 
                              mean_predicted[i] < observed_frequency[i] - 0.1)
    
    if over_confident_bins > valid_bins * 0.3:
        recommendations.append("Model cenderung over-confident - prediksi lebih tinggi dari aktual")
    if under_confident_bins > valid_bins * 0.3:
        recommendations.append("Model cenderung under-confident - prediksi lebih rendah dari aktual")
    
    if not recommendations:
        recommendations.append("Kalibrasi sudah baik - pertahankan kualitas dengan monitoring berkala")
    
    insights.append(
        dbc.Card([
            dbc.CardHeader(html.H6("Rekomendasi", className="mb-0")),
            dbc.CardBody([
                html.Ol([html.Li(rec) for rec in recommendations], className="mb-0")
            ])
        ], className="mb-3")
    )
    
    # Usage tips
    tips = [
        "Gunakan reliability diagram untuk identifikasi visual bins yang bermasalah",
        "Perhatikan ukuran marker di reliability diagram - bins dengan sample kecil kurang reliable",
        "Histogram menunjukkan distribusi prediksi - idealnya tersebar di range [0, 1]",
        "Gunakan quantile binning untuk data dengan distribusi tidak merata",
        "Gunakan uniform binning untuk menilai kalibrasi di seluruh range probabilitas",
        "Export hasil untuk dokumentasi dan tracking improvement over time"
    ]
    
    insights.append(
        dbc.Card([
            dbc.CardHeader(html.H6("Tips Penggunaan", className="mb-0")),
            dbc.CardBody([
                html.Ul([html.Li(tip) for tip in tips], className="mb-0 small")
            ])
        ])
    )
    
    return html.Div(insights)


def generate_info_panel(prob_column, target_variable, n_bins, strategy, 
                       n_samples, brier_score, ece):
    """Generate comprehensive info panel with analysis parameters."""
    
    # Determine quality badges
    if brier_score < 0.05:
        brier_badge = dbc.Badge("Excellent", color="success", className="ms-2")
    elif brier_score < 0.10:
        brier_badge = dbc.Badge("Good", color="info", className="ms-2")
    elif brier_score < 0.15:
        brier_badge = dbc.Badge("Fair", color="warning", className="ms-2")
    else:
        brier_badge = dbc.Badge("Poor", color="danger", className="ms-2")
    
    if ece < 0.05:
        ece_badge = dbc.Badge("Excellent", color="success", className="ms-2")
    elif ece < 0.10:
        ece_badge = dbc.Badge("Good", color="info", className="ms-2")
    elif ece < 0.15:
        ece_badge = dbc.Badge("Fair", color="warning", className="ms-2")
    else:
        ece_badge = dbc.Badge("Poor", color="danger", className="ms-2")
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-info-circle me-2"),
                "Informasi Analisis Kalibrasi"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Parameter Analisis", className="text-primary mb-3"),
                    html.Dl([
                        html.Dt("Kolom Probabilitas:"),
                        html.Dd(prob_column, className="mb-2"),
                        html.Dt("Target Variable:"),
                        html.Dd(target_variable, className="mb-2"),
                        html.Dt("Jumlah Bins:"),
                        html.Dd(f"{n_bins} bins", className="mb-2"),
                        html.Dt("Strategi Binning:"),
                        html.Dd(f"{strategy.capitalize()}", className="mb-2"),
                    ], className="mb-0")
                ], md=6),
                dbc.Col([
                    html.H6("Ringkasan Data", className="text-primary mb-3"),
                    html.Dl([
                        html.Dt("Total Samples:"),
                        html.Dd(f"{n_samples:,} prediksi", className="mb-2"),
                        html.Dt("Brier Score:"),
                        html.Dd([f"{brier_score:.6f}", brier_badge], className="mb-2"),
                        html.Dt("Expected Calibration Error:"),
                        html.Dd([f"{ece:.6f}", ece_badge], className="mb-2"),
                    ], className="mb-0")
                ], md=6)
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H6("Interpretasi Metrik", className="text-primary mb-2"),
                    html.Ul([
                        html.Li([
                            html.Strong("Brier Score: "),
                            "Mengukur akurasi prediksi probabilitas. Semakin rendah semakin baik. ",
                            "< 0.05 = excellent, 0.05-0.10 = good, 0.10-0.15 = fair, > 0.15 = poor"
                        ], className="small mb-2"),
                        html.Li([
                            html.Strong("ECE: "),
                            "Mengukur rata-rata deviasi dari kalibrasi sempurna. Semakin rendah semakin baik. ",
                            "Threshold sama dengan Brier Score."
                        ], className="small mb-2"),
                        html.Li([
                            html.Strong("Reliability Diagram: "),
                            "Titik pada garis diagonal = kalibrasi sempurna. ",
                            "Di atas diagonal = under-confident, di bawah diagonal = over-confident."
                        ], className="small mb-0")
                    ], className="mb-0")
                ])
            ])
        ])
    ], className="mb-3")


def prepare_export_data(reliability_result, brier_score, ece, 
                       prob_column, target_variable, n_bins, strategy):
    """Prepare data for export."""
    return {
        'reliability_result': {
            'bin_edges': reliability_result['bin_edges'].tolist(),
            'mean_predicted': reliability_result['mean_predicted'].tolist(),
            'observed_frequency': reliability_result['observed_frequency'].tolist(),
            'n_samples': reliability_result['n_samples'].tolist(),
            'bin_centers': reliability_result['bin_centers'].tolist()
        },
        'brier_score': float(brier_score),
        'ece': float(ece),
        'prob_column': prob_column,
        'target_variable': target_variable,
        'n_bins': int(n_bins),
        'strategy': strategy,
        'n_samples': int(np.sum(reliability_result['n_samples']))
    }


def create_empty_figure(message="Tidak ada data"):
    """Create empty figure with message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=400
    )
    return fig


def create_error_alert(message):
    """Create error alert."""
    return dbc.Alert([
        html.I(className="bi bi-exclamation-triangle me-2"),
        message
    ], color="warning", className="mb-0")


def create_empty_info_panel():
    """Create empty info panel."""
    return dbc.Card([
        dbc.CardBody([
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                "Klik tombol 'Calculate Calibration' untuk memulai analisis"
            ], color="info", className="mb-0")
        ])
    ], className="mb-3")


print("\n" + "="*80)
print("[CALIBRATION CALLBACKS] ✅ All callbacks registered successfully!")
print("  - Dropdown callback: calib-prob-column-dropdown")
print("  - Main analysis callback: update_calibration_analysis")
print("  - Export callback: export_calibration_results")
print("  - Help modal callback: toggle_help_modal")
print("="*80 + "\n")
