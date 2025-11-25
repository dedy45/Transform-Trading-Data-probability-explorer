"""
Regime Explorer Callbacks

Callbacks for Regime Explorer tab including regime-based win rate analysis,
R-multiple analysis, and regime comparison.

Enhanced with:
- Interactive controls for regime selection and filtering
- Comprehensive info panel with analysis parameters
- Auto-generated insights with regime recommendations
- Export functionality for regime analysis results
- Help modal with interpretation guide
"""

from dash import Input, Output, State, callback_context, html, dcc
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from backend.calculators.regime_analysis import (
    compute_regime_probabilities,
    compute_regime_threshold_probs,
    create_regime_comparison_table,
    filter_by_regime
)
from backend.utils.data_cache import get_merged_data, get_trade_data
import io



# ============================================
# HELPER FUNCTIONS
# ============================================

def create_regime_comparison_table_component(comparison_df):
    """Create regime comparison table component."""
    import dash_bootstrap_components as dbc
    from dash import html, dash_table
    
    if comparison_df is None or comparison_df.empty:
        return html.Div("No regime data available")
    
    df = comparison_df.copy()
    
    # Prepare data for table
    table_data = []
    for _, row in df.iterrows():
        table_data.append({
            'Regime': str(row['regime']),
            'Win Rate': f"{row['win_rate']:.1%}",
            'CI': f"[{row['win_rate_ci_lower']:.1%}, {row['win_rate_ci_upper']:.1%}]",
            'Mean R': f"{row['mean_r']:.2f}",
            'Median R': f"{row['median_r']:.2f}",
            'Std R': f"{row['std_r']:.2f}",
            'Trades': int(row['n_trades']),
            'Reliable': '✓' if row['reliable'] else '✗'
        })
    
    # Create DataTable
    table = dash_table.DataTable(
        data=table_data,
        columns=[
            {'name': 'Regime', 'id': 'Regime'},
            {'name': 'Win Rate', 'id': 'Win Rate'},
            {'name': 'CI (95%)', 'id': 'CI'},
            {'name': 'Mean R', 'id': 'Mean R'},
            {'name': 'Median R', 'id': 'Median R'},
            {'name': 'Std R', 'id': 'Std R'},
            {'name': 'Trades', 'id': 'Trades'},
            {'name': 'Reliable', 'id': 'Reliable'}
        ],
        sort_action='native',
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'fontFamily': 'Arial, sans-serif'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'Reliable', 'filter_query': '{Reliable} = "✓"'},
                'backgroundColor': 'rgba(50, 180, 50, 0.2)',
                'color': 'green',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'Reliable', 'filter_query': '{Reliable} = "✗"'},
                'backgroundColor': 'rgba(220, 50, 50, 0.2)',
                'color': 'red'
            }
        ]
    )
    
    return table
    
    table_body = [html.Tbody(rows)]
    
    table = dbc.Table(
        table_header + table_body,
        bordered=True,
        hover=True,
        responsive=True,
        striped=True
    )
    
    return table



def create_regime_winrate_chart_enhanced(regime_probs_df):
    """Create enhanced regime win rate bar chart."""
    if regime_probs_df is None or regime_probs_df.empty:
        return go.Figure()
    
    df = regime_probs_df.copy()
    
    # Color code by win rate
    colors = ['green' if wr > 0.5 else 'red' for wr in df['win_rate']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['regime'],
        y=df['win_rate'],
        name='Win Rate',
        marker_color=colors,
        error_y=dict(
            type='data',
            symmetric=False,
            array=df['ci_upper'] - df['win_rate'],
            arrayminus=df['win_rate'] - df['ci_lower']
        ),
        text=[f"{wr:.1%}<br>n={n}" for wr, n in zip(df['win_rate'], df['n_trades'])],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Win Rate: %{y:.1%}<br>Trades: %{customdata}<extra></extra>',
        customdata=df['n_trades']
    ))
    
    # Add 50% reference line
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50% (Break-even)")
    
    fig.update_layout(
        title='Win Rate by Regime',
        xaxis_title='Regime',
        yaxis_title='Win Rate',
        yaxis_tickformat='.0%',
        height=450,
        showlegend=False
    )
    
    return fig


def create_regime_r_multiple_chart_enhanced(regime_thresholds_df):
    """Create enhanced regime R-multiple threshold chart."""
    if regime_thresholds_df is None or regime_thresholds_df.empty:
        return go.Figure()
    
    df = regime_thresholds_df.copy()
    
    fig = go.Figure()
    
    # P(R >= 1)
    if 'p_r_gte_1_0' in df.columns:
        fig.add_trace(go.Bar(
            x=df['regime'],
            y=df['p_r_gte_1_0'],
            name='P(R ≥ 1)',
            marker_color='lightblue',
            text=[f"{p:.1%}" for p in df['p_r_gte_1_0']],
            textposition='outside'
        ))
    
    # P(R >= 2)
    if 'p_r_gte_2_0' in df.columns:
        fig.add_trace(go.Bar(
            x=df['regime'],
            y=df['p_r_gte_2_0'],
            name='P(R ≥ 2)',
            marker_color='darkblue',
            text=[f"{p:.1%}" for p in df['p_r_gte_2_0']],
            textposition='outside'
        ))
    
    fig.update_layout(
        title='R-Multiple Thresholds by Regime',
        xaxis_title='Regime',
        yaxis_title='Probability',
        yaxis_tickformat='.0%',
        barmode='group',
        height=450
    )
    
    return fig


def generate_regime_insights(comparison_df, regime_column, target_var, conf_level, min_samples):
    """Generate comprehensive regime insights."""
    if comparison_df is None or comparison_df.empty:
        return create_empty_insights()
    
    df = comparison_df.copy()
    
    insights = []
    
    # Find best and worst regimes
    best_idx = df['win_rate'].idxmax()
    worst_idx = df['win_rate'].idxmin()
    
    best_regime = df.loc[best_idx]
    worst_regime = df.loc[worst_idx]
    
    # Overall assessment
    spread = best_regime['win_rate'] - worst_regime['win_rate']
    
    if spread > 0.2:
        quality = "SIGNIFIKAN"
        quality_color = "success"
        quality_msg = f"Ada perbedaan signifikan ({spread:.1%}) antara regime terbaik dan terburuk!"
    elif spread > 0.1:
        quality = "MODERATE"
        quality_color = "info"
        quality_msg = f"Ada perbedaan moderate ({spread:.1%}) antara regimes."
    else:
        quality = "MINIMAL"
        quality_color = "warning"
        quality_msg = f"Perbedaan antar regimes minimal ({spread:.1%})."
    
    insights.append(
        dbc.Alert([
            html.H5([
                html.I(className="bi-bar-chart-fill me-2"),
                f"Regime Spread: {quality}"
            ], className="alert-heading"),
            html.P(quality_msg, className="mb-0")
        ], color=quality_color, className="mb-3")
    )
    
    # Detailed analysis
    analysis_points = []
    
    analysis_points.append(f"🏆 Best Regime: {best_regime['regime']} (Win Rate: {best_regime['win_rate']:.1%}, Mean R: {best_regime['mean_r']:.2f})")
    analysis_points.append(f"⚠️ Worst Regime: {worst_regime['regime']} (Win Rate: {worst_regime['win_rate']:.1%}, Mean R: {worst_regime['mean_r']:.2f})")
    analysis_points.append(f"📊 Total Regimes: {len(df)}")
    analysis_points.append(f"✓ Reliable Regimes: {df['reliable'].sum()} (sample size >= {min_samples})")
    
    # Profitable regimes
    profitable = df[df['win_rate'] > 0.5]
    if len(profitable) > 0:
        analysis_points.append(f"✓ {len(profitable)} regimes profitable (win rate > 50%)")
    
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
    
    if best_regime['win_rate'] > 0.55 and best_regime['mean_r'] > 0:
        recommendations.append(f"Focus trading pada regime '{best_regime['regime']}' - edge terbukti kuat")
    
    if worst_regime['win_rate'] < 0.45 or worst_regime['mean_r'] < 0:
        recommendations.append(f"Avoid atau reduce exposure pada regime '{worst_regime['regime']}'")
    
    if spread > 0.15:
        recommendations.append("Gunakan regime filter sebagai entry criteria dalam trading system")
    
    if df['reliable'].sum() < len(df) * 0.5:
        recommendations.append("Kumpulkan lebih banyak data untuk regimes dengan sample size kecil")
    
    if not recommendations:
        recommendations.append("Monitor regime performance secara berkala")
    
    insights.append(
        dbc.Card([
            dbc.CardHeader(html.H6("Rekomendasi", className="mb-0")),
            dbc.CardBody([
                html.Ol([html.Li(rec) for rec in recommendations], className="mb-0")
            ])
        ], className="mb-3")
    )
    
    return html.Div(insights)


def generate_info_panel(regime_column, target_var, conf_level, min_samples, total_trades, total_regimes):
    """Generate info panel with analysis parameters."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-info-circle me-2"),
                "Informasi Analisis Regime"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Parameter Analisis", className="text-primary mb-3"),
                    html.Dl([
                        html.Dt("Regime Column:"),
                        html.Dd(regime_column, className="mb-2"),
                        html.Dt("Target Variable:"),
                        html.Dd(target_var, className="mb-2"),
                        html.Dt("Confidence Level:"),
                        html.Dd(f"{conf_level}%", className="mb-2"),
                        html.Dt("Min Samples:"),
                        html.Dd(f"{min_samples} trades", className="mb-2"),
                    ], className="mb-0")
                ], md=6),
                dbc.Col([
                    html.H6("Ringkasan Data", className="text-primary mb-3"),
                    html.Dl([
                        html.Dt("Total Trades:"),
                        html.Dd(f"{total_trades:,} trades", className="mb-2"),
                        html.Dt("Total Regimes:"),
                        html.Dd(f"{total_regimes} regimes", className="mb-2"),
                    ], className="mb-0")
                ], md=6)
            ])
        ])
    ], className="mb-3")


def prepare_export_data(comparison_df, probs_df, thresholds_df, regime_column, target_var, conf_level, min_samples):
    """Prepare data for export."""
    if comparison_df is None or comparison_df.empty:
        return None
    
    df = comparison_df.copy()
    
    # Find best and worst
    best_idx = df['win_rate'].idxmax()
    worst_idx = df['win_rate'].idxmin()
    
    best_regime = df.loc[best_idx]
    worst_regime = df.loc[worst_idx]
    
    # Prepare regime details
    regime_details = []
    for _, row in df.iterrows():
        # Find corresponding threshold data
        threshold_row = None
        if not thresholds_df.empty:
            matching = thresholds_df[thresholds_df['regime'] == row['regime']]
            if not matching.empty:
                threshold_row = matching.iloc[0]
        
        detail = {
            'regime': row['regime'],
            'win_rate': float(row['win_rate']),
            'ci_lower': float(row['win_rate_ci_lower']),
            'ci_upper': float(row['win_rate_ci_upper']),
            'mean_r': float(row['mean_r']),
            'median_r': float(row['median_r']),
            'std_r': float(row['std_r']),
            'p_r_gte_1': float(threshold_row.get('p_r_gte_1_0', 0)) if threshold_row is not None else 0,
            'p_r_gte_2': float(threshold_row.get('p_r_gte_2_0', 0)) if threshold_row is not None else 0,
            'n_trades': int(row['n_trades']),
            'reliable': bool(row['reliable']),
            'quality': 'Excellent' if row['win_rate'] > 0.6 else 'Good' if row['win_rate'] > 0.5 else 'Fair' if row['win_rate'] > 0.4 else 'Poor'
        }
        regime_details.append(detail)
    
    # Recommendations
    recommendations = []
    if best_regime['win_rate'] > 0.55:
        recommendations.append(f"Focus trading pada regime '{best_regime['regime']}'")
    if worst_regime['win_rate'] < 0.45:
        recommendations.append(f"Avoid regime '{worst_regime['regime']}'")
    
    return {
        'regime_column': regime_column,
        'target_variable': target_var,
        'confidence_level': conf_level,
        'min_samples': min_samples,
        'total_regimes': len(df),
        'best_regime': {
            'regime': best_regime['regime'],
            'win_rate': float(best_regime['win_rate']),
            'mean_r': float(best_regime['mean_r']),
            'n_trades': int(best_regime['n_trades'])
        },
        'worst_regime': {
            'regime': worst_regime['regime'],
            'win_rate': float(worst_regime['win_rate']),
            'mean_r': float(worst_regime['mean_r']),
            'n_trades': int(worst_regime['n_trades'])
        },
        'regime_details': regime_details,
        'recommendations': recommendations
    }


def create_empty_info_panel():
    """Create empty info panel."""
    return dbc.Card([
        dbc.CardBody([
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                "Klik tombol 'Calculate Regime Analysis' untuk memulai analisis"
            ], color="info", className="mb-0")
        ])
    ], className="mb-3")


def create_empty_insights():
    """Create empty insights."""
    return dbc.Alert([
        html.I(className="bi bi-info-circle me-2"),
        "Calculate regime analysis to see insights and recommendations"
    ], color="info", className="mb-0")


def create_error_info_panel(error_msg):
    """Create error info panel."""
    return dbc.Card([
        dbc.CardBody([
            dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                html.Strong("Error: "),
                str(error_msg)
            ], color="danger", className="mb-0")
        ])
    ], className="mb-3")


# ============================================
# REGISTER CALLBACKS
# ============================================

def register_regime_explorer_callbacks(app):
    """
    Register all Regime Explorer callbacks.
    
    Args:
        app: Dash application instance
    """
    
    print(f"[REGIME EXPLORER] Registering main callback to app instance: {id(app)}")
    print(f"[REGIME EXPLORER] App has {len(app.callback_map)} callbacks before registration")
    
    @app.callback(
        [
            Output('regime-winrate-chart', 'figure'),
            Output('regime-r-multiple-chart', 'figure'),
            Output('regime-comparison-table', 'children'),
            Output('regime-summary-cards', 'children'),
            Output('regime-insights-content', 'children'),
            Output('regime-info-panel', 'children'),
            Output('regime-results-store', 'data'),
            Output('regime-probs-store', 'data')
        ],
        [
            Input('regime-calculate-btn', 'n_clicks'),
            Input('main-tabs', 'active_tab')
        ],
        [
            State('merged-data-store', 'data'),
            State('trade-data-store', 'data'),
            State('regime-column-dropdown', 'value'),
            State('regime-target-variable-dropdown', 'value'),
            State('regime-confidence-level-slider', 'value'),
            State('regime-min-samples-slider', 'value'),
            State('regime-selection-checklist', 'value')
        ],
        prevent_initial_call=True
    )
    def update_regime_analysis(n_clicks, active_tab, merged_data, trade_data,
                              regime_column, target_var, conf_level, min_samples, selected_regimes):
        """Update all regime analysis visualizations."""
        # PRINT FIRST - NO TRY-EXCEPT HERE! FORCE FLUSH!
        import sys
        print(f"\n{'='*60}", flush=True)
        print(f"[REGIME EXPLORER] ⚡ CALLBACK TRIGGERED!", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  - n_clicks: {n_clicks}", flush=True)
        print(f"  - active_tab: {active_tab}", flush=True)
        print(f"  - merged_data exists: {merged_data is not None}", flush=True)
        print(f"  - trade_data exists: {trade_data is not None}", flush=True)
        print(f"  - regime_column: {regime_column}", flush=True)
        print(f"  - target_var: {target_var}", flush=True)
        print(f"{'='*60}\n", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Check what triggered the callback
        ctx = callback_context
        if not ctx.triggered:
            print("[REGIME EXPLORER] No trigger, preventing update")
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        print(f"[REGIME EXPLORER] Triggered by: {trigger_id}")
        
        # Skip if just switching tabs to different tab
        if trigger_id == 'main-tabs' and active_tab != 'regime-explorer':
            print(f"[REGIME EXPLORER] Wrong tab ({active_tab}), preventing update")
            raise PreventUpdate
        
        # Wrap everything in try-except to catch any silent failures
        try:
            # Try to get data from stores first
            data_src = merged_data if merged_data else trade_data
            
            # If no data in stores, try cache
            if not data_src:
                print(f"  - No data in stores, checking cache...")
                cached = get_merged_data()
                if cached is None:
                    cached = get_trade_data()
                if cached is None:
                    print(f"  - No data in cache either, returning empty charts")
                    from frontend.components.regime_winrate_chart import create_empty_regime_winrate_chart
                    from frontend.components.regime_r_multiple_chart import create_empty_regime_r_multiple_chart
                    from frontend.components.regime_comparison_table import create_empty_regime_comparison_table
                    from frontend.components.regime_filter_controls import create_empty_regime_summary_cards
                    
                    empty_info = create_empty_info_panel()
                    empty_insights = create_empty_insights()
                    
                    return (
                        create_empty_regime_winrate_chart(),
                        create_empty_regime_r_multiple_chart(),
                        create_empty_regime_comparison_table(),
                        create_empty_regime_summary_cards(),
                        empty_insights,
                        empty_info,
                        None,
                        None
                    )
                data_src = cached
            import io
            if isinstance(data_src, dict) and 'data' in data_src:
                df = pd.read_json(io.StringIO(data_src['data']), orient='split')
            elif isinstance(data_src, pd.DataFrame):
                df = data_src.copy()
            else:
                df = pd.DataFrame(data_src)
            print(f"  - Parsed DataFrame: {len(df)} rows, {len(df.columns)} columns")
            
            # Ensure target column exists
            if target_var == 'trade_success':
                if 'trade_success' in df.columns:
                    df['trade_success'] = pd.to_numeric(df['trade_success'], errors='coerce').fillna(0).astype(int)
                else:
                    if 'R_multiple' in df.columns:
                        df['trade_success'] = (df['R_multiple'] > 0).astype(int)
                        print(f"  - Created trade_success from R_multiple")
                    elif 'net_profit' in df.columns:
                        df['trade_success'] = (df['net_profit'] > 0).astype(int)
                        print(f"  - Created trade_success from net_profit")
                    else:
                        print(f"  - ERROR: No suitable column for trade_success")
                        raise ValueError("Cannot create trade_success column")
            elif target_var == 'y_hit_1R':
                if 'y_hit_1R' not in df.columns and 'R_multiple' in df.columns:
                    df['y_hit_1R'] = (df['R_multiple'] >= 1).astype(int)
                    print(f"  - Created y_hit_1R from R_multiple")
            elif target_var == 'y_hit_2R':
                if 'y_hit_2R' not in df.columns and 'R_multiple' in df.columns:
                    df['y_hit_2R'] = (df['R_multiple'] >= 2).astype(int)
                    print(f"  - Created y_hit_2R from R_multiple")
            
            # Validate target column exists
            if target_var not in df.columns:
                print(f"  - ERROR: Target column {target_var} not found")
                raise ValueError(f"Target column {target_var} not available")
            
            if df.empty:
                print(f"  - DataFrame empty")
                raise ValueError("Invalid data")
            
            # Validate and set regime column
            # Check if regime_column exists in data, if not, find or create one
            if not regime_column or regime_column not in df.columns:
                print(f"  - Regime column '{regime_column}' not found, searching for alternatives...")
                
                # Try to find existing regime columns
                regime_cols = [col for col in df.columns if 'regime' in col.lower()]
                if regime_cols:
                    regime_column = regime_cols[0]
                    print(f"  - Found regime column: {regime_column}")
                else:
                    # Create regime column from available data
                    print(f"  - No regime columns found, creating from available data...")
                    try:
                        if 'entry_time' in df.columns:
                            df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
                            hours = df['entry_time'].dt.hour.fillna(0).astype(int)
                            df['time_regime'] = pd.cut(
                                hours,
                                bins=[-1, 6, 12, 18, 24],
                                labels=['Night', 'Morning', 'Afternoon', 'Evening']
                            )
                            regime_column = 'time_regime'
                            print(f"  - Created time_regime from entry_time")
                        elif 'Timestamp' in df.columns:
                            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                            hours = df['Timestamp'].dt.hour.fillna(0).astype(int)
                            df['time_regime'] = pd.cut(
                                hours,
                                bins=[-1, 6, 12, 18, 24],
                                labels=['Night', 'Morning', 'Afternoon', 'Evening']
                            )
                            regime_column = 'time_regime'
                            print(f"  - Created time_regime from Timestamp")
                        elif 'R_multiple' in df.columns:
                            df['r_regime'] = pd.qcut(
                                df['R_multiple'].fillna(0),
                                q=3,
                                labels=['Low', 'Medium', 'High'],
                                duplicates='drop'
                            )
                            regime_column = 'r_regime'
                            print(f"  - Created r_regime from R_multiple")
                        else:
                            print(f"  - ERROR: Cannot create regime column, no suitable data found")
                            # Create a simple regime based on index
                            df['index_regime'] = pd.qcut(df.index, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                            regime_column = 'index_regime'
                            print(f"  - Created index_regime as fallback")
                    except Exception as e:
                        print(f"  - ERROR creating regime column: {e}")
                        # Create a simple regime based on index
                        df['index_regime'] = pd.qcut(df.index, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                        regime_column = 'index_regime'
                        print(f"  - Created index_regime as fallback after error")
            
            # Filter by selected regimes if specified
            if selected_regimes:
                df = filter_by_regime(df, regime_column, selected_regimes)
            
            # Use target_var if provided
            if not target_var:
                target_var = 'trade_success'
            
            # Use conf_level if provided
            if not conf_level:
                conf_level = 95
            
            # Use min_samples if provided
            if not min_samples:
                min_samples = 5
            
            # Compute regime probabilities
            regime_probs_df = compute_regime_probabilities(
                df, regime_column, target_var, 
                conf_level=conf_level/100, min_samples=min_samples
            )
            
            # Compute regime threshold probabilities
            regime_thresholds_df = compute_regime_threshold_probs(
                df, regime_column, conf_level=conf_level/100, min_samples=min_samples
            )
            
            # Create comparison table
            comparison_table_df = create_regime_comparison_table(
                df, regime_column, target_var, conf_level=conf_level/100, min_samples=min_samples
            )
            
            # Create visualizations
            winrate_fig = create_regime_winrate_chart_enhanced(regime_probs_df)
            r_multiple_fig = create_regime_r_multiple_chart_enhanced(regime_thresholds_df)
            comparison_table = create_regime_comparison_table_component(comparison_table_df)
            
            # Create summary cards
            from frontend.components.regime_filter_controls import create_regime_summary_cards
            summary_cards = create_regime_summary_cards(comparison_table_df)
            
            # Generate insights
            insights = generate_regime_insights(
                comparison_table_df, regime_column, target_var, conf_level, min_samples
            )
            
            # Generate info panel
            info_panel = generate_info_panel(
                regime_column, target_var, conf_level, min_samples,
                len(df), len(comparison_table_df)
            )
            
            # Prepare export data
            export_data = prepare_export_data(
                comparison_table_df, regime_probs_df, regime_thresholds_df,
                regime_column, target_var, conf_level, min_samples
            )
            
            # Prepare probs data
            probs_data = {
                'regime_probs': regime_probs_df.to_dict('records') if not regime_probs_df.empty else [],
                'regime_thresholds': regime_thresholds_df.to_dict('records') if not regime_thresholds_df.empty else []
            }
            
            print(f"[Regime Explorer] Analysis complete!")
            print(f"  - Regimes analyzed: {len(comparison_table_df)}")
            print(f"  - Total trades: {len(df)}")
            
            return (
                winrate_fig,
                r_multiple_fig,
                comparison_table,
                summary_cards,
                insights,
                info_panel,
                export_data,
                probs_data
            )
            
        except Exception as e:
            print(f"  - ERROR in regime analysis: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty charts on error
            from frontend.components.regime_winrate_chart import create_empty_regime_winrate_chart
            from frontend.components.regime_r_multiple_chart import create_empty_regime_r_multiple_chart
            from frontend.components.regime_comparison_table import create_empty_regime_comparison_table
            from frontend.components.regime_filter_controls import create_empty_regime_summary_cards
            
            error_info = create_error_info_panel(str(e))
            error_insights = create_empty_insights()
            
            return (
                create_empty_regime_winrate_chart(),
                create_empty_regime_r_multiple_chart(),
                create_empty_regime_comparison_table(),
                create_empty_regime_summary_cards(),
                error_insights,
                error_info,
                None,
                None
            )
        
        except Exception as outer_e:
            # Catch any error not caught by inner try-except
            print(f"  - OUTER ERROR: {outer_e}")
            import traceback
            traceback.print_exc()
            
            # Return empty charts
            from frontend.components.regime_winrate_chart import create_empty_regime_winrate_chart
            from frontend.components.regime_r_multiple_chart import create_empty_regime_r_multiple_chart
            from frontend.components.regime_comparison_table import create_empty_regime_comparison_table
            from frontend.components.regime_filter_controls import create_empty_regime_summary_cards
            
            return (
                create_empty_regime_winrate_chart(),
                create_empty_regime_r_multiple_chart(),
                create_empty_regime_comparison_table(),
                create_empty_regime_summary_cards(),
                create_empty_insights(),
                create_error_info_panel(str(outer_e)),
                None,
                None
            )



    print(f"[REGIME EXPLORER] Main callback registered successfully")
    print(f"[REGIME EXPLORER] App now has {len(app.callback_map)} callbacks after main callback")
    
    # Export callback
    print("[REGIME EXPLORER] Registering export callback...")
    
    @app.callback(
        Output('regime-download-data', 'data'),
        Input('regime-export-btn', 'n_clicks'),
        State('regime-results-store', 'data'),
        prevent_initial_call=True
    )
    def export_regime_results(n_clicks, results_data):
        """Export regime analysis results to CSV."""
        print(f"\n{'='*60}")
        print(f"[REGIME EXPORT] Callback triggered!")
        print(f"[REGIME EXPORT] n_clicks: {n_clicks}")
        print(f"[REGIME EXPORT] results_data exists: {results_data is not None}")
        print(f"{'='*60}\n")
        
        if not n_clicks:
            print("[REGIME EXPORT] No clicks, preventing update")
            raise PreventUpdate
        
        if not results_data:
            print("[REGIME EXPORT] ERROR: No results data!")
            raise PreventUpdate
        
        try:
            from datetime import datetime
            
            print(f"[REGIME EXPORT] Starting export...")
            
            # Create comprehensive export
            export_lines = []
            
            # Header
            export_lines.append("Regime Analysis Results")
            export_lines.append(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            export_lines.append("")
            
            # Analysis Parameters
            export_lines.append("=== ANALYSIS PARAMETERS ===")
            export_lines.append(f"Regime Column: {results_data.get('regime_column', 'N/A')}")
            export_lines.append(f"Target Variable: {results_data.get('target_variable', 'N/A')}")
            export_lines.append(f"Confidence Level: {results_data.get('confidence_level', 95)}%")
            export_lines.append(f"Min Samples: {results_data.get('min_samples', 5)}")
            export_lines.append(f"Total Regimes: {results_data.get('total_regimes', 0)}")
            export_lines.append("")
            
            # Performance Summary
            export_lines.append("=== REGIME PERFORMANCE SUMMARY ===")
            best_regime = results_data.get('best_regime', {})
            worst_regime = results_data.get('worst_regime', {})
            
            if best_regime:
                export_lines.append(f"Best Regime: {best_regime.get('regime', 'N/A')}")
                export_lines.append(f"  Win Rate: {best_regime.get('win_rate', 0):.2%}")
                export_lines.append(f"  Mean R: {best_regime.get('mean_r', 0):.2f}")
                export_lines.append(f"  Sample Size: {best_regime.get('n_trades', 0)}")
            
            if worst_regime:
                export_lines.append(f"Worst Regime: {worst_regime.get('regime', 'N/A')}")
                export_lines.append(f"  Win Rate: {worst_regime.get('win_rate', 0):.2%}")
                export_lines.append(f"  Mean R: {worst_regime.get('mean_r', 0):.2f}")
                export_lines.append(f"  Sample Size: {worst_regime.get('n_trades', 0)}")
            
            if best_regime and worst_regime:
                spread = best_regime.get('win_rate', 0) - worst_regime.get('win_rate', 0)
                export_lines.append(f"Regime Spread: {spread:.2%}")
            
            export_lines.append("")
            
            # Per-Regime Details
            export_lines.append("=== PER-REGIME DETAILS ===")
            export_lines.append("Regime,Win Rate,CI Lower,CI Upper,Mean R,Median R,Std R,P(R>=1),P(R>=2),Sample Size,Reliable,Quality")
            
            regime_details = results_data.get('regime_details', [])
            for detail in regime_details:
                export_lines.append(
                    f"{detail.get('regime', '')},"
                    f"{detail.get('win_rate', 0):.4f},"
                    f"{detail.get('ci_lower', 0):.4f},"
                    f"{detail.get('ci_upper', 0):.4f},"
                    f"{detail.get('mean_r', 0):.4f},"
                    f"{detail.get('median_r', 0):.4f},"
                    f"{detail.get('std_r', 0):.4f},"
                    f"{detail.get('p_r_gte_1', 0):.4f},"
                    f"{detail.get('p_r_gte_2', 0):.4f},"
                    f"{detail.get('n_trades', 0)},"
                    f"{detail.get('reliable', False)},"
                    f"{detail.get('quality', 'N/A')}"
                )
            
            export_lines.append("")
            
            # Recommendations
            recommendations = results_data.get('recommendations', [])
            if recommendations:
                export_lines.append("=== RECOMMENDATIONS ===")
                for i, rec in enumerate(recommendations, 1):
                    export_lines.append(f"{i}. {rec}")
            
            # Create CSV content
            csv_content = "\n".join(export_lines)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            regime_col = results_data.get('regime_column', 'regime')
            filename = f"regime_analysis_{regime_col}_{timestamp}.csv"
            
            print(f"\n[REGIME EXPORT] SUCCESS!")
            print(f"  - Generated CSV with {len(export_lines)} lines")
            print(f"  - Filename: {filename}")
            print(f"  - CSV size: {len(csv_content)} characters")
            print(f"  - Returning download data...\n")
            
            return dict(content=csv_content, filename=filename)
            
        except Exception as e:
            print(f"[REGIME EXPORT] ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise PreventUpdate
    
    print("[REGIME EXPLORER] Export callback registered successfully")
    
    # Help modal callback
    print("[REGIME EXPLORER] Registering help modal callback...")
    
    @app.callback(
        Output('regime-help-modal', 'is_open'),
        [Input('regime-help-btn', 'n_clicks'), Input('regime-help-close', 'n_clicks')],
        [State('regime-help-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_regime_help_modal(n1, n2, is_open):
        """Toggle regime help modal visibility."""
        if n1 or n2:
            return not is_open
        return is_open
    
    print("[REGIME EXPLORER] Help modal callback registered successfully")


print("[OK] Regime Explorer callbacks registered - ALL 3 CALLBACKS")
