"""
Sequential Analysis Callbacks

Callbacks for Sequential Analysis tab including Markov transition matrix,
streak analysis, and conditional probability visualizations.
"""

from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objects as go
from backend.calculators.sequential_analysis import (
    compute_first_order_markov,
    compute_streak_distribution,
    compute_winrate_given_loss_streak,
    find_max_streaks
)
from backend.utils.data_cache import get_merged_data, get_trade_data


def register_sequential_analysis_callbacks(app):
    """
    Register all Sequential Analysis callbacks.
    
    Args:
        app: Dash application instance
    """
    
    @app.callback(
        [
            Output('transition-matrix-heatmap', 'figure'),
            Output('streak-distribution-chart', 'figure'),
            Output('conditional-streak-chart', 'figure'),
            Output('markov-summary-cards', 'children'),
            Output('seq-insights-content', 'children'),
            Output('markov-results-store', 'data'),
            Output('streak-results-store', 'data'),
            Output('conditional-results-store', 'data')
        ],
        [Input('merged-data-store', 'data'),
         Input('trade-data-store', 'data'),
         Input('main-tabs', 'active_tab'),
         Input('seq-calculate-btn', 'n_clicks')],
        [State('seq-target-variable-dropdown', 'value'),
         State('seq-confidence-level-slider', 'value'),
         State('seq-max-streak-slider', 'value'),
         State('seq-min-samples-slider', 'value')],
        prevent_initial_call=False
    )
    def update_sequential_analysis(merged_data, trade_data, active_tab, calc_clicks,
                                  target_var, conf_level, max_streak, min_samples):
        """Update all sequential analysis visualizations."""
        print(f"\n[Sequential Analysis] Callback triggered")
        print(f"  - merged_data: {type(merged_data)}")
        print(f"  - trade_data: {type(trade_data)}")
        print(f"  - active_tab: {active_tab}")
        print(f"  - calc_clicks: {calc_clicks}")
        print(f"  - target_var: {target_var}")
        print(f"  - conf_level: {conf_level}%")
        print(f"  - max_streak: {max_streak}")
        print(f"  - min_samples: {min_samples}")
        
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
                # Return empty charts instead of PreventUpdate
                from frontend.components.transition_matrix import create_empty_transition_matrix
                from frontend.components.streak_distribution import create_empty_streak_distribution
                from frontend.components.conditional_streak_chart import create_empty_conditional_streak_chart
                from frontend.components.markov_summary_cards import create_empty_summary_cards
                
                empty_info = create_empty_info_panel()
                return (
                    create_empty_transition_matrix(),
                    create_empty_streak_distribution(),
                    create_empty_conditional_streak_chart(),
                    create_empty_summary_cards(),
                    empty_info,
                    None,  # markov-results-store
                    None,  # streak-results-store
                    None   # conditional-results-store
                )
            data_src = cached
            print(f"  - Found data in cache: {len(cached)} rows")
        
        try:
            import io
            # Parse data
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
            
            if df.empty or target_var not in df.columns:
                print(f"  - DataFrame empty or missing {target_var}")
                raise ValueError("Invalid data")
            
            # Sort by time - CRITICAL for sequential analysis
            time_cols = ['Timestamp', 'entry_time', 'datetime', 'timestamp']
            sorted_by_time = False
            sort_column = None
            
            for c in time_cols:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors='coerce')
                    # Remove rows with invalid timestamps
                    df = df.dropna(subset=[c])
                    # Sort by time
                    df = df.sort_values(by=c).reset_index(drop=True)
                    sorted_by_time = True
                    sort_column = c
                    print(f"  - Sorted by {c}: {df[c].min()} to {df[c].max()}")
                    break
            
            # If no time column, try Ticket_id
            if not sorted_by_time and 'Ticket_id' in df.columns:
                df = df.sort_values(by='Ticket_id').reset_index(drop=True)
                sorted_by_time = True
                sort_column = 'Ticket_id'
                print(f"  - Sorted by Ticket_id: {df['Ticket_id'].min()} to {df['Ticket_id'].max()}")
            
            if not sorted_by_time:
                print(f"  - WARNING: No time/ID column found, using original order")
                print(f"  - WARNING: Sequential analysis may not be accurate!")
                sort_column = 'original_order'
            
            print(f"  - Data ready: {len(df)} rows")
            print(f"  - Success rate ({target_var}): {df[target_var].mean():.1%}")
            
            # Compute Markov transition matrix
            print(f"  - Computing Markov transition matrix...")
            markov_result = compute_first_order_markov(df, target_column=target_var, conf_level=conf_level/100)
            print(f"    P(Win|Win)={markov_result['probs']['P_win_given_win']:.2%}")
            
            # Compute streak distribution
            print(f"  - Computing streak distribution...")
            streak_result = compute_streak_distribution(df, target_column=target_var)
            print(f"    Max win streak={streak_result['max_win_streak']}, Max loss streak={streak_result['max_loss_streak']}")
            
            # Compute conditional win rate given loss streak
            print(f"  - Computing conditional win rate...")
            # If min_samples is 0, use 1 (to include all data)
            actual_min_samples = max(1, min_samples)
            conditional_result = compute_winrate_given_loss_streak(
                df, 
                target_column=target_var,
                max_streak=max_streak,
                conf_level=conf_level/100,
                min_samples=actual_min_samples
            )
            print(f"    Conditional result type: {type(conditional_result)}, shape: {conditional_result.shape if isinstance(conditional_result, pd.DataFrame) else 'N/A'}")
            
            # Find max streaks
            print(f"  - Finding max streaks...")
            max_streaks = find_max_streaks(df, target_column=target_var)
            
            # Compute win streak conditional (loss rate after win streak)
            print(f"  - Computing loss rate after win streaks...")
            win_streak_conditional = compute_lossrate_after_winstreak(df, target_var, max_streak, conf_level/100, actual_min_samples)
            print(f"    Win streak conditional computed: {len(win_streak_conditional)} data points")
            
            # Create visualizations
            print(f"  - Creating visualizations...")
            transition_fig = create_transition_heatmap(markov_result)
            streak_fig = create_streak_distribution_chart(streak_result)
            conditional_fig = create_conditional_streak_chart(conditional_result)
            # Pass streak_result instead of max_streaks to summary cards
            summary_cards = create_markov_summary_cards(markov_result, streak_result)
            
            # Create info panel with analysis details
            info_panel = create_info_panel(
                df=df,
                target_var=target_var,
                conf_level=conf_level,
                max_streak=max_streak,
                min_samples=min_samples,
                markov_result=markov_result,
                streak_result=streak_result,
                conditional_result=conditional_result,
                win_streak_conditional=win_streak_conditional
            )
            
            # Store results for export
            markov_store = {
                'probs': markov_result['probs'],
                'counts': markov_result['counts'],
                'n_transitions': markov_result['n_transitions']
            }
            
            streak_store = {
                'max_win_streak': streak_result['max_win_streak'],
                'max_loss_streak': streak_result['max_loss_streak'],
                'avg_win_streak': streak_result['avg_win_streak'],
                'avg_loss_streak': streak_result['avg_loss_streak'],
                'win_streak_distribution': streak_result['win_streak_distribution'],
                'loss_streak_distribution': streak_result['loss_streak_distribution']
            }
            
            conditional_store = conditional_result.to_dict('records')
            
            print(f"  - Sequential Analysis completed successfully!")
            return transition_fig, streak_fig, conditional_fig, summary_cards, info_panel, markov_store, streak_store, conditional_store
            
        except Exception as e:
            print(f"  - ERROR in sequential analysis: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty charts on error
            from frontend.components.transition_matrix import create_empty_transition_matrix
            from frontend.components.streak_distribution import create_empty_streak_distribution
            from frontend.components.conditional_streak_chart import create_empty_conditional_streak_chart
            from frontend.components.markov_summary_cards import create_empty_summary_cards
            
            error_info = create_error_info_panel(str(e))
            
            return (
                create_empty_transition_matrix(),
                create_empty_streak_distribution(),
                create_empty_conditional_streak_chart(),
                create_empty_summary_cards(),
                error_info,
                None,  # markov-results-store
                None,  # streak-results-store
                None   # conditional-results-store
            )
    
    # Export results callback
    @app.callback(
        Output('seq-download-data', 'data'),
        Input('seq-export-btn', 'n_clicks'),
        [State('markov-results-store', 'data'),
         State('streak-results-store', 'data'),
         State('conditional-results-store', 'data'),
         State('seq-target-variable-dropdown', 'value')],
        prevent_initial_call=True
    )
    def export_sequential_results(n_clicks, markov_data, streak_data, conditional_data, target_var):
        """Export sequential analysis results to CSV"""
        print(f"\n{'='*60}")
        print(f"[EXPORT] Callback triggered!")
        print(f"[EXPORT] n_clicks: {n_clicks}")
        print(f"[EXPORT] markov_data exists: {markov_data is not None}")
        print(f"[EXPORT] streak_data exists: {streak_data is not None}")
        print(f"[EXPORT] conditional_data exists: {conditional_data is not None}")
        print(f"[EXPORT] target_var: {target_var}")
        print(f"{'='*60}\n")
        
        if not n_clicks:
            print("[EXPORT] No clicks, preventing update")
            raise PreventUpdate
        
        if not markov_data or not streak_data or not conditional_data:
            print(f"[EXPORT] ERROR: Missing data!")
            print(f"  - markov_data: {bool(markov_data)}")
            print(f"  - streak_data: {bool(streak_data)}")
            print(f"  - conditional_data: {bool(conditional_data)}")
            raise PreventUpdate
        
        print(f"[EXPORT] Starting export for target_var={target_var}")
        
        from datetime import datetime
        import io
        
        # Create export data
        export_lines = []
        export_lines.append("Sequential Analysis Results")
        export_lines.append(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        export_lines.append(f"Target Variable: {target_var}")
        export_lines.append("")
        
        # Markov Transition Matrix
        if markov_data:
            export_lines.append("=== MARKOV TRANSITION MATRIX ===")
            export_lines.append("Transition,Probability,Count")
            probs = markov_data.get('probs', {})
            counts = markov_data.get('counts', {})
            export_lines.append(f"Success->Success,{probs.get('P_win_given_win', 0):.4f},{counts.get('win_to_win', 0)}")
            export_lines.append(f"Success->Failure,{probs.get('P_loss_given_win', 0):.4f},{counts.get('win_to_loss', 0)}")
            export_lines.append(f"Failure->Success,{probs.get('P_win_given_loss', 0):.4f},{counts.get('loss_to_win', 0)}")
            export_lines.append(f"Failure->Failure,{probs.get('P_loss_given_loss', 0):.4f},{counts.get('loss_to_loss', 0)}")
            export_lines.append("")
        
        # Streak Distribution
        if streak_data:
            export_lines.append("=== STREAK DISTRIBUTION ===")
            export_lines.append(f"Max Win Streak: {streak_data.get('max_win_streak', 0)}")
            export_lines.append(f"Max Loss Streak: {streak_data.get('max_loss_streak', 0)}")
            export_lines.append(f"Avg Win Streak: {streak_data.get('avg_win_streak', 0):.2f}")
            export_lines.append(f"Avg Loss Streak: {streak_data.get('avg_loss_streak', 0):.2f}")
            export_lines.append("")
            
            # Win streaks detail
            export_lines.append("Win Streak Length,Frequency,Percentage")
            win_dist = streak_data.get('win_streak_distribution', {})
            total_win = sum(win_dist.values())
            for length in sorted(win_dist.keys()):
                freq = win_dist[length]
                pct = freq / total_win * 100 if total_win > 0 else 0
                export_lines.append(f"{length},{freq},{pct:.2f}%")
            export_lines.append("")
            
            # Loss streaks detail
            export_lines.append("Loss Streak Length,Frequency,Percentage")
            loss_dist = streak_data.get('loss_streak_distribution', {})
            total_loss = sum(loss_dist.values())
            for length in sorted(loss_dist.keys()):
                freq = loss_dist[length]
                pct = freq / total_loss * 100 if total_loss > 0 else 0
                export_lines.append(f"{length},{freq},{pct:.2f}%")
            export_lines.append("")
        
        # Conditional Probabilities
        if conditional_data:
            export_lines.append("=== CONDITIONAL PROBABILITIES ===")
            export_lines.append("Loss Streak Length,Win Rate,Opportunities,Wins,CI Lower,CI Upper,Reliable")
            for row in conditional_data:
                # Handle None values safely
                loss_streak = row.get('loss_streak_length', 0) if row.get('loss_streak_length') is not None else 0
                win_rate = row.get('win_rate', 0) if row.get('win_rate') is not None else 0
                n_opp = row.get('n_opportunities', 0) if row.get('n_opportunities') is not None else 0
                n_wins = row.get('n_wins', 0) if row.get('n_wins') is not None else 0
                ci_lower = row.get('ci_lower', 0) if row.get('ci_lower') is not None else 0
                ci_upper = row.get('ci_upper', 0) if row.get('ci_upper') is not None else 0
                reliable = row.get('reliable', False) if row.get('reliable') is not None else False
                
                export_lines.append(
                    f"{loss_streak},"
                    f"{win_rate:.4f},"
                    f"{n_opp},"
                    f"{n_wins},"
                    f"{ci_lower:.4f},"
                    f"{ci_upper:.4f},"
                    f"{reliable}"
                )
        
        # Create CSV content
        csv_content = "\n".join(export_lines)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"sequential_analysis_{target_var}_{timestamp}.csv"
        
        print(f"\n[EXPORT] SUCCESS!")
        print(f"  - Generated CSV with {len(export_lines)} lines")
        print(f"  - Filename: {filename}")
        print(f"  - CSV size: {len(csv_content)} characters")
        print(f"  - Returning download data...\n")
        
        return dict(content=csv_content, filename=filename)
    
    # Help modal callback
    @app.callback(
        Output('seq-help-modal', 'is_open'),
        [Input('seq-help-btn', 'n_clicks'), Input('seq-help-close', 'n_clicks')],
        [State('seq-help-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_seq_help_modal(n1, n2, is_open):
        """Toggle sequential analysis help modal visibility."""
        if n1 or n2:
            return not is_open
        return is_open


def create_transition_heatmap(markov_result):
    """Create Markov transition matrix heatmap."""
    probs = markov_result.get('probs', {})
    
    # Create matrix
    matrix = [
        [probs.get('P_win_given_win', 0), probs.get('P_loss_given_win', 0)],
        [probs.get('P_win_given_loss', 0), probs.get('P_loss_given_loss', 0)]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=['Win', 'Loss'],
        y=['Win', 'Loss'],
        colorscale='RdYlGn',
        text=[[f'{val:.2%}' for val in row] for row in matrix],
        texttemplate='%{text}',
        textfont={"size": 16},
        colorbar=dict(title="Probability")
    ))
    
    fig.update_layout(
        title='Markov Transition Matrix',
        xaxis_title='Next Trade',
        yaxis_title='Current Trade',
        height=400
    )
    
    return fig


def create_streak_distribution_chart(streak_result):
    """Create streak distribution bar chart."""
    win_streaks = streak_result.get('win_streaks', [])
    loss_streaks = streak_result.get('loss_streaks', [])
    
    # Count streak lengths
    from collections import Counter
    win_counts = Counter(win_streaks)
    loss_counts = Counter(loss_streaks)
    
    max_streak = max(max(win_counts.keys(), default=0), max(loss_counts.keys(), default=0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(range(1, max_streak + 1)),
        y=[win_counts.get(i, 0) for i in range(1, max_streak + 1)],
        name='Win Streaks',
        marker_color='green'
    ))
    
    fig.add_trace(go.Bar(
        x=list(range(1, max_streak + 1)),
        y=[loss_counts.get(i, 0) for i in range(1, max_streak + 1)],
        name='Loss Streaks',
        marker_color='red'
    ))
    
    fig.update_layout(
        title='Streak Length Distribution',
        xaxis_title='Streak Length',
        yaxis_title='Frequency',
        barmode='group',
        height=400
    )
    
    return fig


def create_conditional_streak_chart(conditional_result):
    """Create conditional win rate given loss streak chart."""
    # conditional_result is a DataFrame, not a dict
    if isinstance(conditional_result, pd.DataFrame):
        # Filter out rows with no data
        valid_data = conditional_result[conditional_result['n_opportunities'] > 0].copy()
        
        if valid_data.empty:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title='Win Rate After Loss Streak',
                xaxis_title='Loss Streak Length',
                yaxis_title='Win Rate',
                height=400
            )
            return fig
        
        streaks = valid_data['loss_streak_length'].tolist()
        win_rates = valid_data['win_rate'].tolist()
        ci_lower = valid_data['ci_lower'].tolist()
        ci_upper = valid_data['ci_upper'].tolist()
    else:
        # Fallback for dict format (shouldn't happen)
        streaks = list(conditional_result.keys())
        win_rates = [conditional_result[k]['win_rate'] for k in streaks]
        ci_lower = [conditional_result[k].get('ci_lower', 0) for k in streaks]
        ci_upper = [conditional_result[k].get('ci_upper', 1) for k in streaks]
    
    fig = go.Figure()
    
    # Add confidence interval band
    fig.add_trace(go.Scatter(
        x=streaks + streaks[::-1],
        y=ci_upper + ci_lower[::-1],
        fill='toself',
        fillcolor='rgba(0,100,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Add win rate line
    fig.add_trace(go.Scatter(
        x=streaks,
        y=win_rates,
        mode='lines+markers',
        name='Win Rate',
        line=dict(color='blue', width=2),
        marker=dict(size=8),
        hovertemplate='Streak: %{x}<br>Win Rate: %{y:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Win Rate After Loss Streak',
        xaxis_title='Loss Streak Length',
        yaxis_title='Win Rate',
        yaxis_tickformat='.0%',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_markov_summary_cards(markov_result, max_streaks):
    """Create summary cards for Markov metrics."""
    import dash_bootstrap_components as dbc
    from dash import html
    
    probs = markov_result.get('probs', {})
    
    cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("P(Win | Win)", className="text-muted"),
                    html.H3(f"{probs.get('P_win_given_win', 0):.1%}", className="text-success")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("P(Win | Loss)", className="text-muted"),
                    html.H3(f"{probs.get('P_win_given_loss', 0):.1%}", className="text-primary")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Max Win Streak", className="text-muted"),
                    html.H3(f"{max_streaks.get('max_win_streak', 0)}", className="text-success")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Max Loss Streak", className="text-muted"),
                    html.H3(f"{max_streaks.get('max_loss_streak', 0)}", className="text-danger")
                ])
            ])
        ], width=3),
    ])
    
    return cards


def compute_lossrate_after_winstreak(df, target_column, max_streak, conf_level, min_samples):
    """
    Calculate P(Loss | win_streak = k) for various win streak lengths.
    Similar to compute_winrate_given_loss_streak but for win streaks.
    """
    if df.empty or len(df) < 2:
        return []
    
    outcomes = df[target_column].values
    results = []
    
    for k in range(max_streak + 1):
        n_opportunities = 0
        n_losses = 0
        
        # Scan through outcomes looking for win streaks of length k
        current_win_streak = 0
        
        for i in range(len(outcomes)):
            if i == 0:
                if outcomes[i] == 1:
                    current_win_streak = 1
                else:
                    current_win_streak = 0
                continue
            
            # Check if previous trade was a win
            if outcomes[i - 1] == 1:
                current_win_streak += 1
            else:
                current_win_streak = 0
            
            # If we have exactly k consecutive wins before this trade
            if current_win_streak == k:
                n_opportunities += 1
                if outcomes[i] == 0:  # Next trade is loss
                    n_losses += 1
        
        # Calculate loss rate and CI
        if n_opportunities > 0:
            loss_rate = n_losses / n_opportunities
            
            # Calculate CI using beta distribution
            from backend.models.confidence_intervals import beta_posterior_ci
            ci_result = beta_posterior_ci(
                successes=n_losses,
                total=n_opportunities,
                conf_level=conf_level
            )
            
            results.append({
                'win_streak_length': k,
                'n_opportunities': n_opportunities,
                'n_losses': n_losses,
                'loss_rate': loss_rate,
                'ci_lower': ci_result['ci_lower'],
                'ci_upper': ci_result['ci_upper'],
                'reliable': n_opportunities >= min_samples
            })
        else:
            results.append({
                'win_streak_length': k,
                'n_opportunities': 0,
                'n_losses': 0,
                'loss_rate': 0,
                'ci_lower': 0,
                'ci_upper': 0,
                'reliable': False
            })
    
    return results


def create_info_panel(df, target_var, conf_level, max_streak, min_samples, 
                     markov_result, streak_result, conditional_result, win_streak_conditional=None):
    """Create detailed info panel showing analysis parameters and results"""
    import dash_bootstrap_components as dbc
    from dash import html
    
    # Calculate statistics
    total_trades = len(df)
    success_rate = df[target_var].mean()
    n_successes = int(df[target_var].sum())
    n_failures = total_trades - n_successes
    
    # Get time range info
    time_info = ""
    if 'Timestamp' in df.columns:
        time_range = f"{df['Timestamp'].min().strftime('%Y-%m-%d')} to {df['Timestamp'].max().strftime('%Y-%m-%d')}"
        time_info = f"Time Range: {time_range}"
    elif 'entry_time' in df.columns:
        time_range = f"{df['entry_time'].min().strftime('%Y-%m-%d')} to {df['entry_time'].max().strftime('%Y-%m-%d')}"
        time_info = f"Time Range: {time_range}"
    elif 'Ticket_id' in df.columns:
        time_info = f"Ticket ID Range: {df['Ticket_id'].min()} to {df['Ticket_id'].max()}"
    
    # Calculate R_multiple stats if available
    r_stats = ""
    if 'R_multiple' in df.columns:
        avg_r = df['R_multiple'].mean()
        median_r = df['R_multiple'].median()
        max_r = df['R_multiple'].max()
        min_r = df['R_multiple'].min()
        r_stats = f"R-Multiple: Avg={avg_r:.2f}, Median={median_r:.2f}, Range=[{min_r:.2f}, {max_r:.2f}]"
    
    # Get Markov stats
    probs = markov_result['probs']
    counts = markov_result['counts']
    n_transitions = markov_result['n_transitions']
    
    # Get streak stats
    max_win = streak_result['max_win_streak']
    max_loss = streak_result['max_loss_streak']
    avg_win = streak_result['avg_win_streak']
    avg_loss = streak_result['avg_loss_streak']
    n_win_streaks = len(streak_result['win_streaks'])
    n_loss_streaks = len(streak_result['loss_streaks'])
    
    # Get conditional stats
    reliable_rows = conditional_result[conditional_result['reliable']]
    n_reliable_streaks = len(reliable_rows)
    
    # Target variable label and explanation
    target_labels = {
        'trade_success': 'Trade Success (Win/Loss)',
        'y_hit_1R': 'Hit 1R Target (R_multiple ≥ 1.0)',
        'y_hit_2R': 'Hit 2R Target (R_multiple ≥ 2.0)'
    }
    target_label = target_labels.get(target_var, target_var)
    
    target_explanations = {
        'trade_success': 'Win = R_multiple > 0 (any profit)',
        'y_hit_1R': 'Success = R_multiple ≥ 1.0 (at least 1x risk)',
        'y_hit_2R': 'Success = R_multiple ≥ 2.0 (at least 2x risk)'
    }
    target_explanation = target_explanations.get(target_var, '')
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-info-circle me-2"),
                "Analysis Information & Data Summary"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            # Analysis Parameters
            dbc.Row([
                dbc.Col([
                    html.H6("Analysis Parameters", className="text-primary mb-3"),
                    html.Ul([
                        html.Li([html.Strong("Target Variable: "), target_label]),
                        html.Li([html.Strong("Confidence Level: "), f"{conf_level}%"]),
                        html.Li([html.Strong("Max Streak Length: "), f"{max_streak}"]),
                        html.Li([html.Strong("Min Samples: "), f"{min_samples}"]),
                    ], className="mb-0")
                ], md=6),
                dbc.Col([
                    html.H6("Data Summary", className="text-success mb-3"),
                    html.Ul([
                        html.Li([html.Strong("Total Trades: "), f"{total_trades:,}"]),
                        html.Li([html.Strong("Successes: "), f"{n_successes:,} ({success_rate:.1%})"]),
                        html.Li([html.Strong("Failures: "), f"{n_failures:,} ({(1-success_rate):.1%})"]),
                        html.Li([html.Strong("Transitions Analyzed: "), f"{n_transitions:,}"]),
                    ], className="mb-0"),
                    html.Hr(className="my-2"),
                    html.Small([
                        html.Strong("Data Order: "),
                        "Sorted by time/Ticket_id for sequential analysis"
                    ], className="text-muted d-block"),
                    html.Small([
                        html.Strong(target_explanation)
                    ], className="text-info d-block mt-1"),
                    html.Small(time_info, className="text-muted d-block mt-1") if time_info else None,
                    html.Small(r_stats, className="text-muted d-block mt-1") if r_stats else None,
                ], md=6)
            ], className="mb-3"),
            
            html.Hr(),
            
            # Markov Analysis Results
            dbc.Row([
                dbc.Col([
                    html.H6("Markov Transition Results", className="text-info mb-3"),
                    dbc.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Transition"),
                                html.Th("Probability"),
                                html.Th("Count"),
                                html.Th("Interpretation")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td("Success → Success"),
                                html.Td(f"{probs['P_win_given_win']:.1%}"),
                                html.Td(f"{counts['win_to_win']}"),
                                html.Td("Momentum effect" if probs['P_win_given_win'] > 0.5 else "No momentum")
                            ]),
                            html.Tr([
                                html.Td("Success → Failure"),
                                html.Td(f"{probs['P_loss_given_win']:.1%}"),
                                html.Td(f"{counts['win_to_loss']}"),
                                html.Td("After success")
                            ]),
                            html.Tr([
                                html.Td("Failure → Success"),
                                html.Td(f"{probs['P_win_given_loss']:.1%}"),
                                html.Td(f"{counts['loss_to_win']}"),
                                html.Td("Recovery rate" if probs['P_win_given_loss'] > 0.5 else "Poor recovery")
                            ]),
                            html.Tr([
                                html.Td("Failure → Failure"),
                                html.Td(f"{probs['P_loss_given_loss']:.1%}"),
                                html.Td(f"{counts['loss_to_loss']}"),
                                html.Td("Loss clustering" if probs['P_loss_given_loss'] > 0.5 else "No clustering")
                            ])
                        ])
                    ], bordered=True, hover=True, size='sm')
                ], md=12)
            ], className="mb-3"),
            
            html.Hr(),
            
            # Streak Analysis Results
            dbc.Row([
                dbc.Col([
                    html.H6("Streak Analysis Results", className="text-warning mb-3"),
                    dbc.Alert([
                        html.I(className="bi bi-info-circle me-2"),
                        html.Strong("What is a Streak? "),
                        f"A streak is a group of consecutive wins or losses. "
                        f"Example: W-W-W-L-L = 1 win streak (length 3) + 1 loss streak (length 2). "
                        f"Your {total_trades} trades are grouped into {n_win_streaks + n_loss_streaks} streaks total."
                    ], color="info", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Strong("Win Streaks:", className="text-success"),
                                html.Br(),
                                f"• Max: {max_win} consecutive wins",
                                html.Br(),
                                f"• Average: {avg_win:.2f} wins per streak",
                                html.Br(),
                                f"• Total groups: {n_win_streaks} win streaks",
                                html.Br(),
                                html.Small([
                                    f"({n_successes} total wins grouped into {n_win_streaks} streaks)"
                                ], className="text-muted")
                            ])
                        ], md=6),
                        dbc.Col([
                            html.Div([
                                html.Strong("Loss Streaks:", className="text-danger"),
                                html.Br(),
                                f"• Max: {max_loss} consecutive losses",
                                html.Br(),
                                f"• Average: {avg_loss:.2f} losses per streak",
                                html.Br(),
                                f"• Total groups: {n_loss_streaks} loss streaks",
                                html.Br(),
                                html.Small([
                                    f"({n_failures} total losses grouped into {n_loss_streaks} streaks)"
                                ], className="text-muted")
                            ])
                        ], md=6)
                    ])
                ], md=12)
            ], className="mb-3"),
            
            html.Hr(),
            
            # Detailed Streak Tables
            dbc.Row([
                dbc.Col([
                    html.H6("Win Streak Distribution Table", className="text-success mb-3"),
                    html.Small("Shows frequency and loss probability AFTER each win streak length", className="text-muted d-block mb-2"),
                    create_streak_table(streak_result, 'win', avg_win, win_streak_conditional)
                ], md=6),
                dbc.Col([
                    html.H6("Loss Streak Distribution Table", className="text-danger mb-3"),
                    html.Small("Shows frequency and win probability AFTER each loss streak length", className="text-muted d-block mb-2"),
                    create_streak_table(streak_result, 'loss', avg_loss, conditional_result.to_dict('records') if isinstance(conditional_result, pd.DataFrame) else conditional_result)
                ], md=6)
            ], className="mb-3"),
            
            html.Hr(),
            
            # Conditional Probability Results
            dbc.Row([
                dbc.Col([
                    html.H6("Conditional Probability Results", className="text-danger mb-3"),
                    html.P([
                        html.Strong("Reliable Data Points: "),
                        f"{n_reliable_streaks} out of {len(conditional_result)} streak lengths have sufficient samples (≥{min_samples})"
                    ]),
                    html.P([
                        html.Strong("Analysis Range: "),
                        f"Loss streaks from 0 to {max_streak} consecutive failures"
                    ]),
                    html.Small([
                        html.I(className="bi bi-lightbulb me-2"),
                        "The conditional probability chart shows how success rate changes after consecutive failures. "
                        "Upward trend = mean reversion, Downward trend = momentum, Flat = independence."
                    ], className="text-muted")
                ], md=12)
            ]),
            
            html.Hr(),
            
            # Reliability Assessment
            dbc.Row([
                dbc.Col([
                    html.H6("Reliability Assessment", className="text-info mb-3"),
                    html.Div(children=generate_reliability_assessment(
                        total_trades, n_transitions, markov_result, 
                        streak_result, conditional_result, min_samples
                    ))
                ], md=12)
            ], className="mb-3"),
            
            html.Hr(),
            
            # Key Insights
            dbc.Row([
                dbc.Col([
                    html.H6("Key Insights & Recommendations", className="text-success mb-3"),
                    html.Div(id='auto-insights', children=generate_insights(
                        markov_result, streak_result, success_rate, total_trades
                    ))
                ], md=12)
            ])
        ])
    ], className="mb-3")


def create_streak_table(streak_result, streak_type='win', avg_streak=0, conditional_data=None):
    """Create detailed streak distribution table with win probability after streak"""
    import dash_bootstrap_components as dbc
    from dash import html, dash_table
    import pandas as pd
    
    if streak_type == 'win':
        dist = streak_result.get('win_streak_distribution', {})
        color = 'success'
        label = 'Win'
        example = 'W-W-W'
        # For win streaks, we want P(Loss | after Win Streak = k)
        # This shows probability of losing after winning streak
        show_conditional = True  # Show loss probability after win streak
        conditional_label = 'Loss'
    else:
        dist = streak_result.get('loss_streak_distribution', {})
        color = 'danger'
        label = 'Loss'
        example = 'L-L-L'
        # For loss streaks, we want P(Win | after Loss Streak = k)
        show_conditional = True  # Very meaningful!
        conditional_label = 'Win'
    
    if not dist:
        return dbc.Alert("No data available", color="light")
    
    # Calculate total and cumulative
    total_streaks = sum(dist.values())
    
    # Calculate total individual outcomes
    total_outcomes = sum(length * count for length, count in dist.items())
    
    # Create conditional probability lookup
    conditional_lookup = {}
    if show_conditional and conditional_data is not None:
        if streak_type == 'loss':
            # For loss streaks: P(Win | after Loss Streak = k)
            for row in conditional_data:
                length = row['loss_streak_length']
                win_rate = row['win_rate']
                n_opp = row['n_opportunities']
                reliable = row['reliable']
                conditional_lookup[length] = {
                    'probability': win_rate,
                    'n_opportunities': n_opp,
                    'reliable': reliable
                }
        else:
            # For win streaks: P(Loss | after Win Streak = k)
            # conditional_data here should be win streak conditional data
            if isinstance(conditional_data, list):
                for row in conditional_data:
                    length = row.get('win_streak_length', 0)
                    loss_rate = row.get('loss_rate', 0)
                    n_opp = row.get('n_opportunities', 0)
                    reliable = row.get('reliable', False)
                    conditional_lookup[length] = {
                        'probability': loss_rate,
                        'n_opportunities': n_opp,
                        'reliable': reliable
                    }
    
    # Create data for DataTable (sortable)
    table_data = []
    
    for length in sorted(dist.keys()):  # All streak lengths
        count = dist[length]
        percentage = count / total_streaks * 100
        
        row_data = {
            'Streak': length,
            'Frequency': count,
            'Distribution': f"{percentage:.2f}%",
            'Distribution_num': percentage  # For sorting
        }
        
        # Get conditional probability if available
        if show_conditional and length in conditional_lookup:
            cond = conditional_lookup[length]
            prob = cond['probability'] * 100
            n_opp = cond['n_opportunities']
            reliable = cond['reliable']
            
            # Determine quality based on what we're measuring
            if streak_type == 'loss':
                # For loss streaks: Higher win probability = BAIK
                if prob >= 70:
                    quality = "BAIK"
                elif prob >= 50:
                    quality = "SEDANG"
                else:
                    quality = "BURUK"
            else:
                # For win streaks: Lower loss probability = BAIK (inverted from loss streak logic)
                if prob < 50:  # <50% loss = BAIK (good, low loss rate)
                    quality = "BAIK"
                elif prob <= 70:  # 50-70% loss = SEDANG (moderate)
                    quality = "SEDANG"
                else:  # >70% loss = BURUK (bad, high loss rate)
                    quality = "BURUK"
            
            row_data[f'{conditional_label}_Prob'] = f"{prob:.1f}%"
            row_data[f'{conditional_label}_Prob_num'] = prob  # For sorting
            row_data['Opportunities'] = n_opp
            row_data['Quality'] = quality
            row_data['Reliable'] = reliable
        else:
            row_data[f'{conditional_label}_Prob'] = "N/A"
            row_data[f'{conditional_label}_Prob_num'] = -1
            row_data['Opportunities'] = 0
            row_data['Quality'] = "N/A"
            row_data['Reliable'] = False
        
        table_data.append(row_data)
    
    # Create DataFrame for easier manipulation
    df_table = pd.DataFrame(table_data)
    
    # Define columns for DataTable
    conditional_prob_col = f'{conditional_label}_Prob'
    
    columns = [
        {'name': 'Streak', 'id': 'Streak', 'type': 'numeric'},
        {'name': 'Frequency', 'id': 'Frequency', 'type': 'numeric'},
        {'name': 'Distribution', 'id': 'Distribution', 'type': 'text'},
        {'name': f'{conditional_label} Prob', 'id': conditional_prob_col, 'type': 'text'},
        {'name': 'Opportunities', 'id': 'Opportunities', 'type': 'numeric'},
        {'name': 'Quality', 'id': 'Quality', 'type': 'text'}
    ]
    
    # Create DataTable with sorting
    table = dash_table.DataTable(
        data=df_table.to_dict('records'),
        columns=columns,
        sort_action='native',  # Enable sorting
        sort_mode='single',
        style_table={'maxHeight': '400px', 'overflowY': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '8px',
            'fontSize': '0.9rem'
        },
        style_header={
            'backgroundColor': '#f8f9fa',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'border': '1px solid #dee2e6'
        },
        style_data_conditional=[
            # Reliable data - blue background
            {
                'if': {'filter_query': '{Reliable} = true'},
                'backgroundColor': '#f0f8ff'
            },
            # Unreliable data - yellow background
            {
                'if': {'filter_query': '{Reliable} = false'},
                'backgroundColor': '#fff8f0'
            },
            # Quality colors
            {
                'if': {'filter_query': '{Quality} = "BAIK"', 'column_id': 'Quality'},
                'backgroundColor': '#d4edda',
                'color': '#155724',
                'fontWeight': 'bold'
            },
            {
                'if': {'filter_query': '{Quality} = "SEDANG"', 'column_id': 'Quality'},
                'backgroundColor': '#fff3cd',
                'color': '#856404',
                'fontWeight': 'bold'
            },
            {
                'if': {'filter_query': '{Quality} = "BURUK"', 'column_id': 'Quality'},
                'backgroundColor': '#f8d7da',
                'color': '#721c24',
                'fontWeight': 'bold'
            }
        ],
        style_data={
            'border': '1px solid #dee2e6'
        }
    )
    
    # Create explanation based on type
    if show_conditional:
        if streak_type == 'loss':
            explanation = dbc.Alert([
                html.I(className="bi bi-lightbulb me-2"),
                html.Strong("How to Read: "),
                html.Br(),
                "• ",
                html.Strong("Streak Length 7: "),
                "Kalah 7x berturut-turut",
                html.Br(),
                "• ",
                html.Strong("Frequency 10: "),
                "Terjadi 10 kali",
                html.Br(),
                "• ",
                html.Strong("Win Probability 70%: "),
                "Dari 10 kali loss streak 7, 7 kali diikuti WIN (70%)",
                html.Br(),
                "• ",
                html.Strong("Quality BAIK: "),
                "≥70% = BAIK, 50-70% = SEDANG, <50% = BURUK",
                html.Br(),
                html.Small([
                    html.I(className="bi bi-star me-1"),
                    "Blue background = Reliable data (≥min samples), Yellow = Low samples"
                ], className="text-muted")
            ], color="light", className="mb-2 small")
        else:
            explanation = dbc.Alert([
                html.I(className="bi bi-lightbulb me-2"),
                html.Strong("How to Read: "),
                html.Br(),
                "• ",
                html.Strong("Streak Length 3: "),
                "Menang 3x berturut-turut",
                html.Br(),
                "• ",
                html.Strong("Frequency 28: "),
                "Terjadi 28 kali",
                html.Br(),
                "• ",
                html.Strong("Loss Probability 60%: "),
                "Dari 28 kali win streak 3, 17 kali diikuti LOSS (60%)",
                html.Br(),
                "• ",
                html.Strong("Quality: "),
                "<50% = BAIK, 50-70% = SEDANG, >70% = BURUK (for loss probability)",
                html.Br(),
                html.Small([
                    html.I(className="bi bi-star me-1"),
                    "Blue background = Reliable data (≥min samples), Yellow = Low samples"
                ], className="text-muted")
            ], color="light", className="mb-2 small")
    else:
        explanation = dbc.Alert([
            html.I(className="bi bi-lightbulb me-2"),
            html.Strong(f"Example: "),
            f"Streak length 3 means {example} (3 consecutive {label.lower()}s). "
            f"If this happened 28 times, you had 28 groups of 3 consecutive {label.lower()}s."
        ], color="light", className="mb-2 small")
    
    return html.Div([
        explanation,
        table,
        html.Div([
            html.Small([
                html.Strong(f"Total {label} Streaks (Groups): "),
                f"{total_streaks:,}"
            ], className="text-muted d-block"),
            html.Small([
                html.Strong(f"Total Individual {label}s: "),
                f"{total_outcomes:,}",
                html.Span(f" (from {total_streaks:,} groups)", className="text-muted")
            ], className="text-muted d-block"),
            html.Small([
                html.I(className="bi bi-info-circle me-1"),
                f"Average {avg_streak:.2f} {label.lower()}s per streak"
            ], className="text-info d-block mt-1")
        ], className="mt-2")
    ])


def generate_reliability_assessment(total_trades, n_transitions, markov_result, 
                                   streak_result, conditional_result, min_samples):
    """Generate reliability assessment of the analysis"""
    import dash_bootstrap_components as dbc
    from dash import html
    
    assessments = []
    
    # Sample size assessment
    if total_trades < 100:
        assessments.append(dbc.Alert([
            html.I(className="bi bi-exclamation-triangle me-2"),
            html.Strong("Small Sample Size: "),
            f"Only {total_trades} trades analyzed. Results may not be statistically significant. "
            f"Recommended: ≥100 trades for reliable analysis, ≥500 for high confidence."
        ], color="warning"))
    elif total_trades < 500:
        assessments.append(dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            html.Strong("Moderate Sample Size: "),
            f"{total_trades} trades analyzed. Results are moderately reliable. "
            f"More data (≥500 trades) would increase confidence."
        ], color="info"))
    else:
        assessments.append(dbc.Alert([
            html.I(className="bi bi-check-circle me-2"),
            html.Strong("Good Sample Size: "),
            f"{total_trades} trades analyzed. Results are statistically reliable."
        ], color="success"))
    
    # Transition count assessment
    counts = markov_result['counts']
    min_transition_count = min(counts.values())
    
    if min_transition_count < 10:
        assessments.append(dbc.Alert([
            html.I(className="bi bi-exclamation-triangle me-2"),
            html.Strong("Low Transition Counts: "),
            f"Some transitions have <10 occurrences (min={min_transition_count}). "
            f"Probabilities may be unreliable. Collect more data."
        ], color="warning"))
    elif min_transition_count < 30:
        assessments.append(dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            html.Strong("Moderate Transition Counts: "),
            f"Minimum transition count: {min_transition_count}. "
            f"Probabilities are moderately reliable. More data recommended."
        ], color="info"))
    
    # Conditional probability reliability
    reliable_rows = conditional_result[conditional_result['reliable']]
    reliability_pct = len(reliable_rows) / len(conditional_result) * 100
    
    if reliability_pct < 50:
        assessments.append(dbc.Alert([
            html.I(className="bi bi-exclamation-triangle me-2"),
            html.Strong("Low Conditional Reliability: "),
            f"Only {reliability_pct:.0f}% of streak lengths have ≥{min_samples} samples. "
            f"Consider reducing 'Min Samples' or 'Max Streak Length' settings."
        ], color="warning"))
    
    # Confidence interval width assessment
    ci = markov_result['ci']
    avg_ci_width = sum([
        ci['P_win_given_win']['ci_upper'] - ci['P_win_given_win']['ci_lower'],
        ci['P_win_given_loss']['ci_upper'] - ci['P_win_given_loss']['ci_lower']
    ]) / 2
    
    if avg_ci_width > 0.3:  # >30% width
        assessments.append(dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            html.Strong("Wide Confidence Intervals: "),
            f"Average CI width: {avg_ci_width:.1%}. High uncertainty in probability estimates. "
            f"More data or lower confidence level would narrow intervals."
        ], color="info"))
    
    # Overall assessment
    if total_trades >= 500 and min_transition_count >= 30 and reliability_pct >= 70:
        assessments.append(dbc.Alert([
            html.I(className="bi bi-trophy me-2"),
            html.Strong("High Quality Analysis: "),
            f"Sufficient data ({total_trades} trades), good transition counts (min={min_transition_count}), "
            f"and {reliability_pct:.0f}% reliable conditional probabilities. Results are trustworthy."
        ], color="success"))
    
    return assessments


def generate_insights(markov_result, streak_result, success_rate, total_trades):
    """Generate automatic insights from analysis results"""
    import dash_bootstrap_components as dbc
    from dash import html
    
    probs = markov_result['probs']
    counts = markov_result['counts']
    insights = []
    
    # Momentum vs Mean Reversion (with sample counts)
    if probs['P_win_given_win'] > probs['P_win_given_loss'] + 0.05:
        insights.append(dbc.Alert([
            html.I(className="bi bi-arrow-up-circle me-2"),
            html.Strong("Momentum Effect Detected: "),
            f"Success rate after success ({probs['P_win_given_win']:.1%}, n={counts['win_to_win']}) "
            f"is significantly higher than after failure ({probs['P_win_given_loss']:.1%}, n={counts['loss_to_win']}). ",
            html.Br(),
            html.Strong("Recommendation: "),
            f"Consider trading more aggressively after wins. "
            f"This pattern occurred {counts['win_to_win']} times out of {counts['win_to_win'] + counts['win_to_loss']} opportunities after wins."
        ], color="success"))
    elif probs['P_win_given_loss'] > probs['P_win_given_win'] + 0.05:
        insights.append(dbc.Alert([
            html.I(className="bi bi-arrow-counterclockwise me-2"),
            html.Strong("Mean Reversion Detected: "),
            f"Success rate after failure ({probs['P_win_given_loss']:.1%}, n={counts['loss_to_win']}) "
            f"is higher than after success ({probs['P_win_given_win']:.1%}, n={counts['win_to_win']}). ",
            html.Br(),
            html.Strong("Recommendation: "),
            f"System recovers well after losses. Good entry points after losing streaks. "
            f"This recovery happened {counts['loss_to_win']} times out of {counts['loss_to_win'] + counts['loss_to_loss']} opportunities after losses."
        ], color="info"))
    else:
        insights.append(dbc.Alert([
            html.I(className="bi bi-shuffle me-2"),
            html.Strong("Independent Outcomes: "),
            f"Success rates after wins ({probs['P_win_given_win']:.1%}, n={counts['win_to_win']}) "
            f"and losses ({probs['P_win_given_loss']:.1%}, n={counts['loss_to_win']}) are similar. ",
            html.Br(),
            html.Strong("Interpretation: "),
            "Outcomes appear independent (random walk). Past results don't predict future outcomes."
        ], color="secondary"))
    
    # Streak warnings with context
    max_loss = streak_result['max_loss_streak']
    avg_loss = streak_result['avg_loss_streak']
    n_loss_streaks = len(streak_result['loss_streaks'])
    
    if max_loss >= 10:
        insights.append(dbc.Alert([
            html.I(className="bi bi-exclamation-triangle me-2"),
            html.Strong("High Drawdown Risk: "),
            f"Maximum loss streak of {max_loss} consecutive failures detected (average: {avg_loss:.1f}). ",
            html.Br(),
            html.Strong("Risk Assessment: "),
            f"Out of {n_loss_streaks} losing streaks, the worst was {max_loss} consecutive losses. "
            f"Ensure your risk management can handle {max_loss}+ consecutive losses. "
            f"With {success_rate:.1%} success rate, probability of {max_loss} consecutive losses is approximately {((1-success_rate)**max_loss):.2%}."
        ], color="danger"))
    elif max_loss >= 5:
        insights.append(dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            html.Strong("Moderate Drawdown Risk: "),
            f"Maximum loss streak: {max_loss} consecutive failures (average: {avg_loss:.1f}). "
            f"Plan for potential {max_loss}+ consecutive losses in risk management."
        ], color="warning"))
    
    # Recovery rate with actionable advice
    if probs['P_win_given_loss'] < 0.4:
        insights.append(dbc.Alert([
            html.I(className="bi bi-exclamation-octagon me-2"),
            html.Strong("Poor Recovery Rate: "),
            f"Only {probs['P_win_given_loss']:.1%} success rate after failures ({counts['loss_to_win']} wins out of {counts['loss_to_win'] + counts['loss_to_loss']} attempts). ",
            html.Br(),
            html.Strong("Action Required: "),
            "System struggles to recover from losses. Consider: (1) Taking a break after losses, "
            "(2) Reviewing entry criteria after losses, (3) Reducing position size after losses."
        ], color="warning"))
    elif probs['P_win_given_loss'] > 0.6:
        insights.append(dbc.Alert([
            html.I(className="bi bi-check-circle me-2"),
            html.Strong("Excellent Recovery Rate: "),
            f"Strong {probs['P_win_given_loss']:.1%} success rate after failures ({counts['loss_to_win']} wins out of {counts['loss_to_win'] + counts['loss_to_loss']} attempts). ",
            html.Br(),
            html.Strong("Opportunity: "),
            "System recovers well. Consider increasing position size slightly after 1-2 losses (with proper risk management)."
        ], color="success"))
    
    # Overall success rate context
    if success_rate < 0.4:
        insights.append(dbc.Alert([
            html.I(className="bi bi-exclamation-circle me-2"),
            html.Strong("Low Overall Success Rate: "),
            f"Overall success rate is only {success_rate:.1%} ({int(success_rate * total_trades)} successes out of {total_trades} trades). ",
            html.Br(),
            html.Strong("Note: "),
            "Low success rate can still be profitable if average win > average loss (positive expectancy). "
            "Check R-multiple statistics to assess profitability."
        ], color="info"))
    elif success_rate > 0.6:
        insights.append(dbc.Alert([
            html.I(className="bi bi-trophy me-2"),
            html.Strong("High Overall Success Rate: "),
            f"Strong {success_rate:.1%} success rate ({int(success_rate * total_trades)} successes out of {total_trades} trades). "
            "Good foundation for profitable trading."
        ], color="success"))
    
    if not insights:
        insights.append(dbc.Alert([
            html.I(className="bi bi-check-circle me-2"),
            "Analysis completed. Review the charts above for detailed patterns."
        ], color="light"))
    
    return insights


def create_empty_info_panel():
    """Create empty info panel"""
    import dash_bootstrap_components as dbc
    from dash import html
    
    return dbc.Alert([
        html.I(className="bi bi-info-circle me-2"),
        "Load trade data and click 'Calculate Sequential Analysis' to see detailed information"
    ], color="info", className="mb-0")


def create_error_info_panel(error_msg):
    """Create error info panel"""
    import dash_bootstrap_components as dbc
    from dash import html
    
    return dbc.Alert([
        html.I(className="bi bi-exclamation-triangle me-2"),
        html.Strong("Error: "),
        error_msg
    ], color="danger", className="mb-0")


print("[OK] Sequential Analysis callbacks registered")
