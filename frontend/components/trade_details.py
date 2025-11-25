"""
Trade Details Panel Component
Display detailed information about selected trades
"""
import dash_bootstrap_components as dbc
from dash import html, dash_table
import pandas as pd
import plotly.graph_objects as go


def create_trade_details_summary(selected_trades_df, selection_description):
    """
    Create summary of selected trades
    
    Parameters:
    -----------
    selected_trades_df : pd.DataFrame
        Filtered trades dataframe
    selection_description : str
        Description of the selection (e.g., "trend_strength=[0.4-0.5], vol_regime=1")
    
    Returns:
    --------
    html.Div
        Trade details summary component
    """
    if selected_trades_df is None or len(selected_trades_df) == 0:
        return html.Div([
            dbc.Alert(
                "No trades in selected cell/bin",
                color="warning",
                className="mb-0"
            )
        ])
    
    # Calculate metrics
    n_trades = len(selected_trades_df)
    win_rate = selected_trades_df['trade_success'].mean()
    avg_r = selected_trades_df['R_multiple'].mean()
    expectancy = selected_trades_df['net_profit'].mean()
    total_profit = selected_trades_df['net_profit'].sum()
    
    # Winners and losers
    winners = selected_trades_df[selected_trades_df['trade_success'] == 1]
    losers = selected_trades_df[selected_trades_df['trade_success'] == 0]
    
    avg_win = winners['R_multiple'].mean() if len(winners) > 0 else 0
    avg_loss = losers['R_multiple'].mean() if len(losers) > 0 else 0
    
    # MAE/MFE
    avg_mae = selected_trades_df['MAE_R'].mean()
    avg_mfe = selected_trades_df['MFE_R'].mean()
    
    return html.Div([
        # Selection description
        dbc.Alert([
            html.H6("Selected Cell/Bin", className="alert-heading mb-2"),
            html.P(selection_description, className="mb-0 font-monospace")
        ], color="primary", className="mb-3"),
        
        # Metrics cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Trades", className="text-muted mb-1"),
                        html.H4(f"{n_trades}", className="mb-0")
                    ])
                ], className="text-center")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Win Rate", className="text-muted mb-1"),
                        html.H4(
                            f"{win_rate:.1%}",
                            className=f"mb-0 text-{'success' if win_rate >= 0.5 else 'danger'}"
                        )
                    ])
                ], className="text-center")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Avg R", className="text-muted mb-1"),
                        html.H4(
                            f"{avg_r:.2f}",
                            className=f"mb-0 text-{'success' if avg_r > 0 else 'danger'}"
                        )
                    ])
                ], className="text-center")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Expectancy", className="text-muted mb-1"),
                        html.H4(
                            f"${expectancy:.0f}",
                            className=f"mb-0 text-{'success' if expectancy > 0 else 'danger'}"
                        )
                    ])
                ], className="text-center")
            ], md=3)
        ], className="mb-3"),
        
        # Detailed statistics
        dbc.Row([
            dbc.Col([
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Metric"),
                            html.Th("Value", className="text-end")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([html.Td("Total Profit"), html.Td(f"${total_profit:.2f}", className="text-end")]),
                        html.Tr([html.Td("Winners"), html.Td(f"{len(winners)} ({len(winners)/n_trades:.1%})", className="text-end")]),
                        html.Tr([html.Td("Losers"), html.Td(f"{len(losers)} ({len(losers)/n_trades:.1%})", className="text-end")]),
                        html.Tr([html.Td("Avg Win"), html.Td(f"{avg_win:.2f}R", className="text-end text-success")]),
                        html.Tr([html.Td("Avg Loss"), html.Td(f"{avg_loss:.2f}R", className="text-end text-danger")]),
                        html.Tr([html.Td("Avg MAE"), html.Td(f"{avg_mae:.2f}R", className="text-end")]),
                        html.Tr([html.Td("Avg MFE"), html.Td(f"{avg_mfe:.2f}R", className="text-end")]),
                    ])
                ], bordered=True, hover=True, size="sm")
            ])
        ])
    ])


def create_trade_list_table(selected_trades_df, max_rows=10):
    """
    Create table of selected trades
    
    Parameters:
    -----------
    selected_trades_df : pd.DataFrame
        Filtered trades dataframe
    max_rows : int
        Maximum number of rows to display
    
    Returns:
    --------
    dash_table.DataTable
        Trade list table
    """
    if selected_trades_df is None or len(selected_trades_df) == 0:
        return html.Div([
            dbc.Alert("No trades to display", color="info", className="mb-0")
        ])
    
    # Select columns to display
    display_cols = [
        'Ticket_id', 'entry_time', 'Type', 'entry_price', 
        'ClosePrice', 'R_multiple', 'net_profit', 'holding_minutes'
    ]
    
    # Filter to available columns
    available_cols = [col for col in display_cols if col in selected_trades_df.columns]
    
    # Prepare data
    display_df = selected_trades_df[available_cols].head(max_rows).copy()
    
    # Format columns
    if 'entry_time' in display_df.columns:
        display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Create table
    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[
            {'name': col.replace('_', ' ').title(), 'id': col}
            for col in available_cols
        ],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '8px',
            'fontSize': '12px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{R_multiple} > 0'},
                'backgroundColor': 'rgba(50, 180, 50, 0.1)'
            },
            {
                'if': {'filter_query': '{R_multiple} < 0'},
                'backgroundColor': 'rgba(220, 50, 50, 0.1)'
            }
        ],
        page_size=max_rows,
        sort_action='native',
        filter_action='native'
    )
    
    return html.Div([
        html.H6(f"Trade List (showing {min(len(selected_trades_df), max_rows)} of {len(selected_trades_df)})", 
                className="mb-2"),
        table
    ])


def create_mae_mfe_mini_chart(selected_trades_df):
    """
    Create mini MAE/MFE scatter plot for selected trades
    
    Parameters:
    -----------
    selected_trades_df : pd.DataFrame
        Filtered trades dataframe
    
    Returns:
    --------
    plotly.graph_objects.Figure
        MAE/MFE scatter plot
    """
    if selected_trades_df is None or len(selected_trades_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=250
        )
        return fig
    
    # Separate winners and losers
    winners = selected_trades_df[selected_trades_df['trade_success'] == 1]
    losers = selected_trades_df[selected_trades_df['trade_success'] == 0]
    
    fig = go.Figure()
    
    # Add winners
    if len(winners) > 0:
        fig.add_trace(go.Scatter(
            x=winners['MAE_R'],
            y=winners['MFE_R'],
            mode='markers',
            name='Winners',
            marker=dict(
                color='green',
                size=8,
                opacity=0.6
            ),
            hovertemplate='MAE: %{x:.2f}R<br>MFE: %{y:.2f}R<extra></extra>'
        ))
    
    # Add losers
    if len(losers) > 0:
        fig.add_trace(go.Scatter(
            x=losers['MAE_R'],
            y=losers['MFE_R'],
            mode='markers',
            name='Losers',
            marker=dict(
                color='red',
                size=8,
                opacity=0.6
            ),
            hovertemplate='MAE: %{x:.2f}R<br>MFE: %{y:.2f}R<extra></extra>'
        ))
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title="MAE vs MFE",
        xaxis=dict(title="MAE (R)"),
        yaxis=dict(title="MFE (R)"),
        height=250,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=40, r=20, t=40, b=40)
    )
    
    return fig


def create_empty_trade_details():
    """Create empty trade details placeholder"""
    return html.Div([
        dbc.Alert(
            [
                html.I(className="bi bi-info-circle me-2"),
                "Click on a heatmap cell or distribution bar to view trade details"
            ],
            color="light",
            className="mb-0"
        )
    ])
