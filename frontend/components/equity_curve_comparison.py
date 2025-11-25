"""
Equity Curve Comparison Component
Overlaid equity curves for multiple scenarios with drawdown visualization
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_equity_curve_comparison_chart(scenarios_data, show_drawdown=True, starting_equity=10000):
    """
    Create equity curve comparison chart with multiple scenarios
    
    Args:
        scenarios_data: Dictionary mapping scenario names to trade DataFrames
        show_drawdown: Boolean to show/hide drawdown shading
        starting_equity: Starting equity amount (default: 10000)
    
    Returns:
        Plotly Figure object
    """
    if not scenarios_data or len(scenarios_data) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No scenarios to display. Add scenarios to see equity curves.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=500
        )
        return fig
    
    fig = go.Figure()
    
    # Color palette for scenarios
    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]
    
    # Process each scenario
    for idx, (scenario_name, trades_df) in enumerate(scenarios_data.items()):
        if trades_df is None or len(trades_df) == 0:
            continue
        
        # Calculate cumulative equity
        if 'net_profit' in trades_df.columns:
            cumulative_profit = trades_df['net_profit'].cumsum()
            equity_curve = starting_equity + cumulative_profit
            
            # Calculate drawdown
            running_max = equity_curve.cummax()
            drawdown = equity_curve - running_max
            drawdown_pct = (drawdown / running_max * 100)
            
            # Create trade numbers for x-axis
            trade_numbers = list(range(len(equity_curve)))
            
            # Add equity curve line
            color = colors[idx % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=trade_numbers,
                y=equity_curve,
                mode='lines',
                name=scenario_name,
                line=dict(color=color, width=2),
                hovertemplate=(
                    f'<b>{scenario_name}</b><br>' +
                    'Trade: %{x}<br>' +
                    'Equity: $%{y:,.2f}<br>' +
                    '<extra></extra>'
                )
            ))
            
            # Add drawdown shading if enabled
            if show_drawdown and scenario_name == 'Baseline':
                # Only show drawdown for baseline to avoid clutter
                fig.add_trace(go.Scatter(
                    x=trade_numbers,
                    y=running_max,
                    mode='lines',
                    name='Peak Equity',
                    line=dict(color='rgba(128, 128, 128, 0.3)', width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Fill between equity and peak
                fig.add_trace(go.Scatter(
                    x=trade_numbers + trade_numbers[::-1],
                    y=list(equity_curve) + list(running_max)[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    line=dict(color='rgba(255, 0, 0, 0)'),
                    name='Drawdown',
                    showlegend=True,
                    hoverinfo='skip'
                ))
    
    # Update layout
    fig.update_layout(
        title='Equity Curve Comparison',
        xaxis_title='Trade Number',
        yaxis_title='Equity ($)',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=500,
        template='plotly_white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickprefix='$', tickformat=',.0f')
    
    return fig


def create_drawdown_comparison_chart(scenarios_data, starting_equity=10000):
    """
    Create drawdown comparison chart showing drawdown curves for all scenarios
    
    Args:
        scenarios_data: Dictionary mapping scenario names to trade DataFrames
        starting_equity: Starting equity amount (default: 10000)
    
    Returns:
        Plotly Figure object
    """
    if not scenarios_data or len(scenarios_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No scenarios to display.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, (scenario_name, trades_df) in enumerate(scenarios_data.items()):
        if trades_df is None or len(trades_df) == 0:
            continue
        
        if 'net_profit' in trades_df.columns:
            cumulative_profit = trades_df['net_profit'].cumsum()
            equity_curve = starting_equity + cumulative_profit
            
            # Calculate drawdown percentage
            running_max = equity_curve.cummax()
            drawdown = equity_curve - running_max
            drawdown_pct = (drawdown / running_max * 100)
            
            trade_numbers = list(range(len(drawdown_pct)))
            color = colors[idx % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=trade_numbers,
                y=drawdown_pct,
                mode='lines',
                name=scenario_name,
                line=dict(color=color, width=2),
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}',
                hovertemplate=(
                    f'<b>{scenario_name}</b><br>' +
                    'Trade: %{x}<br>' +
                    'Drawdown: %{y:.2f}%<br>' +
                    '<extra></extra>'
                )
            ))
    
    fig.update_layout(
        title='Drawdown Comparison',
        xaxis_title='Trade Number',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=400,
        template='plotly_white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_yaxes(ticksuffix='%')
    
    return fig


def create_equity_growth_comparison(scenarios_data, starting_equity=10000):
    """
    Create bar chart comparing final equity growth across scenarios
    
    Args:
        scenarios_data: Dictionary mapping scenario names to trade DataFrames
        starting_equity: Starting equity amount (default: 10000)
    
    Returns:
        Plotly Figure object
    """
    if not scenarios_data or len(scenarios_data) == 0:
        return go.Figure()
    
    scenario_names = []
    equity_growth = []
    colors_list = []
    
    for scenario_name, trades_df in scenarios_data.items():
        if trades_df is None or len(trades_df) == 0:
            continue
        
        if 'net_profit' in trades_df.columns:
            total_profit = trades_df['net_profit'].sum()
            final_equity = starting_equity + total_profit
            growth_pct = ((final_equity - starting_equity) / starting_equity * 100)
            
            scenario_names.append(scenario_name)
            equity_growth.append(growth_pct)
            
            # Color based on performance
            if growth_pct > 0:
                colors_list.append('#2ca02c')  # Green
            else:
                colors_list.append('#d62728')  # Red
    
    fig = go.Figure(data=[
        go.Bar(
            x=scenario_names,
            y=equity_growth,
            marker_color=colors_list,
            text=[f'{val:.1f}%' for val in equity_growth],
            textposition='outside',
            hovertemplate=(
                '<b>%{x}</b><br>' +
                'Growth: %{y:.2f}%<br>' +
                '<extra></extra>'
            )
        )
    ])
    
    fig.update_layout(
        title='Equity Growth Comparison',
        xaxis_title='Scenario',
        yaxis_title='Equity Growth (%)',
        height=350,
        template='plotly_white',
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False
    )
    
    fig.update_yaxes(ticksuffix='%')
    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
    
    return fig
