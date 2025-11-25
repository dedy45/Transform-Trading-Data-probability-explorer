"""
Trade Distribution Comparison Component
Histogram, box plot, and violin plot comparisons of trade distributions
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_distribution_comparison_chart(scenarios_data, distribution_type='histogram', metric='R_multiple'):
    """
    Create distribution comparison chart for specified metric
    
    Args:
        scenarios_data: Dictionary mapping scenario names to trade DataFrames
        distribution_type: Type of chart ('histogram', 'box', 'violin')
        metric: Metric to compare ('R_multiple', 'net_profit', 'holding_minutes')
    
    Returns:
        Plotly Figure object
    """
    if not scenarios_data or len(scenarios_data) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No scenarios to display. Add scenarios to see distributions.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    if distribution_type == 'histogram':
        return create_histogram_comparison(scenarios_data, metric)
    elif distribution_type == 'box':
        return create_box_plot_comparison(scenarios_data, metric)
    elif distribution_type == 'violin':
        return create_violin_plot_comparison(scenarios_data, metric)
    else:
        return go.Figure()


def create_histogram_comparison(scenarios_data, metric='R_multiple'):
    """
    Create overlaid histogram comparison
    
    Args:
        scenarios_data: Dictionary mapping scenario names to trade DataFrames
        metric: Metric to compare
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Determine metric label and format
    metric_labels = {
        'R_multiple': 'R-Multiple',
        'net_profit': 'Net Profit ($)',
        'holding_minutes': 'Holding Time (minutes)',
        'MAE_R': 'MAE (R)',
        'MFE_R': 'MFE (R)'
    }
    
    metric_label = metric_labels.get(metric, metric)
    
    for idx, (scenario_name, trades_df) in enumerate(scenarios_data.items()):
        if trades_df is None or len(trades_df) == 0 or metric not in trades_df.columns:
            continue
        
        values = trades_df[metric].dropna()
        
        if len(values) == 0:
            continue
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Histogram(
            x=values,
            name=scenario_name,
            opacity=0.6,
            marker_color=color,
            nbinsx=30,
            hovertemplate=(
                f'<b>{scenario_name}</b><br>' +
                f'{metric_label}: %{{x}}<br>' +
                'Count: %{y}<br>' +
                '<extra></extra>'
            )
        ))
    
    fig.update_layout(
        title=f'{metric_label} Distribution Comparison',
        xaxis_title=metric_label,
        yaxis_title='Frequency',
        barmode='overlay',
        height=400,
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add vertical line at 0 for R-multiple
    if metric == 'R_multiple':
        fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.5)
        fig.add_vline(x=1, line_dash='dash', line_color='green', opacity=0.3)
    
    return fig


def create_box_plot_comparison(scenarios_data, metric='R_multiple'):
    """
    Create box plot comparison
    
    Args:
        scenarios_data: Dictionary mapping scenario names to trade DataFrames
        metric: Metric to compare
    
    Returns:
        Plotly Figure object
    """
    # Prepare data for box plot
    all_data = []
    
    for scenario_name, trades_df in scenarios_data.items():
        if trades_df is None or len(trades_df) == 0 or metric not in trades_df.columns:
            continue
        
        values = trades_df[metric].dropna()
        
        for value in values:
            all_data.append({
                'Scenario': scenario_name,
                'Value': value
            })
    
    if len(all_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for box plot.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    df = pd.DataFrame(all_data)
    
    # Determine metric label
    metric_labels = {
        'R_multiple': 'R-Multiple',
        'net_profit': 'Net Profit ($)',
        'holding_minutes': 'Holding Time (minutes)',
        'MAE_R': 'MAE (R)',
        'MFE_R': 'MFE (R)'
    }
    
    metric_label = metric_labels.get(metric, metric)
    
    # Create box plot
    fig = px.box(
        df,
        x='Scenario',
        y='Value',
        color='Scenario',
        title=f'{metric_label} Distribution Comparison (Box Plot)',
        labels={'Value': metric_label}
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add horizontal line at 0 for R-multiple
    if metric == 'R_multiple':
        fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
        fig.add_hline(y=1, line_dash='dash', line_color='green', opacity=0.3)
    
    return fig


def create_violin_plot_comparison(scenarios_data, metric='R_multiple'):
    """
    Create violin plot comparison
    
    Args:
        scenarios_data: Dictionary mapping scenario names to trade DataFrames
        metric: Metric to compare
    
    Returns:
        Plotly Figure object
    """
    # Prepare data for violin plot
    all_data = []
    
    for scenario_name, trades_df in scenarios_data.items():
        if trades_df is None or len(trades_df) == 0 or metric not in trades_df.columns:
            continue
        
        values = trades_df[metric].dropna()
        
        for value in values:
            all_data.append({
                'Scenario': scenario_name,
                'Value': value
            })
    
    if len(all_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for violin plot.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    df = pd.DataFrame(all_data)
    
    # Determine metric label
    metric_labels = {
        'R_multiple': 'R-Multiple',
        'net_profit': 'Net Profit ($)',
        'holding_minutes': 'Holding Time (minutes)',
        'MAE_R': 'MAE (R)',
        'MFE_R': 'MFE (R)'
    }
    
    metric_label = metric_labels.get(metric, metric)
    
    # Create violin plot
    fig = px.violin(
        df,
        x='Scenario',
        y='Value',
        color='Scenario',
        box=True,
        points='outliers',
        title=f'{metric_label} Distribution Comparison (Violin Plot)',
        labels={'Value': metric_label}
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add horizontal line at 0 for R-multiple
    if metric == 'R_multiple':
        fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
        fig.add_hline(y=1, line_dash='dash', line_color='green', opacity=0.3)
    
    return fig


def create_win_loss_distribution_comparison(scenarios_data):
    """
    Create stacked bar chart comparing win/loss distribution
    
    Args:
        scenarios_data: Dictionary mapping scenario names to trade DataFrames
    
    Returns:
        Plotly Figure object
    """
    if not scenarios_data or len(scenarios_data) == 0:
        return go.Figure()
    
    scenario_names = []
    win_counts = []
    loss_counts = []
    
    for scenario_name, trades_df in scenarios_data.items():
        if trades_df is None or len(trades_df) == 0:
            continue
        
        if 'trade_success' in trades_df.columns:
            wins = (trades_df['trade_success'] == 1).sum()
            losses = (trades_df['trade_success'] == 0).sum()
            
            scenario_names.append(scenario_name)
            win_counts.append(wins)
            loss_counts.append(losses)
    
    if len(scenario_names) == 0:
        return go.Figure()
    
    fig = go.Figure(data=[
        go.Bar(
            name='Wins',
            x=scenario_names,
            y=win_counts,
            marker_color='#2ca02c',
            text=win_counts,
            textposition='inside',
            hovertemplate='<b>Wins</b><br>Count: %{y}<extra></extra>'
        ),
        go.Bar(
            name='Losses',
            x=scenario_names,
            y=loss_counts,
            marker_color='#d62728',
            text=loss_counts,
            textposition='inside',
            hovertemplate='<b>Losses</b><br>Count: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Win/Loss Distribution Comparison',
        xaxis_title='Scenario',
        yaxis_title='Number of Trades',
        barmode='stack',
        height=350,
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def create_r_multiple_percentile_comparison(scenarios_data):
    """
    Create bar chart comparing R-multiple percentiles across scenarios
    
    Args:
        scenarios_data: Dictionary mapping scenario names to trade DataFrames
    
    Returns:
        Plotly Figure object
    """
    if not scenarios_data or len(scenarios_data) == 0:
        return go.Figure()
    
    scenario_names = []
    p25_values = []
    p50_values = []
    p75_values = []
    
    for scenario_name, trades_df in scenarios_data.items():
        if trades_df is None or len(trades_df) == 0:
            continue
        
        if 'R_multiple' in trades_df.columns:
            r_values = trades_df['R_multiple'].dropna()
            
            if len(r_values) > 0:
                scenario_names.append(scenario_name)
                p25_values.append(r_values.quantile(0.25))
                p50_values.append(r_values.quantile(0.50))
                p75_values.append(r_values.quantile(0.75))
    
    if len(scenario_names) == 0:
        return go.Figure()
    
    fig = go.Figure(data=[
        go.Bar(
            name='25th Percentile',
            x=scenario_names,
            y=p25_values,
            marker_color='#ff7f0e',
            hovertemplate='<b>P25</b><br>R: %{y:.2f}<extra></extra>'
        ),
        go.Bar(
            name='Median (50th)',
            x=scenario_names,
            y=p50_values,
            marker_color='#1f77b4',
            hovertemplate='<b>Median</b><br>R: %{y:.2f}<extra></extra>'
        ),
        go.Bar(
            name='75th Percentile',
            x=scenario_names,
            y=p75_values,
            marker_color='#2ca02c',
            hovertemplate='<b>P75</b><br>R: %{y:.2f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='R-Multiple Percentiles Comparison',
        xaxis_title='Scenario',
        yaxis_title='R-Multiple',
        barmode='group',
        height=350,
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
    fig.add_hline(y=1, line_dash='dash', line_color='green', opacity=0.3)
    
    return fig
