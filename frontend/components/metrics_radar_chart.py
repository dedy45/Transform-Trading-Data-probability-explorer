"""
Metrics Radar Chart Component
Multi-axis radar chart for comparing normalized performance metrics
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def normalize_metric(values, metric_name, inverse=False):
    """
    Normalize metric values to 0-100 scale
    
    Args:
        values: List of metric values
        metric_name: Name of the metric
        inverse: If True, lower values are better (e.g., max_drawdown)
    
    Returns:
        List of normalized values (0-100)
    """
    if len(values) == 0:
        return []
    
    values = np.array(values)
    
    # Handle special cases
    if metric_name == 'max_drawdown':
        # For drawdown, we want to normalize so that 0% DD = 100 score
        # and larger DD = lower score
        max_dd = np.max(np.abs(values))
        if max_dd == 0:
            return [100] * len(values)
        normalized = 100 * (1 - np.abs(values) / max_dd)
        return normalized.tolist()
    
    # For other metrics, normalize to 0-100 range
    min_val = np.min(values)
    max_val = np.max(values)
    
    if max_val == min_val:
        return [50] * len(values)  # All same, return middle value
    
    if inverse:
        # Lower is better
        normalized = 100 * (1 - (values - min_val) / (max_val - min_val))
    else:
        # Higher is better
        normalized = 100 * (values - min_val) / (max_val - min_val)
    
    return normalized.tolist()


def create_metrics_radar_chart(comparison_df):
    """
    Create radar chart comparing normalized metrics across scenarios
    
    Args:
        comparison_df: DataFrame with scenario comparison data
    
    Returns:
        Plotly Figure object
    """
    if comparison_df is None or len(comparison_df) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No scenarios to display. Add scenarios to see metrics comparison.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=450
        )
        return fig
    
    # Define metrics to include in radar chart
    metrics = [
        ('win_rate', 'Win Rate', False),
        ('avg_r', 'Avg R', False),
        ('expectancy', 'Expectancy', False),
        ('profit_factor', 'Profit Factor', False),
        ('sharpe_ratio', 'Sharpe Ratio', False),
        ('max_drawdown', 'Max Drawdown', True)  # Inverse: lower is better
    ]
    
    # Filter to available metrics
    available_metrics = [(m, l, i) for m, l, i in metrics if m in comparison_df.columns]
    
    if len(available_metrics) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No metrics available for comparison.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Extract metric values for normalization
    metric_names = [m[0] for m in available_metrics]
    metric_labels = [m[1] for m in available_metrics]
    
    # Normalize each metric across all scenarios
    normalized_data = {}
    for metric_name, metric_label, inverse in available_metrics:
        values = comparison_df[metric_name].values
        normalized_values = normalize_metric(values, metric_name, inverse)
        normalized_data[metric_name] = normalized_values
    
    # Create traces for each scenario
    fig = go.Figure()
    
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
    
    for idx, row in comparison_df.iterrows():
        scenario_name = row['scenario_name']
        
        # Get normalized values for this scenario
        r_values = [normalized_data[m][idx] for m in metric_names]
        
        # Close the radar chart by repeating first value
        theta_values = metric_labels + [metric_labels[0]]
        r_values_closed = r_values + [r_values[0]]
        
        # Determine line style
        if scenario_name == 'Baseline':
            line_width = 3
            line_dash = 'solid'
        else:
            line_width = 2
            line_dash = 'solid'
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatterpolar(
            r=r_values_closed,
            theta=theta_values,
            fill='toself',
            fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}',
            name=scenario_name,
            line=dict(color=color, width=line_width, dash=line_dash),
            hovertemplate=(
                f'<b>{scenario_name}</b><br>' +
                '%{theta}: %{r:.1f}/100<br>' +
                '<extra></extra>'
            )
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='',
                tickmode='linear',
                tick0=0,
                dtick=20
            )
        ),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.05
        ),
        title='Performance Metrics Comparison (Normalized 0-100)',
        height=450,
        margin=dict(l=80, r=150, t=80, b=50)
    )
    
    return fig


def create_metrics_heatmap(comparison_df):
    """
    Create heatmap showing normalized metrics for all scenarios
    
    Args:
        comparison_df: DataFrame with scenario comparison data
    
    Returns:
        Plotly Figure object
    """
    if comparison_df is None or len(comparison_df) == 0:
        return go.Figure()
    
    # Define metrics
    metrics = [
        ('win_rate', 'Win Rate', False),
        ('avg_r', 'Avg R', False),
        ('expectancy', 'Expectancy', False),
        ('profit_factor', 'Profit Factor', False),
        ('sharpe_ratio', 'Sharpe Ratio', False),
        ('max_drawdown', 'Max Drawdown', True)
    ]
    
    # Filter to available metrics
    available_metrics = [(m, l, i) for m, l, i in metrics if m in comparison_df.columns]
    
    if len(available_metrics) == 0:
        return go.Figure()
    
    # Normalize metrics
    metric_names = [m[0] for m in available_metrics]
    metric_labels = [m[1] for m in available_metrics]
    
    normalized_matrix = []
    for metric_name, metric_label, inverse in available_metrics:
        values = comparison_df[metric_name].values
        normalized_values = normalize_metric(values, metric_name, inverse)
        normalized_matrix.append(normalized_values)
    
    # Transpose for heatmap (scenarios as columns, metrics as rows)
    normalized_matrix = np.array(normalized_matrix)
    
    scenario_names = comparison_df['scenario_name'].tolist()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=normalized_matrix,
        x=scenario_names,
        y=metric_labels,
        colorscale='RdYlGn',
        text=np.round(normalized_matrix, 1),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title='Score<br>(0-100)'),
        hovertemplate=(
            '<b>%{y}</b><br>' +
            'Scenario: %{x}<br>' +
            'Score: %{z:.1f}/100<br>' +
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title='Metrics Heatmap (Normalized Scores)',
        xaxis_title='Scenario',
        yaxis_title='Metric',
        height=400,
        margin=dict(l=150, r=50, t=80, b=100)
    )
    
    # Rotate x-axis labels if many scenarios
    if len(scenario_names) > 5:
        fig.update_xaxes(tickangle=-45)
    
    return fig


def calculate_composite_score(comparison_df, weights=None):
    """
    Calculate composite score for each scenario based on weighted metrics
    
    Args:
        comparison_df: DataFrame with scenario comparison data
        weights: Dictionary of metric weights (default: equal weights)
    
    Returns:
        Series with composite scores for each scenario
    """
    if comparison_df is None or len(comparison_df) == 0:
        return pd.Series()
    
    # Default weights
    if weights is None:
        weights = {
            'win_rate': 0.20,
            'avg_r': 0.20,
            'expectancy': 0.20,
            'profit_factor': 0.15,
            'sharpe_ratio': 0.15,
            'max_drawdown': 0.10
        }
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Calculate normalized scores
    composite_scores = []
    
    for idx, row in comparison_df.iterrows():
        score = 0
        total_weight_used = 0
        
        for metric, weight in weights.items():
            if metric in comparison_df.columns:
                # Get all values for normalization
                values = comparison_df[metric].values
                inverse = (metric == 'max_drawdown')
                normalized = normalize_metric(values, metric, inverse)
                
                score += normalized[idx] * weight
                total_weight_used += weight
        
        # Normalize to 0-100 scale
        if total_weight_used > 0:
            score = score / total_weight_used
        
        composite_scores.append(score)
    
    return pd.Series(composite_scores, index=comparison_df.index)
