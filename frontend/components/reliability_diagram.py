"""
Reliability Diagram Component for Calibration Lab

This component creates a reliability diagram visualization for the Calibration Lab page.
"""

import plotly.graph_objects as go
import numpy as np


def create_reliability_diagram(reliability_result):
    """
    Create reliability diagram from reliability result dictionary.
    
    Parameters
    ----------
    reliability_result : dict
        Dictionary containing:
        - 'mean_predicted': array of mean predicted probabilities per bin
        - 'observed_frequency': array of observed frequencies per bin
        - 'counts': array of sample counts per bin
    
    Returns
    -------
    plotly.graph_objects.Figure
        Reliability diagram figure
    """
    mean_predicted = reliability_result['mean_predicted']
    observed_frequency = reliability_result['observed_frequency']
    counts = reliability_result['counts']
    
    # Filter out NaN values
    mask = ~np.isnan(mean_predicted)
    mean_predicted = mean_predicted[mask]
    observed_frequency = observed_frequency[mask]
    counts = counts[mask]
    
    fig = go.Figure()
    
    # Add perfect calibration line (y=x)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash', width=2),
        showlegend=True
    ))
    
    # Add actual calibration curve
    fig.add_trace(go.Scatter(
        x=mean_predicted,
        y=observed_frequency,
        mode='markers+lines',
        name='Actual Calibration',
        marker=dict(
            size=10,
            color=counts,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Count'),
            line=dict(color='white', width=1)
        ),
        line=dict(color='blue', width=2),
        showlegend=True,
        hovertemplate=(
            '<b>Bin</b><br>'
            'Mean Predicted: %{x:.3f}<br>'
            'Observed Freq: %{y:.3f}<br>'
            'Count: %{marker.color}<br>'
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title='Reliability Diagram',
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Observed Frequency',
        xaxis=dict(range=[0, 1], constrain='domain'),
        yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1),
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig


def create_empty_reliability_diagram():
    """
    Create empty reliability diagram (no data state).
    
    Returns
    -------
    plotly.graph_objects.Figure
        Empty reliability diagram figure
    """
    fig = go.Figure()
    
    # Add perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash', width=2),
        showlegend=True
    ))
    
    # Add placeholder text
    fig.add_annotation(
        text="No calibration data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    
    fig.update_layout(
        title='Reliability Diagram',
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Observed Frequency',
        xaxis=dict(range=[0, 1], constrain='domain'),
        yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1),
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        template='plotly_white'
    )
    
    return fig
