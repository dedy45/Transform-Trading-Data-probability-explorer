"""
1D Distribution Chart Component
Bar chart with confidence intervals for 1D probability distributions
"""
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def create_probability_distribution_1d(prob_results_1d, feature_name, target_name):
    """
    Create 1D probability distribution bar chart with confidence intervals
    
    Parameters:
    -----------
    prob_results_1d : pd.DataFrame
        Results from compute_1d_probability with columns:
        - bin_index, bin_left, bin_right, label
        - n, successes, p_est, ci_lower, ci_upper
        - mean_R, p_hit_1R, p_hit_2R
    feature_name : str
        Name of feature
    target_name : str
        Name of target variable
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive bar chart with error bars
    
    Features:
    ---------
    - Bars colored by probability level (red to green)
    - Error bars showing confidence intervals
    - Sample size annotations
    - Mean R-multiple as secondary axis
    - Hover info with all metrics
    """
    if prob_results_1d is None or len(prob_results_1d) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available. Click 'Calculate Probabilities' to generate distribution.",
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
    
    # Prepare data
    x_labels = prob_results_1d['label'].tolist()
    probabilities = prob_results_1d['p_est'].tolist()
    ci_lower = prob_results_1d['ci_lower'].tolist()
    ci_upper = prob_results_1d['ci_upper'].tolist()
    sample_sizes = prob_results_1d['n'].tolist()
    mean_r_values = prob_results_1d['mean_R'].tolist()
    
    # Calculate error bars
    error_y_minus = [p - ci_l for p, ci_l in zip(probabilities, ci_lower)]
    error_y_plus = [ci_u - p for p, ci_u in zip(probabilities, ci_upper)]
    
    # Color bars based on probability
    colors = []
    for p in probabilities:
        if p < 0.4:
            colors.append('rgb(220, 50, 50)')  # Red
        elif p < 0.5:
            colors.append('rgb(255, 150, 100)')  # Orange
        elif p < 0.6:
            colors.append('rgb(255, 200, 100)')  # Light orange
        elif p < 0.7:
            colors.append('rgb(150, 220, 150)')  # Light green
        else:
            colors.append('rgb(50, 180, 50)')  # Green
    
    # Create hover text
    hover_text = []
    for i in range(len(prob_results_1d)):
        text = (
            f"Bin: {x_labels[i]}<br>"
            f"Probability: {probabilities[i]:.1%}<br>"
            f"CI: [{ci_lower[i]:.1%}, {ci_upper[i]:.1%}]<br>"
            f"Sample Size: {sample_sizes[i]}<br>"
            f"Mean R: {mean_r_values[i]:.2f}<br>"
            f"P(Hit 1R): {prob_results_1d.iloc[i]['p_hit_1R']:.1%}<br>"
            f"P(Hit 2R): {prob_results_1d.iloc[i]['p_hit_2R']:.1%}"
        )
        hover_text.append(text)
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add probability bars
    fig.add_trace(go.Bar(
        x=x_labels,
        y=[p * 100 for p in probabilities],  # Convert to percentage
        error_y=dict(
            type='data',
            symmetric=False,
            array=[e * 100 for e in error_y_plus],
            arrayminus=[e * 100 for e in error_y_minus],
            color='rgba(0, 0, 0, 0.3)',
            thickness=1.5,
            width=4
        ),
        marker=dict(
            color=colors,
            line=dict(color='rgba(0, 0, 0, 0.3)', width=1)
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name='Probability',
        yaxis='y'
    ))
    
    # Add mean R line on secondary axis
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=mean_r_values,
        mode='lines+markers',
        name='Mean R-Multiple',
        line=dict(color='blue', width=2, dash='dash'),
        marker=dict(size=8, symbol='diamond'),
        yaxis='y2',
        hovertemplate='Mean R: %{y:.2f}<extra></extra>'
    ))
    
    # Add sample size annotations
    annotations = []
    for i, (label, n) in enumerate(zip(x_labels, sample_sizes)):
        annotations.append(
            dict(
                x=label,
                y=probabilities[i] * 100 + 5,
                text=f"n={n}",
                showarrow=False,
                font=dict(size=9, color='gray'),
                yref='y'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"P({target_name}) Distribution by {feature_name}",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=feature_name,
            tickangle=-45
        ),
        yaxis=dict(
            title="Probability (%)",
            side='left',
            range=[0, 100],
            ticksuffix='%'
        ),
        yaxis2=dict(
            title="Mean R-Multiple",
            side='right',
            overlaying='y',
            showgrid=False
        ),
        annotations=annotations,
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig


def create_confidence_interval_chart(prob_results_1d, feature_name, target_name):
    """
    Create confidence interval visualization
    
    Parameters:
    -----------
    prob_results_1d : pd.DataFrame
        Results from compute_1d_probability
    feature_name : str
        Name of feature
    target_name : str
        Name of target variable
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Confidence interval chart
    
    Features:
    ---------
    - Point estimates with error bars
    - Color-coded by sample size reliability
    - Horizontal reference line at base rate
    """
    if prob_results_1d is None or len(prob_results_1d) == 0:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
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
    
    # Prepare data
    x_labels = prob_results_1d['label'].tolist()
    probabilities = prob_results_1d['p_est'].tolist()
    ci_lower = prob_results_1d['ci_lower'].tolist()
    ci_upper = prob_results_1d['ci_upper'].tolist()
    sample_sizes = prob_results_1d['n'].tolist()
    
    # Color by sample size reliability
    min_samples = 20
    colors = []
    for n in sample_sizes:
        if n < min_samples:
            colors.append('rgb(220, 50, 50)')  # Red - unreliable
        elif n < min_samples * 2:
            colors.append('rgb(255, 200, 100)')  # Orange - moderate
        else:
            colors.append('rgb(50, 180, 50)')  # Green - reliable
    
    # Create figure
    fig = go.Figure()
    
    # Add confidence intervals as error bars
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=[p * 100 for p in probabilities],
        error_y=dict(
            type='data',
            symmetric=False,
            array=[(ci_u - p) * 100 for p, ci_u in zip(probabilities, ci_upper)],
            arrayminus=[(p - ci_l) * 100 for p, ci_l in zip(probabilities, ci_lower)],
            color='rgba(0, 0, 0, 0.3)',
            thickness=2,
            width=6
        ),
        mode='markers',
        marker=dict(
            size=12,
            color=colors,
            line=dict(color='rgba(0, 0, 0, 0.5)', width=1)
        ),
        name='Probability with CI',
        hovertemplate=(
            'Bin: %{x}<br>'
            'Probability: %{y:.1f}%<br>'
            '<extra></extra>'
        )
    ))
    
    # Add base rate reference line
    base_rate = prob_results_1d['p_est'].mean()
    fig.add_hline(
        y=base_rate * 100,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Base Rate: {base_rate:.1%}",
        annotation_position="right"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Confidence Intervals for P({target_name})",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=feature_name,
            tickangle=-45
        ),
        yaxis=dict(
            title="Probability (%)",
            range=[0, 100],
            ticksuffix='%'
        ),
        height=400,
        showlegend=False
    )
    
    return fig


def create_empty_distribution():
    """Create empty distribution placeholder"""
    fig = go.Figure()
    fig.add_annotation(
        text="Select a feature and click 'Calculate Probabilities'",
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
