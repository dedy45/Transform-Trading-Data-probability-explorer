"""
2D Probability Heatmap Component
Interactive heatmap for visualizing 2D probability distributions
"""
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def create_probability_heatmap_2d(prob_results_2d, feature_x_name, feature_y_name, target_name):
    """
    Create interactive 2D probability heatmap
    
    Parameters:
    -----------
    prob_results_2d : pd.DataFrame
        Results from compute_2d_probability with columns:
        - bin_x, bin_y, x_left, x_right, y_left, y_right
        - n, successes, p_est, ci_lower, ci_upper
        - mean_R, p_hit_1R, p_hit_2R
    feature_x_name : str
        Name of feature on X-axis
    feature_y_name : str
        Name of feature on Y-axis
    target_name : str
        Name of target variable
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive heatmap figure
    
    Features:
    ---------
    - Color scale from red (low probability) to green (high probability)
    - Hover info: probability, CI, sample size, mean R
    - Click to select cell
    - Opacity based on sample size (low samples = more transparent)
    - Annotations showing probability values
    """
    if prob_results_2d is None or len(prob_results_2d) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available. Click 'Calculate Probabilities' to generate heatmap.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=600
        )
        return fig
    
    # Pivot data for heatmap
    pivot_prob = prob_results_2d.pivot_table(
        index='bin_y',
        columns='bin_x',
        values='p_est',
        fill_value=np.nan
    )
    
    pivot_n = prob_results_2d.pivot_table(
        index='bin_y',
        columns='bin_x',
        values='n',
        fill_value=0
    )
    
    pivot_ci_lower = prob_results_2d.pivot_table(
        index='bin_y',
        columns='bin_x',
        values='ci_lower',
        fill_value=np.nan
    )
    
    pivot_ci_upper = prob_results_2d.pivot_table(
        index='bin_y',
        columns='bin_x',
        values='ci_upper',
        fill_value=np.nan
    )
    
    pivot_mean_r = prob_results_2d.pivot_table(
        index='bin_y',
        columns='bin_x',
        values='mean_R',
        fill_value=np.nan
    )
    
    # Create hover text
    hover_text = []
    for i in range(len(pivot_prob.index)):
        hover_row = []
        for j in range(len(pivot_prob.columns)):
            prob = pivot_prob.iloc[i, j]
            n = pivot_n.iloc[i, j]
            ci_l = pivot_ci_lower.iloc[i, j]
            ci_u = pivot_ci_upper.iloc[i, j]
            mean_r = pivot_mean_r.iloc[i, j]
            
            if pd.notna(prob):
                text = (
                    f"Probability: {prob:.1%}<br>"
                    f"CI: [{ci_l:.1%}, {ci_u:.1%}]<br>"
                    f"Sample Size: {int(n)}<br>"
                    f"Mean R: {mean_r:.2f}<br>"
                    f"Bin X: {j}<br>"
                    f"Bin Y: {i}"
                )
            else:
                text = "Insufficient data"
            hover_row.append(text)
        hover_text.append(hover_row)
    
    # Calculate opacity based on sample size
    # More samples = more opaque
    min_samples = 20
    max_samples = pivot_n.max().max()
    opacity_matrix = np.clip(pivot_n / max_samples, 0.3, 1.0)
    opacity_matrix = np.where(pivot_n < min_samples, 0.2, opacity_matrix)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_prob.values * 100,  # Convert to percentage
        x=pivot_prob.columns,
        y=pivot_prob.index,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorscale=[
            [0, 'rgb(220, 50, 50)'],      # Red for low probability
            [0.3, 'rgb(255, 200, 100)'],  # Orange
            [0.5, 'rgb(255, 255, 150)'],  # Yellow
            [0.7, 'rgb(150, 220, 150)'],  # Light green
            [1, 'rgb(50, 180, 50)']       # Green for high probability
        ],
        colorbar=dict(
            title="Probability (%)",
            ticksuffix="%"
        ),
        zmin=0,
        zmax=100
    ))
    
    # Add annotations for probability values
    annotations = []
    for i in range(len(pivot_prob.index)):
        for j in range(len(pivot_prob.columns)):
            prob = pivot_prob.iloc[i, j]
            n = pivot_n.iloc[i, j]
            if pd.notna(prob) and n >= min_samples:
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f"{prob:.0%}",
                        showarrow=False,
                        font=dict(
                            color='white' if prob < 0.5 else 'black',
                            size=10
                        )
                    )
                )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"P({target_name}) by {feature_x_name} and {feature_y_name}",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=feature_x_name,
            side='bottom'
        ),
        yaxis=dict(
            title=feature_y_name,
            autorange='reversed'  # Top to bottom
        ),
        annotations=annotations,
        height=600,
        hovermode='closest'
    )
    
    return fig


def create_empty_heatmap():
    """Create empty heatmap placeholder"""
    fig = go.Figure()
    fig.add_annotation(
        text="Select features and click 'Calculate Probabilities'",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=600
    )
    return fig
