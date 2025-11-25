"""
Calibration Histogram Component
Histogram of predicted probabilities distribution
"""
import plotly.graph_objects as go
import numpy as np


def create_calibration_histogram(predicted_probs, n_bins=20):
    """
    Create histogram of predicted probabilities distribution
    
    Parameters:
    -----------
    predicted_probs : array-like
        Array of predicted probabilities (0-1)
    n_bins : int
        Number of bins for histogram
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive histogram of predicted probabilities
    
    Features:
    ---------
    - Histogram showing distribution of predictions
    - Statistics overlay (mean, median, std)
    - Color coding by probability range
    - Hover info with bin details
    """
    if predicted_probs is None or len(predicted_probs) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No probability predictions available. Load data to view distribution.",
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
    
    predicted_probs = np.asarray(predicted_probs)
    
    # Calculate statistics
    mean_prob = np.mean(predicted_probs)
    median_prob = np.median(predicted_probs)
    std_prob = np.std(predicted_probs)
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=predicted_probs,
        nbinsx=n_bins,
        name='Predicted Probabilities',
        marker=dict(
            color='rgba(70, 130, 180, 0.7)',
            line=dict(color='white', width=1)
        ),
        hovertemplate=(
            'Probability Range: %{x}<br>'
            'Count: %{y}<br>'
            '<extra></extra>'
        )
    ))
    
    # Add mean line
    fig.add_vline(
        x=mean_prob,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_prob:.1%}",
        annotation_position="top"
    )
    
    # Add median line
    fig.add_vline(
        x=median_prob,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Median: {median_prob:.1%}",
        annotation_position="bottom"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Distribution of Predicted Probabilities",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Predicted Probability",
            range=[0, 1],
            tickformat='.0%',
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            title="Frequency (Count)",
            gridcolor='lightgray',
            showgrid=True
        ),
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        annotations=[
            dict(
                text=(
                    f"<b>Statistics</b><br>"
                    f"Mean: {mean_prob:.1%}<br>"
                    f"Median: {median_prob:.1%}<br>"
                    f"Std Dev: {std_prob:.1%}<br>"
                    f"N: {len(predicted_probs)}"
                ),
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                xanchor='right', yanchor='top',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='gray',
                borderwidth=1,
                borderpad=10,
                font=dict(size=11)
            )
        ]
    )
    
    return fig


def create_empty_calibration_histogram():
    """Create empty calibration histogram placeholder"""
    fig = go.Figure()
    fig.add_annotation(
        text="Load probability predictions to view distribution",
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
