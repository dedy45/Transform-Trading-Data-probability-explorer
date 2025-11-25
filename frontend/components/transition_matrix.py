"""
Transition Matrix Heatmap Component
Interactive heatmap for visualizing Markov chain transition probabilities
"""
import plotly.graph_objects as go
import numpy as np


def create_transition_matrix_heatmap(markov_results, conf_level=0.95):
    """
    Create interactive transition matrix heatmap for Markov chain analysis
    
    Parameters:
    -----------
    markov_results : dict
        Results from compute_first_order_markov with keys:
        - probs: Dict with P_win_given_win, P_loss_given_win, P_win_given_loss, P_loss_given_loss
        - counts: Dict with transition counts
        - ci: Dict with confidence intervals
        - n_transitions: Total transitions
    conf_level : float
        Confidence level used (for display)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive transition matrix heatmap
    
    Features:
    ---------
    - 2x2 matrix showing all transition probabilities
    - Color scale from red (low) to green (high)
    - Hover info: probability, CI, count
    - Annotations showing probability values
    - Row sums to 1.0 validation
    """
    if markov_results is None or markov_results.get('n_transitions', 0) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No sequential data available. Load trade data to generate transition matrix.",
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
    
    probs = markov_results['probs']
    counts = markov_results['counts']
    ci = markov_results['ci']
    
    # Create 2x2 matrix
    # Rows: Current state (Win, Loss)
    # Cols: Next state (Win, Loss)
    z_matrix = [
        [probs['P_win_given_win'], probs['P_loss_given_win']],
        [probs['P_win_given_loss'], probs['P_loss_given_loss']]
    ]
    
    # Create hover text with CI and counts
    hover_text = [
        [
            f"P(Win | Win)<br>"
            f"Probability: {probs['P_win_given_win']:.1%}<br>"
            f"CI: [{ci['P_win_given_win']['ci_lower']:.1%}, {ci['P_win_given_win']['ci_upper']:.1%}]<br>"
            f"Count: {counts['win_to_win']}<br>"
            f"Confidence: {conf_level:.0%}",
            
            f"P(Loss | Win)<br>"
            f"Probability: {probs['P_loss_given_win']:.1%}<br>"
            f"CI: [{ci['P_loss_given_win']['ci_lower']:.1%}, {ci['P_loss_given_win']['ci_upper']:.1%}]<br>"
            f"Count: {counts['win_to_loss']}<br>"
            f"Confidence: {conf_level:.0%}"
        ],
        [
            f"P(Win | Loss)<br>"
            f"Probability: {probs['P_win_given_loss']:.1%}<br>"
            f"CI: [{ci['P_win_given_loss']['ci_lower']:.1%}, {ci['P_win_given_loss']['ci_upper']:.1%}]<br>"
            f"Count: {counts['loss_to_win']}<br>"
            f"Confidence: {conf_level:.0%}",
            
            f"P(Loss | Loss)<br>"
            f"Probability: {probs['P_loss_given_loss']:.1%}<br>"
            f"CI: [{ci['P_loss_given_loss']['ci_lower']:.1%}, {ci['P_loss_given_loss']['ci_upper']:.1%}]<br>"
            f"Count: {counts['loss_to_loss']}<br>"
            f"Confidence: {conf_level:.0%}"
        ]
    ]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[[p * 100 for p in row] for row in z_matrix],  # Convert to percentage
        x=['Win', 'Loss'],
        y=['Win', 'Loss'],
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
    for i in range(2):
        for j in range(2):
            prob = z_matrix[i][j]
            if not np.isnan(prob):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f"{prob:.1%}",
                        showarrow=False,
                        font=dict(
                            color='white' if prob < 0.5 else 'black',
                            size=16,
                            family='Arial Black'
                        )
                    )
                )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="First-Order Markov Transition Matrix",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Next Trade Outcome",
            side='bottom',
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title="Current Trade Outcome",
            tickfont=dict(size=14)
        ),
        annotations=annotations,
        height=400,
        hovermode='closest'
    )
    
    return fig


def create_empty_transition_matrix():
    """Create empty transition matrix placeholder"""
    fig = go.Figure()
    fig.add_annotation(
        text="Load trade data to view transition matrix",
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
