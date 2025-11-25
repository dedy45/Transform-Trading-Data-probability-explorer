"""
Streak Distribution Component
Visualize win and loss streak distributions
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_streak_distribution_chart(streak_results):
    """
    Create streak distribution visualization with separate charts for wins and losses
    
    Parameters:
    -----------
    streak_results : dict
        Results from compute_streak_distribution with keys:
        - win_streaks: List of win streak lengths
        - loss_streaks: List of loss streak lengths
        - win_streak_distribution: Dict mapping streak length to count
        - loss_streak_distribution: Dict mapping streak length to count
        - max_win_streak: Maximum consecutive wins
        - max_loss_streak: Maximum consecutive losses
        - avg_win_streak: Average win streak length
        - avg_loss_streak: Average loss streak length
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Dual bar chart showing win and loss streak distributions
    
    Features:
    ---------
    - Side-by-side bar charts for wins and losses
    - Color-coded (green for wins, red for losses)
    - Annotations for max and average streaks
    - Hover info with count and percentage
    """
    if streak_results is None or len(streak_results.get('win_streaks', [])) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No streak data available. Load trade data to analyze streaks.",
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
    
    win_dist = streak_results['win_streak_distribution']
    loss_dist = streak_results['loss_streak_distribution']
    max_win = streak_results['max_win_streak']
    max_loss = streak_results['max_loss_streak']
    avg_win = streak_results['avg_win_streak']
    avg_loss = streak_results['avg_loss_streak']
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Win Streaks', 'Loss Streaks'),
        horizontal_spacing=0.12
    )
    
    # Win streaks bar chart
    if win_dist:
        win_lengths = sorted(win_dist.keys())
        win_counts = [win_dist[length] for length in win_lengths]
        total_win_streaks = sum(win_counts)
        win_percentages = [count / total_win_streaks * 100 for count in win_counts]
        
        win_hover_text = [
            f"Streak Length: {length}<br>"
            f"Count: {count}<br>"
            f"Percentage: {pct:.1f}%"
            for length, count, pct in zip(win_lengths, win_counts, win_percentages)
        ]
        
        fig.add_trace(
            go.Bar(
                x=win_lengths,
                y=win_counts,
                name='Win Streaks',
                marker=dict(
                    color='rgb(50, 180, 50)',
                    line=dict(color='rgb(30, 130, 30)', width=1)
                ),
                text=win_hover_text,
                hovertemplate='%{text}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Loss streaks bar chart
    if loss_dist:
        loss_lengths = sorted(loss_dist.keys())
        loss_counts = [loss_dist[length] for length in loss_lengths]
        total_loss_streaks = sum(loss_counts)
        loss_percentages = [count / total_loss_streaks * 100 for count in loss_counts]
        
        loss_hover_text = [
            f"Streak Length: {length}<br>"
            f"Count: {count}<br>"
            f"Percentage: {pct:.1f}%"
            for length, count, pct in zip(loss_lengths, loss_counts, loss_percentages)
        ]
        
        fig.add_trace(
            go.Bar(
                x=loss_lengths,
                y=loss_counts,
                name='Loss Streaks',
                marker=dict(
                    color='rgb(220, 50, 50)',
                    line=dict(color='rgb(180, 30, 30)', width=1)
                ),
                text=loss_hover_text,
                hovertemplate='%{text}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # Update axes
    fig.update_xaxes(title_text="Streak Length", row=1, col=1)
    fig.update_xaxes(title_text="Streak Length", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    # Add annotations for max and average
    annotations = [
        dict(
            x=0.25, y=1.15,
            xref='paper', yref='paper',
            text=f"Max: {max_win} | Avg: {avg_win:.1f}",
            showarrow=False,
            font=dict(size=12, color='rgb(50, 180, 50)'),
            xanchor='center'
        ),
        dict(
            x=0.75, y=1.15,
            xref='paper', yref='paper',
            text=f"Max: {max_loss} | Avg: {avg_loss:.1f}",
            showarrow=False,
            font=dict(size=12, color='rgb(220, 50, 50)'),
            xanchor='center'
        )
    ]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Consecutive Win/Loss Streak Distribution",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        showlegend=False,
        height=400,
        annotations=annotations,
        hovermode='closest'
    )
    
    return fig


def create_empty_streak_distribution():
    """Create empty streak distribution placeholder"""
    fig = go.Figure()
    fig.add_annotation(
        text="Load trade data to view streak distribution",
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
