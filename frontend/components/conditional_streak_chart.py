"""
Conditional Streak Chart Component
Visualize P(Win | loss_streak = k) for various loss streak lengths
"""
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def create_conditional_streak_chart(conditional_results):
    """
    Create P(Win | loss_streak = k) visualization
    
    Parameters:
    -----------
    conditional_results : pd.DataFrame
        Results from compute_winrate_given_loss_streak with columns:
        - loss_streak_length: Length of preceding loss streak
        - n_opportunities: Number of times this streak occurred
        - n_wins: Number of wins after this streak
        - win_rate: P(Win | loss_streak = k)
        - ci_lower: Lower bound of CI
        - ci_upper: Upper bound of CI
        - reliable: Boolean indicating sufficient sample size
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Line chart with error bars showing conditional win rates
    
    Features:
    ---------
    - Line chart with confidence interval bands
    - Color-coded by reliability (sufficient samples)
    - Horizontal line showing baseline win rate
    - Hover info with win rate, CI, and sample size
    - Annotations for significant deviations
    """
    if conditional_results is None or len(conditional_results) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No conditional probability data available. Load trade data to analyze.",
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
    
    # Filter to only show reliable data points
    df = conditional_results.copy()
    
    # Calculate baseline win rate (k=0, no loss streak)
    baseline_row = df[df['loss_streak_length'] == 0]
    if len(baseline_row) > 0 and baseline_row.iloc[0]['reliable']:
        baseline_win_rate = baseline_row.iloc[0]['win_rate']
    else:
        # Use overall average if k=0 not reliable
        reliable_df = df[df['reliable']]
        if len(reliable_df) > 0:
            baseline_win_rate = reliable_df['win_rate'].mean()
        else:
            baseline_win_rate = 0.5
    
    # Separate reliable and unreliable data
    reliable_df = df[df['reliable']].copy()
    unreliable_df = df[~df['reliable']].copy()
    
    fig = go.Figure()
    
    # Add confidence interval band for reliable data
    if len(reliable_df) > 0:
        # Upper bound
        fig.add_trace(go.Scatter(
            x=reliable_df['loss_streak_length'],
            y=reliable_df['ci_upper'] * 100,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Lower bound (fill to upper)
        fig.add_trace(go.Scatter(
            x=reliable_df['loss_streak_length'],
            y=reliable_df['ci_lower'] * 100,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(100, 150, 200, 0.2)',
            name='95% CI',
            hoverinfo='skip'
        ))
        
        # Main line for reliable data
        hover_text_reliable = [
            f"Loss Streak: {int(row['loss_streak_length'])}<br>"
            f"Win Rate: {row['win_rate']:.1%}<br>"
            f"CI: [{row['ci_lower']:.1%}, {row['ci_upper']:.1%}]<br>"
            f"Sample Size: {int(row['n_opportunities'])}<br>"
            f"Wins: {int(row['n_wins'])}"
            for _, row in reliable_df.iterrows()
        ]
        
        fig.add_trace(go.Scatter(
            x=reliable_df['loss_streak_length'],
            y=reliable_df['win_rate'] * 100,
            mode='lines+markers',
            name='Win Rate (Reliable)',
            line=dict(color='rgb(50, 120, 200)', width=3),
            marker=dict(size=8, color='rgb(50, 120, 200)'),
            text=hover_text_reliable,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Add unreliable data points (dashed line, smaller markers)
    if len(unreliable_df) > 0:
        hover_text_unreliable = [
            f"Loss Streak: {int(row['loss_streak_length'])}<br>"
            f"Win Rate: {row['win_rate']:.1%}<br>"
            f"Sample Size: {int(row['n_opportunities'])} (Low)<br>"
            f"⚠️ Insufficient data"
            for _, row in unreliable_df.iterrows()
        ]
        
        fig.add_trace(go.Scatter(
            x=unreliable_df['loss_streak_length'],
            y=unreliable_df['win_rate'] * 100,
            mode='lines+markers',
            name='Win Rate (Unreliable)',
            line=dict(color='rgb(150, 150, 150)', width=2, dash='dash'),
            marker=dict(size=6, color='rgb(150, 150, 150)', symbol='x'),
            text=hover_text_unreliable,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Add baseline win rate horizontal line
    fig.add_hline(
        y=baseline_win_rate * 100,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Baseline: {baseline_win_rate:.1%}",
        annotation_position="right"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="P(Win | Loss Streak = k)",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Consecutive Loss Streak Length",
            tickmode='linear',
            tick0=0,
            dtick=1
        ),
        yaxis=dict(
            title="Win Rate (%)",
            ticksuffix="%",
            range=[0, 100]
        ),
        height=400,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_empty_conditional_streak_chart():
    """Create empty conditional streak chart placeholder"""
    fig = go.Figure()
    fig.add_annotation(
        text="Load trade data to view conditional win rates",
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
