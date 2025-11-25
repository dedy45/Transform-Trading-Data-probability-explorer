"""
Regime Win Rate Chart Component
Bar chart showing win rate per market regime
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_regime_winrate_chart(regime_probs_df):
    """
    Create bar chart showing win rate per regime with confidence intervals
    
    Parameters:
    -----------
    regime_probs_df : pd.DataFrame
        Results from compute_regime_probabilities with columns:
        - regime: Regime identifier
        - n_trades: Number of trades
        - win_rate: Win rate (proportion)
        - ci_lower: Lower CI bound
        - ci_upper: Upper CI bound
        - reliable: Boolean for sample size adequacy
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart with error bars showing win rate per regime
    
    Features:
    ---------
    - Color-coded bars (green for >50%, red for <50%)
    - Error bars showing confidence intervals
    - Sample size annotations
    - Horizontal reference line at 50%
    - Hover info with detailed statistics
    """
    if regime_probs_df is None or regime_probs_df.empty:
        return create_empty_regime_winrate_chart()
    
    # Sort by win rate descending
    df = regime_probs_df.sort_values('win_rate', ascending=False).copy()
    
    # Determine bar colors based on win rate
    colors = ['rgba(50, 180, 50, 0.7)' if wr >= 0.5 else 'rgba(220, 50, 50, 0.7)' 
              for wr in df['win_rate']]
    
    # Calculate error bar sizes
    error_y_minus = df['win_rate'] - df['ci_lower']
    error_y_plus = df['ci_upper'] - df['win_rate']
    
    # Create bar chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=df['regime'].astype(str),
        y=df['win_rate'],
        name='Win Rate',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0, 0, 0, 0.3)', width=1)
        ),
        error_y=dict(
            type='data',
            symmetric=False,
            array=error_y_plus,
            arrayminus=error_y_minus,
            color='rgba(0, 0, 0, 0.5)',
            thickness=1.5,
            width=4
        ),
        text=[f"{wr:.1%}" for wr in df['win_rate']],
        textposition='outside',
        textfont=dict(size=11, color='black'),
        hovertemplate=(
            '<b>Regime: %{x}</b><br>' +
            'Win Rate: %{y:.1%}<br>' +
            'CI: [%{customdata[0]:.1%}, %{customdata[1]:.1%}]<br>' +
            'Trades: %{customdata[2]}<br>' +
            'Wins: %{customdata[3]}<br>' +
            'Reliable: %{customdata[4]}<br>' +
            '<extra></extra>'
        ),
        customdata=np.column_stack((
            df['ci_lower'],
            df['ci_upper'],
            df['n_trades'],
            df['n_wins'],
            df['reliable'].map({True: 'Yes', False: 'No'})
        ))
    ))
    
    # Add 50% reference line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="50% (Break-even)",
        annotation_position="right"
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Win Rate by Market Regime',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2c3e50'}
        },
        xaxis_title='Market Regime',
        yaxis_title='Win Rate',
        yaxis=dict(
            tickformat='.0%',
            range=[0, max(1.0, df['ci_upper'].max() * 1.1)]
        ),
        hovermode='x unified',
        showlegend=False,
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=60, r=40, t=80, b=60),
        height=450
    )
    
    # Add grid
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
    
    return fig


def create_empty_regime_winrate_chart():
    """Create empty placeholder chart"""
    fig = go.Figure()
    
    fig.add_annotation(
        text="No regime data available<br>Calculate regime analysis to view win rates",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray")
    )
    
    fig.update_layout(
        title='Win Rate by Market Regime',
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        height=450,
        plot_bgcolor='rgba(240, 240, 240, 0.2)',
        paper_bgcolor='white'
    )
    
    return fig
