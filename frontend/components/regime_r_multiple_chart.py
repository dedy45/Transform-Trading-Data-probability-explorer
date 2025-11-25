"""
Regime R-Multiple Chart Component
Grouped bar chart showing P(R>=1) and P(R>=2) per regime
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_regime_r_multiple_chart(regime_threshold_df):
    """
    Create grouped bar chart showing P(R>=1) and P(R>=2) per regime
    
    Parameters:
    -----------
    regime_threshold_df : pd.DataFrame
        Results from compute_regime_threshold_probs with columns:
        - regime: Regime identifier
        - n_trades: Number of trades
        - p_r_gte_1_0: Probability of R >= 1
        - p_r_gte_1_0_ci_lower: Lower CI for P(R >= 1)
        - p_r_gte_1_0_ci_upper: Upper CI for P(R >= 1)
        - p_r_gte_2_0: Probability of R >= 2
        - p_r_gte_2_0_ci_lower: Lower CI for P(R >= 2)
        - p_r_gte_2_0_ci_upper: Upper CI for P(R >= 2)
        - reliable: Boolean for sample size adequacy
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Grouped bar chart with error bars
    
    Features:
    ---------
    - Two bars per regime (R>=1 and R>=2)
    - Error bars showing confidence intervals
    - Color-coded by threshold
    - Hover info with detailed statistics
    - Sorted by P(R>=1) descending
    """
    if regime_threshold_df is None or regime_threshold_df.empty:
        return create_empty_regime_r_multiple_chart()
    
    # Sort by P(R>=1) descending
    df = regime_threshold_df.sort_values('p_r_gte_1_0', ascending=False).copy()
    
    # Create figure
    fig = go.Figure()
    
    # Add P(R >= 1) bars
    error_y_minus_1 = df['p_r_gte_1_0'] - df['p_r_gte_1_0_ci_lower']
    error_y_plus_1 = df['p_r_gte_1_0_ci_upper'] - df['p_r_gte_1_0']
    
    fig.add_trace(go.Bar(
        name='P(R ≥ 1)',
        x=df['regime'].astype(str),
        y=df['p_r_gte_1_0'],
        marker=dict(
            color='rgba(70, 130, 180, 0.7)',
            line=dict(color='rgba(0, 0, 0, 0.3)', width=1)
        ),
        error_y=dict(
            type='data',
            symmetric=False,
            array=error_y_plus_1,
            arrayminus=error_y_minus_1,
            color='rgba(0, 0, 0, 0.5)',
            thickness=1.5,
            width=3
        ),
        text=[f"{p:.1%}" for p in df['p_r_gte_1_0']],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate=(
            '<b>Regime: %{x}</b><br>' +
            'P(R ≥ 1): %{y:.1%}<br>' +
            'CI: [%{customdata[0]:.1%}, %{customdata[1]:.1%}]<br>' +
            'Trades: %{customdata[2]}<br>' +
            '<extra></extra>'
        ),
        customdata=np.column_stack((
            df['p_r_gte_1_0_ci_lower'],
            df['p_r_gte_1_0_ci_upper'],
            df['n_trades']
        ))
    ))
    
    # Add P(R >= 2) bars
    error_y_minus_2 = df['p_r_gte_2_0'] - df['p_r_gte_2_0_ci_lower']
    error_y_plus_2 = df['p_r_gte_2_0_ci_upper'] - df['p_r_gte_2_0']
    
    fig.add_trace(go.Bar(
        name='P(R ≥ 2)',
        x=df['regime'].astype(str),
        y=df['p_r_gte_2_0'],
        marker=dict(
            color='rgba(50, 180, 50, 0.7)',
            line=dict(color='rgba(0, 0, 0, 0.3)', width=1)
        ),
        error_y=dict(
            type='data',
            symmetric=False,
            array=error_y_plus_2,
            arrayminus=error_y_minus_2,
            color='rgba(0, 0, 0, 0.5)',
            thickness=1.5,
            width=3
        ),
        text=[f"{p:.1%}" for p in df['p_r_gte_2_0']],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate=(
            '<b>Regime: %{x}</b><br>' +
            'P(R ≥ 2): %{y:.1%}<br>' +
            'CI: [%{customdata[0]:.1%}, %{customdata[1]:.1%}]<br>' +
            'Trades: %{customdata[2]}<br>' +
            '<extra></extra>'
        ),
        customdata=np.column_stack((
            df['p_r_gte_2_0_ci_lower'],
            df['p_r_gte_2_0_ci_upper'],
            df['n_trades']
        ))
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'R-Multiple Threshold Probabilities by Regime',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2c3e50'}
        },
        xaxis_title='Market Regime',
        yaxis_title='Probability',
        yaxis=dict(
            tickformat='.0%',
            range=[0, 1.0]
        ),
        barmode='group',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=60, r=40, t=100, b=60),
        height=450
    )
    
    # Add grid
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
    
    return fig


def create_empty_regime_r_multiple_chart():
    """Create empty placeholder chart"""
    fig = go.Figure()
    
    fig.add_annotation(
        text="No regime data available<br>Calculate regime analysis to view R-multiple probabilities",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray")
    )
    
    fig.update_layout(
        title='R-Multiple Threshold Probabilities by Regime',
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        height=450,
        plot_bgcolor='rgba(240, 240, 240, 0.2)',
        paper_bgcolor='white'
    )
    
    return fig
