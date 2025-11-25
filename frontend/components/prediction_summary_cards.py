"""
Prediction Summary Cards Component

Summary cards displaying ML prediction results:
1. Prob Win - Calibrated win probability with gauge chart
2. Expected R - Expected R_multiple (P50) with direction icon
3. Interval R - Prediction interval [P10_conf, P90_conf] with bar visualization
4. Setup Quality - Quality category (A+/A/B/C) with color-coded badge
5. Recommendation - Trade recommendation (TRADE/SKIP) with label
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go


def create_prob_win_gauge(prob_win):
    """
    Create gauge chart for probability of win
    
    Parameters:
    -----------
    prob_win : float
        Calibrated win probability (0-1)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Gauge chart
    """
    # Determine color based on probability
    if prob_win >= 0.65:
        color = "#006400"  # dark green
    elif prob_win >= 0.55:
        color = "#32CD32"  # green
    elif prob_win >= 0.45:
        color = "#FFD700"  # yellow
    else:
        color = "#DC143C"  # red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_win * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 45], 'color': '#FFE5E5'},
                {'range': [45, 55], 'color': '#FFFACD'},
                {'range': [55, 65], 'color': '#E0FFE0'},
                {'range': [65, 100], 'color': '#C8FFC8'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prob_win * 100
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'size': 12}
    )
    
    return fig


def create_interval_bar(p10_conf, p90_conf, p50_conf=None):
    """
    Create bar visualization for prediction interval
    
    Parameters:
    -----------
    p10_conf : float
        Lower bound (P10 conformal)
    p90_conf : float
        Upper bound (P90 conformal)
    p50_conf : float, optional
        Median (P50)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart showing interval
    """
    # Calculate interval width
    interval_width = p90_conf - p10_conf
    
    # Determine color based on P50 (if available)
    if p50_conf is not None:
        if p50_conf >= 1.5:
            color = "#006400"  # dark green
        elif p50_conf >= 1.0:
            color = "#32CD32"  # green
        elif p50_conf >= 0.5:
            color = "#FFD700"  # yellow
        else:
            color = "#DC143C"  # red
    else:
        color = "#4169E1"  # royal blue
    
    fig = go.Figure()
    
    # Add interval bar
    fig.add_trace(go.Bar(
        x=[interval_width],
        y=['Interval'],
        orientation='h',
        marker=dict(color=color, opacity=0.6),
        text=[f"{p10_conf:.2f}R to {p90_conf:.2f}R"],
        textposition='inside',
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>Interval</b><br>P10: %{customdata[0]:.2f}R<br>P90: %{customdata[1]:.2f}R<br>Width: %{x:.2f}R<extra></extra>',
        customdata=[[p10_conf, p90_conf]],
        showlegend=False,
        base=p10_conf
    ))
    
    # Add P50 marker if available
    if p50_conf is not None:
        fig.add_trace(go.Scatter(
            x=[p50_conf],
            y=['Interval'],
            mode='markers+text',
            marker=dict(size=15, color='black', symbol='diamond'),
            text=[f"P50: {p50_conf:.2f}R"],
            textposition='top center',
            textfont=dict(size=12, color='black'),
            hovertemplate='<b>P50 (Median)</b><br>%{x:.2f}R<extra></extra>',
            showlegend=False
        ))
    
    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="R_multiple",
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray'
        ),
        yaxis=dict(showticklabels=False),
        font={'size': 12}
    )
    
    return fig


def create_prediction_summary_cards(
    prob_win_raw=None,
    prob_win_calibrated=None,
    R_P10_raw=None,
    R_P50_raw=None,
    R_P90_raw=None,
    R_P10_conf=None,
    R_P90_conf=None,
    quality_label=None,
    recommendation=None,
    skewness=None
):
    """
    Create prediction summary cards with 5 key metrics
    
    Parameters:
    -----------
    prob_win_raw : float or None
        Raw win probability from classifier
    prob_win_calibrated : float or None
        Calibrated win probability (0-1)
    R_P10_raw : float or None
        Raw P10 prediction
    R_P50_raw : float or None
        Raw P50 prediction (median)
    R_P90_raw : float or None
        Raw P90 prediction
    R_P10_conf : float or None
        Conformal adjusted P10
    R_P90_conf : float or None
        Conformal adjusted P90
    quality_label : str or None
        Setup quality (A+/A/B/C)
    recommendation : str or None
        Trade recommendation (TRADE/SKIP)
    skewness : float or None
        Distribution skewness
    
    Returns:
    --------
    dash_bootstrap_components.Row
        Row containing 5 prediction cards
    """
    
    # Card 1: Prob Win with Gauge Chart
    if prob_win_calibrated is not None:
        prob_win_gauge = dcc.Graph(
            figure=create_prob_win_gauge(prob_win_calibrated),
            config={'displayModeBar': False}
        )
        prob_win_text = f"{prob_win_calibrated * 100:.1f}%"
        prob_win_color = "success" if prob_win_calibrated >= 0.55 else "warning" if prob_win_calibrated >= 0.45 else "danger"
        
        # Show calibration improvement if raw is available
        if prob_win_raw is not None:
            calibration_diff = (prob_win_calibrated - prob_win_raw) * 100
            calibration_badge = dbc.Badge(
                f"{'↑' if calibration_diff > 0 else '↓'} {abs(calibration_diff):.1f}% from raw",
                color="info" if abs(calibration_diff) > 0.01 else "secondary",
                className="mt-2"
            )
        else:
            calibration_badge = None
    else:
        prob_win_gauge = html.Div([
            html.I(className="bi bi-dash-circle fs-1 text-muted"),
            html.P("No prediction", className="text-muted mt-2")
        ], className="text-center py-4")
        prob_win_text = "—"
        prob_win_color = "secondary"
        calibration_badge = None
    
    # Card 2: Expected R with Icon
    if R_P50_raw is not None:
        expected_r_value = R_P50_raw
        expected_r_text = f"{expected_r_value:.2f}R"
        
        # Determine icon and color
        if expected_r_value >= 1.5:
            expected_r_icon = "bi-arrow-up-circle-fill"
            expected_r_color = "success"
            expected_r_quality = "Excellent"
        elif expected_r_value >= 1.0:
            expected_r_icon = "bi-arrow-up-circle"
            expected_r_color = "success"
            expected_r_quality = "Good"
        elif expected_r_value >= 0.5:
            expected_r_icon = "bi-arrow-right-circle"
            expected_r_color = "warning"
            expected_r_quality = "Fair"
        elif expected_r_value >= 0:
            expected_r_icon = "bi-arrow-down-circle"
            expected_r_color = "warning"
            expected_r_quality = "Low"
        else:
            expected_r_icon = "bi-arrow-down-circle-fill"
            expected_r_color = "danger"
            expected_r_quality = "Negative"
    else:
        expected_r_text = "—"
        expected_r_icon = "bi-dash-circle"
        expected_r_color = "secondary"
        expected_r_quality = "N/A"
    
    # Card 3: Interval R with Bar Visualization
    if R_P10_conf is not None and R_P90_conf is not None:
        interval_bar = dcc.Graph(
            figure=create_interval_bar(R_P10_conf, R_P90_conf, R_P50_raw),
            config={'displayModeBar': False}
        )
        interval_width = R_P90_conf - R_P10_conf
        interval_text = f"[{R_P10_conf:.2f}R, {R_P90_conf:.2f}R]"
        interval_width_text = f"Width: {interval_width:.2f}R"
        
        # Show skewness if available
        if skewness is not None:
            if skewness > 1.2:
                skew_badge = dbc.Badge("Positive Skew (Upside)", color="success", className="mt-2")
            elif skewness < 0.8:
                skew_badge = dbc.Badge("Negative Skew (Downside)", color="danger", className="mt-2")
            else:
                skew_badge = dbc.Badge("Symmetric", color="info", className="mt-2")
        else:
            skew_badge = None
    else:
        interval_bar = html.Div([
            html.I(className="bi bi-dash-circle fs-1 text-muted"),
            html.P("No interval", className="text-muted mt-2")
        ], className="text-center py-4")
        interval_text = "—"
        interval_width_text = ""
        skew_badge = None
    
    # Card 4: Setup Quality with Color-Coded Badge
    if quality_label is not None:
        quality_colors = {
            'A+': '#006400',  # dark green
            'A': '#32CD32',   # green
            'B': '#FFD700',   # yellow
            'C': '#DC143C'    # red
        }
        quality_descriptions = {
            'A+': 'Excellent Setup',
            'A': 'Good Setup',
            'B': 'Fair Setup',
            'C': 'Poor Setup'
        }
        quality_color_name = {
            'A+': 'success',
            'A': 'success',
            'B': 'warning',
            'C': 'danger'
        }
        
        quality_badge_color = quality_color_name.get(quality_label, 'secondary')
        quality_description = quality_descriptions.get(quality_label, 'Unknown')
        quality_hex_color = quality_colors.get(quality_label, '#6c757d')
    else:
        quality_label = "—"
        quality_badge_color = "secondary"
        quality_description = "No quality assessment"
        quality_hex_color = "#6c757d"
    
    # Card 5: Recommendation with TRADE/SKIP Label
    if recommendation is not None:
        if recommendation == "TRADE":
            rec_color = "success"
            rec_icon = "bi-check-circle-fill"
            rec_description = "Setup meets quality criteria"
        else:  # SKIP
            rec_color = "danger"
            rec_icon = "bi-x-circle-fill"
            rec_description = "Setup below quality threshold"
    else:
        recommendation = "—"
        rec_color = "secondary"
        rec_icon = "bi-dash-circle"
        rec_description = "No recommendation"
    
    return dbc.Row([
        # Card 1: Prob Win
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6([
                            html.I(className="bi bi-percent me-2"),
                            "Win Probability"
                        ], className="text-muted mb-3"),
                        prob_win_gauge,
                        html.Div([
                            dbc.Badge("Calibrated", color=prob_win_color, className="me-2"),
                            calibration_badge if calibration_badge else None
                        ], className="text-center mt-2"),
                        html.P(
                            "Calibrated probability that this setup will result in a win",
                            className="small text-muted mt-2 mb-0"
                        )
                    ])
                ])
            ], className="h-100 shadow-sm")
        ], md=12, lg=6, xl=2, className="mb-3"),
        
        # Card 2: Expected R
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6([
                            html.I(className="bi bi-graph-up me-2"),
                            "Expected R"
                        ], className="text-muted mb-2"),
                        html.I(
                            className=f"{expected_r_icon} fs-1 mb-2",
                            style={"color": f"var(--bs-{expected_r_color})"}
                        ),
                        html.H2(expected_r_text, className="mb-1"),
                        dbc.Badge(expected_r_quality, color=expected_r_color, className="mb-2"),
                        html.P(
                            "Median predicted R_multiple (P50). Positive values indicate expected profit.",
                            className="small text-muted mb-0"
                        )
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm")
        ], md=12, lg=6, xl=2, className="mb-3"),
        
        # Card 3: Interval R
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6([
                            html.I(className="bi bi-arrows-expand me-2"),
                            "Prediction Interval"
                        ], className="text-muted mb-2"),
                        interval_bar,
                        html.Div([
                            html.Small(interval_text, className="text-muted d-block"),
                            html.Small(interval_width_text, className="text-muted d-block"),
                            skew_badge if skew_badge else None
                        ], className="text-center mt-2"),
                        html.P(
                            "90% confidence interval for R_multiple outcome",
                            className="small text-muted mt-2 mb-0"
                        )
                    ])
                ])
            ], className="h-100 shadow-sm")
        ], md=12, lg=6, xl=3, className="mb-3"),
        
        # Card 4: Setup Quality
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6([
                            html.I(className="bi bi-star me-2"),
                            "Setup Quality"
                        ], className="text-muted mb-2"),
                        html.Div([
                            html.Span(
                                quality_label,
                                className="display-4 fw-bold",
                                style={"color": quality_hex_color}
                            )
                        ], className="mb-2"),
                        dbc.Badge(quality_description, color=quality_badge_color, className="mb-2"),
                        html.P(
                            "Quality category based on probability and expected R thresholds",
                            className="small text-muted mb-0"
                        )
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm")
        ], md=12, lg=6, xl=2, className="mb-3"),
        
        # Card 5: Recommendation
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6([
                            html.I(className="bi bi-lightbulb me-2"),
                            "Recommendation"
                        ], className="text-muted mb-2"),
                        html.I(
                            className=f"{rec_icon} fs-1 mb-2",
                            style={"color": f"var(--bs-{rec_color})"}
                        ),
                        html.H2(recommendation, className="mb-1"),
                        dbc.Badge(rec_description, color=rec_color, className="mb-2"),
                        html.P(
                            "Trade recommendation based on setup quality assessment",
                            className="small text-muted mb-0"
                        )
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm")
        ], md=12, lg=6, xl=3, className="mb-3")
    ])


def create_empty_prediction_summary_cards():
    """
    Create empty prediction summary cards (no data state)
    
    Returns:
    --------
    dash_bootstrap_components.Row
        Row with empty cards
    """
    return create_prediction_summary_cards(
        prob_win_raw=None,
        prob_win_calibrated=None,
        R_P10_raw=None,
        R_P50_raw=None,
        R_P90_raw=None,
        R_P10_conf=None,
        R_P90_conf=None,
        quality_label=None,
        recommendation=None,
        skewness=None
    )
