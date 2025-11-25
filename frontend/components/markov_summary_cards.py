"""
Markov Summary Cards Component
Display key Markov chain metrics in card format
"""
import dash_bootstrap_components as dbc
from dash import html
import numpy as np


def create_markov_summary_cards(markov_results, streak_results):
    """
    Create summary cards for Markov chain metrics
    
    Parameters:
    -----------
    markov_results : dict
        Results from compute_first_order_markov
    streak_results : dict
        Results from compute_streak_distribution
    
    Returns:
    --------
    dash_bootstrap_components.Row
        Row containing 4 summary cards
    
    Cards:
    ------
    1. P(Win | Win) - Probability of winning after a win
    2. P(Win | Loss) - Probability of winning after a loss (recovery rate)
    3. Max Win Streak - Longest consecutive wins
    4. Max Loss Streak - Longest consecutive losses
    """
    if markov_results is None or streak_results is None:
        # Return placeholder cards
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("No Data", className="text-muted text-center"),
                        html.P("Load trade data", className="text-center small mb-0")
                    ])
                ], className="text-center")
            ], md=3)
            for _ in range(4)
        ], className="mb-3")
    
    probs = markov_results.get('probs', {})
    ci = markov_results.get('ci', {})
    
    # Card 1: P(Win | Win)
    p_win_win = probs.get('P_win_given_win', np.nan)
    ci_win_win = ci.get('P_win_given_win', {})
    
    if not np.isnan(p_win_win):
        card1_value = f"{p_win_win:.1%}"
        card1_subtitle = f"CI: [{ci_win_win.get('ci_lower', 0):.1%}, {ci_win_win.get('ci_upper', 0):.1%}]"
        card1_color = "success" if p_win_win > 0.5 else "warning"
        card1_icon = "bi-arrow-up-circle" if p_win_win > 0.5 else "bi-dash-circle"
    else:
        card1_value = "N/A"
        card1_subtitle = "Insufficient data"
        card1_color = "secondary"
        card1_icon = "bi-question-circle"
    
    # Card 2: P(Win | Loss) - Recovery Rate
    p_win_loss = probs.get('P_win_given_loss', np.nan)
    ci_win_loss = ci.get('P_win_given_loss', {})
    
    if not np.isnan(p_win_loss):
        card2_value = f"{p_win_loss:.1%}"
        card2_subtitle = f"CI: [{ci_win_loss.get('ci_lower', 0):.1%}, {ci_win_loss.get('ci_upper', 0):.1%}]"
        card2_color = "success" if p_win_loss > 0.5 else "danger"
        card2_icon = "bi-arrow-counterclockwise" if p_win_loss > 0.5 else "bi-exclamation-triangle"
    else:
        card2_value = "N/A"
        card2_subtitle = "Insufficient data"
        card2_color = "secondary"
        card2_icon = "bi-question-circle"
    
    # Card 3: Max Win Streak
    max_win_streak = streak_results.get('max_win_streak', 0)
    avg_win_streak = streak_results.get('avg_win_streak', 0)
    
    card3_value = str(max_win_streak)
    card3_subtitle = f"Avg: {avg_win_streak:.1f}"
    card3_color = "success"
    card3_icon = "bi-trophy"
    
    # Card 4: Max Loss Streak
    max_loss_streak = streak_results.get('max_loss_streak', 0)
    avg_loss_streak = streak_results.get('avg_loss_streak', 0)
    
    card4_value = str(max_loss_streak)
    card4_subtitle = f"Avg: {avg_loss_streak:.1f}"
    card4_color = "danger"
    card4_icon = "bi-exclamation-octagon"
    
    # Create cards
    cards = [
        {
            'title': 'P(Win | Win)',
            'value': card1_value,
            'subtitle': card1_subtitle,
            'color': card1_color,
            'icon': card1_icon,
            'description': 'Momentum: Win after win'
        },
        {
            'title': 'P(Win | Loss)',
            'value': card2_value,
            'subtitle': card2_subtitle,
            'color': card2_color,
            'icon': card2_icon,
            'description': 'Recovery: Win after loss'
        },
        {
            'title': 'Max Win Streak',
            'value': card3_value,
            'subtitle': card3_subtitle,
            'color': card3_color,
            'icon': card3_icon,
            'description': 'Longest winning run'
        },
        {
            'title': 'Max Loss Streak',
            'value': card4_value,
            'subtitle': card4_subtitle,
            'color': card4_color,
            'icon': card4_icon,
            'description': 'Longest losing run'
        }
    ]
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className=f"{card['icon']} me-2", style={'fontSize': '1.5rem'}),
                        html.Span(card['title'], className="fw-bold")
                    ], className="d-flex align-items-center mb-2"),
                    html.H3(card['value'], className=f"text-{card['color']} mb-1"),
                    html.P(card['subtitle'], className="text-muted small mb-1"),
                    html.P(card['description'], className="text-muted small mb-0")
                ])
            ], className="h-100", color=card['color'], outline=True)
        ], md=3)
        for card in cards
    ], className="mb-3")


def create_empty_summary_cards():
    """Create empty summary cards placeholder"""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("No Data", className="text-muted text-center"),
                    html.P("Load trade data", className="text-center small mb-0")
                ])
            ], className="text-center")
        ], md=3)
        for _ in range(4)
    ], className="mb-3")
