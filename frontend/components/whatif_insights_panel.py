"""
What-If Scenarios Insights Panel
Auto-generated insights and recommendations based on scenario comparison
"""
import dash_bootstrap_components as dbc
from dash import html
import pandas as pd
import numpy as np


def generate_whatif_insights(comparison_df, scenarios_data):
    """
    Generate automatic insights from scenario comparison
    
    Args:
        comparison_df: DataFrame with scenario comparison metrics
        scenarios_data: List of scenario dictionaries
    
    Returns:
        List of insight dictionaries with type, message, and severity
    """
    if comparison_df is None or len(comparison_df) <= 1:
        return []
    
    insights = []
    
    # Exclude baseline for comparison
    scenarios_df = comparison_df.iloc[1:].copy()
    baseline = comparison_df.iloc[0]
    
    if len(scenarios_df) == 0:
        return []
    
    # Insight 1: Best Overall Scenario
    if 'total_profit' in scenarios_df.columns and 'max_drawdown' in scenarios_df.columns:
        # Calculate profit/drawdown ratio (avoid division by zero)
        scenarios_df['profit_dd_ratio'] = scenarios_df['total_profit'] / (abs(scenarios_df['max_drawdown']) + 0.01)
        best_idx = scenarios_df['profit_dd_ratio'].idxmax()
        best_scenario = scenarios_df.loc[best_idx]
        
        # Calculate profit improvement percentage
        if abs(baseline['total_profit']) > 0:
            profit_improvement = ((best_scenario['total_profit'] - baseline['total_profit']) / 
                                 abs(baseline['total_profit']) * 100)
        else:
            profit_improvement = 0
        
        # Calculate DD improvement (positive = better)
        dd_improvement = abs(baseline['max_drawdown']) - abs(best_scenario['max_drawdown'])
        
        insights.append({
            'type': 'success',
            'icon': 'trophy',
            'title': 'Best Overall Scenario',
            'message': f"<b>{best_scenario['scenario_name']}</b> shows the best profit/drawdown ratio. "
                      f"Profit: {profit_improvement:+.1f}%, Drawdown: {dd_improvement:+.1f}%",
            'priority': 1
        })
    
    # Insight 2: Highest Profit Scenario
    if 'total_profit' in scenarios_df.columns:
        best_profit_idx = scenarios_df['total_profit'].idxmax()
        best_profit_scenario = scenarios_df.loc[best_profit_idx]
        
        # Calculate profit change percentage (handle zero baseline)
        if abs(baseline['total_profit']) > 0.01:
            profit_change = ((best_profit_scenario['total_profit'] - baseline['total_profit']) / 
                            abs(baseline['total_profit']) * 100)
        else:
            profit_change = 0
        
        if profit_change > 10:
            insights.append({
                'type': 'success',
                'icon': 'graph-up-arrow',
                'title': 'Highest Profit Potential',
                'message': f"<b>{best_profit_scenario['scenario_name']}</b> could increase profit by "
                          f"<b>{profit_change:.1f}%</b> (${best_profit_scenario['total_profit']:,.2f})",
                'priority': 2
            })
    
    # Insight 3: Lowest Drawdown Scenario
    if 'max_drawdown' in scenarios_df.columns:
        best_dd_idx = scenarios_df['max_drawdown'].abs().idxmin()
        best_dd_scenario = scenarios_df.loc[best_dd_idx]
        
        # Calculate reduction properly (positive = improvement)
        # If baseline is -25% and best is -15%, reduction = 10%
        dd_reduction = abs(baseline['max_drawdown']) - abs(best_dd_scenario['max_drawdown'])
        
        if dd_reduction > 2:
            insights.append({
                'type': 'info',
                'icon': 'shield-check',
                'title': 'Lowest Risk Scenario',
                'message': f"<b>{best_dd_scenario['scenario_name']}</b> reduces max drawdown by "
                          f"<b>{dd_reduction:.1f}%</b> (from {baseline['max_drawdown']:.1f}% to "
                          f"{best_dd_scenario['max_drawdown']:.1f}%)",
                'priority': 3
            })
    
    # Insight 4: Best Win Rate Improvement
    if 'win_rate' in scenarios_df.columns:
        best_wr_idx = scenarios_df['win_rate'].idxmax()
        best_wr_scenario = scenarios_df.loc[best_wr_idx]
        wr_improvement = best_wr_scenario['win_rate'] - baseline['win_rate']
        
        if wr_improvement > 5:
            insights.append({
                'type': 'info',
                'icon': 'bullseye',
                'title': 'Best Win Rate',
                'message': f"<b>{best_wr_scenario['scenario_name']}</b> improves win rate by "
                          f"<b>{wr_improvement:.1f}%</b> (from {baseline['win_rate']:.1f}% to "
                          f"{best_wr_scenario['win_rate']:.1f}%)",
                'priority': 4
            })
    
    # Insight 5: Best Sharpe Ratio
    if 'sharpe_ratio' in scenarios_df.columns:
        best_sharpe_idx = scenarios_df['sharpe_ratio'].idxmax()
        best_sharpe_scenario = scenarios_df.loc[best_sharpe_idx]
        sharpe_improvement = best_sharpe_scenario['sharpe_ratio'] - baseline['sharpe_ratio']
        
        if sharpe_improvement > 0.2:
            insights.append({
                'type': 'info',
                'icon': 'star',
                'title': 'Best Risk-Adjusted Return',
                'message': f"<b>{best_sharpe_scenario['scenario_name']}</b> has the best Sharpe ratio "
                          f"({best_sharpe_scenario['sharpe_ratio']:.2f}), improving by "
                          f"{sharpe_improvement:.2f}",
                'priority': 5
            })
    
    # Insight 6: Trade Count Warning
    if 'total_trades' in scenarios_df.columns:
        for idx, row in scenarios_df.iterrows():
            trade_reduction = ((baseline['total_trades'] - row['total_trades']) / 
                              baseline['total_trades'] * 100) if baseline['total_trades'] > 0 else 0
            
            if trade_reduction > 50:
                insights.append({
                    'type': 'warning',
                    'icon': 'exclamation-triangle',
                    'title': 'Significant Trade Reduction',
                    'message': f"<b>{row['scenario_name']}</b> reduces trades by <b>{trade_reduction:.1f}%</b> "
                              f"({row['total_trades']} vs {baseline['total_trades']}). "
                              f"Results may be less reliable with fewer trades.",
                    'priority': 6
                })
    
    # Insight 7: Negative Expectancy Warning
    if 'expectancy' in scenarios_df.columns:
        for idx, row in scenarios_df.iterrows():
            if row['expectancy'] < 0:
                insights.append({
                    'type': 'danger',
                    'icon': 'x-circle',
                    'title': 'Negative Expectancy',
                    'message': f"<b>{row['scenario_name']}</b> has negative expectancy "
                              f"(${row['expectancy']:.2f}). This scenario is not profitable.",
                    'priority': 7
                })
    
    # Insight 8: Profit Factor Analysis
    if 'profit_factor' in scenarios_df.columns:
        excellent_scenarios = scenarios_df[scenarios_df['profit_factor'] > 2.0]
        if len(excellent_scenarios) > 0:
            scenario_names = ', '.join([f"<b>{name}</b>" for name in excellent_scenarios['scenario_name'].tolist()])
            insights.append({
                'type': 'success',
                'icon': 'gem',
                'title': 'Excellent Profit Factor',
                'message': f"{scenario_names} {'has' if len(excellent_scenarios) == 1 else 'have'} "
                          f"profit factor > 2.0, indicating strong profitability.",
                'priority': 8
            })
    
    # Insight 9: Consistency Analysis
    if 'avg_r' in scenarios_df.columns and 'win_rate' in scenarios_df.columns:
        # Look for scenarios with good balance
        scenarios_df['consistency_score'] = scenarios_df['avg_r'] * (scenarios_df['win_rate'] / 100)
        best_consistency_idx = scenarios_df['consistency_score'].idxmax()
        best_consistency = scenarios_df.loc[best_consistency_idx]
        
        insights.append({
            'type': 'info',
            'icon': 'graph-up',
            'title': 'Most Consistent Scenario',
            'message': f"<b>{best_consistency['scenario_name']}</b> shows the best balance between "
                      f"win rate ({best_consistency['win_rate']:.1f}%) and avg R "
                      f"({best_consistency['avg_r']:.2f})",
            'priority': 9
        })
    
    # Insight 10: Scenario Type Recommendations
    scenario_types = {}
    for scenario in scenarios_data:
        scenario_type = scenario['type']
        if scenario_type not in scenario_types:
            scenario_types[scenario_type] = []
        scenario_types[scenario_type].append(scenario['name'])
    
    if len(scenario_types) < 3:
        missing_types = []
        all_types = ['position_sizing', 'sl_tp', 'filter', 'time', 'market_condition', 'money_management']
        for t in all_types:
            if t not in scenario_types:
                missing_types.append(t.replace('_', ' ').title())
        
        if missing_types:
            insights.append({
                'type': 'secondary',
                'icon': 'lightbulb',
                'title': 'Explore More Scenario Types',
                'message': f"Consider testing: {', '.join(missing_types[:3])} scenarios for "
                          f"comprehensive optimization.",
                'priority': 10
            })
    
    # Sort by priority
    insights.sort(key=lambda x: x['priority'])
    
    return insights


def create_whatif_insights_panel(comparison_df, scenarios_data):
    """
    Create insights panel component
    
    Args:
        comparison_df: DataFrame with scenario comparison metrics
        scenarios_data: List of scenario dictionaries
    
    Returns:
        Dash component with insights cards
    """
    insights = generate_whatif_insights(comparison_df, scenarios_data)
    
    if not insights:
        return dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            "Add scenarios and click Refresh to see insights"
        ], color="light", className="mb-0")
    
    insight_cards = []
    
    for insight in insights[:6]:  # Show top 6 insights
        color_map = {
            'success': 'success',
            'info': 'info',
            'warning': 'warning',
            'danger': 'danger',
            'secondary': 'secondary'
        }
        
        color = color_map.get(insight['type'], 'light')
        
        # Parse HTML message to Dash components
        from dash import dcc
        
        card = dbc.Alert([
            html.Div([
                html.I(className=f"bi bi-{insight['icon']} me-2"),
                html.Strong(insight['title'])
            ], className="mb-1"),
            html.Div(
                dcc.Markdown(insight['message'], dangerously_allow_html=True),
                className="text-muted small"
            )
        ], color=color, className="mb-2")
        
        insight_cards.append(card)
    
    return html.Div(insight_cards)


def create_scenario_summary_cards(comparison_df):
    """
    Create summary cards showing key statistics
    
    Args:
        comparison_df: DataFrame with scenario comparison metrics
    
    Returns:
        Dash component with summary cards
    """
    if comparison_df is None or len(comparison_df) <= 1:
        return html.Div()
    
    scenarios_df = comparison_df.iloc[1:].copy()
    baseline = comparison_df.iloc[0]
    
    cards = []
    
    # Card 1: Total Scenarios
    cards.append(
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-layers me-2 text-primary"),
                        html.Small("Total Scenarios", className="text-muted")
                    ]),
                    html.H4(str(len(scenarios_df)), className="mb-0 mt-2")
                ])
            ], className="h-100")
        ], md=3)
    )
    
    # Card 2: Best Improvement
    if 'total_profit' in scenarios_df.columns:
        best_profit = scenarios_df['total_profit'].max()
        
        # Calculate improvement percentage (handle zero/small baseline)
        if abs(baseline['total_profit']) > 0.01:
            improvement = ((best_profit - baseline['total_profit']) / 
                          abs(baseline['total_profit']) * 100)
        else:
            improvement = 0
        
        cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="bi bi-graph-up-arrow me-2 text-success"),
                            html.Small("Best Profit Improvement", className="text-muted")
                        ]),
                        html.H4(f"+{improvement:.1f}%", className="mb-0 mt-2 text-success")
                    ])
                ], className="h-100")
            ], md=3)
        )
    
    # Card 3: Best Drawdown Reduction
    if 'max_drawdown' in scenarios_df.columns:
        # Find scenario with smallest absolute drawdown (best)
        best_dd_idx = scenarios_df['max_drawdown'].abs().idxmin()
        best_dd = scenarios_df.loc[best_dd_idx, 'max_drawdown']
        
        # Debug: Print values to console
        print(f"[DD Reduction Debug]")
        print(f"  Baseline DD: {baseline['max_drawdown']}")
        print(f"  Best DD: {best_dd}")
        print(f"  Baseline DD (abs): {abs(baseline['max_drawdown'])}")
        print(f"  Best DD (abs): {abs(best_dd)}")
        
        # Validate values are reasonable (DD should be between -100% and 0%)
        baseline_dd_abs = abs(baseline['max_drawdown'])
        best_dd_abs = abs(best_dd)
        
        # If values are unreasonable, skip this card
        if baseline_dd_abs > 100 or best_dd_abs > 100:
            print(f"  WARNING: Unreasonable DD values detected!")
            print(f"  Skipping DD Reduction card")
        else:
            # Calculate reduction (positive = improvement)
            # If baseline is -25% and best is -15%, reduction = 10% (improvement)
            reduction = baseline_dd_abs - best_dd_abs
            print(f"  Reduction: {reduction}")
            
            # Display with appropriate sign and color
            if reduction > 0:
                # Positive reduction = improvement (DD decreased)
                display_text = f"↓ {reduction:.1f}%"
                text_color = "text-success"
            elif reduction < 0:
                # Negative reduction = worse (DD increased)
                display_text = f"↑ {abs(reduction):.1f}%"
                text_color = "text-danger"
            else:
                # No change
                display_text = "0.0%"
                text_color = "text-secondary"
            
            cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="bi bi-shield-check me-2 text-info"),
                                html.Small("Best DD Reduction", className="text-muted")
                            ]),
                            html.H4(display_text, className=f"mb-0 mt-2 {text_color}"),
                            html.Small(
                                f"From {baseline_dd_abs:.1f}% to {best_dd_abs:.1f}%",
                                className="text-muted"
                            )
                        ])
                    ], className="h-100")
                ], md=3)
            )
    
    # Card 4: Avg Win Rate Improvement
    if 'win_rate' in scenarios_df.columns:
        avg_wr = scenarios_df['win_rate'].mean()
        wr_change = avg_wr - baseline['win_rate']
        
        cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="bi bi-bullseye me-2 text-warning"),
                            html.Small("Avg WR Change", className="text-muted")
                        ]),
                        html.H4(f"{wr_change:+.1f}%", 
                               className=f"mb-0 mt-2 text-{'success' if wr_change > 0 else 'danger'}")
                    ])
                ], className="h-100")
            ], md=3)
        )
    
    return dbc.Row(cards, className="mb-3")


def create_recommendations_panel(comparison_df, scenarios_data):
    """
    Create recommendations panel with actionable advice
    
    Args:
        comparison_df: DataFrame with scenario comparison metrics
        scenarios_data: List of scenario dictionaries
    
    Returns:
        Dash component with recommendations
    """
    if comparison_df is None or len(comparison_df) <= 1:
        return html.Div()
    
    scenarios_df = comparison_df.iloc[1:].copy()
    baseline = comparison_df.iloc[0]
    
    recommendations = []
    
    # Recommendation 1: Best scenario to implement
    if 'total_profit' in scenarios_df.columns and 'max_drawdown' in scenarios_df.columns:
        scenarios_df['score'] = (scenarios_df['total_profit'] / abs(baseline['total_profit'])) * \
                                (1 - scenarios_df['max_drawdown'].abs() / 100)
        best_idx = scenarios_df['score'].idxmax()
        best_scenario = scenarios_df.loc[best_idx]
        
        recommendations.append({
            'title': '🎯 Recommended Implementation',
            'message': f"Start with <b>{best_scenario['scenario_name']}</b> - it offers the best "
                      f"balance of profit improvement and risk reduction.",
            'action': 'Review parameters and test on demo account first'
        })
    
    # Recommendation 2: Risk management
    if 'max_drawdown' in scenarios_df.columns:
        high_dd_scenarios = scenarios_df[scenarios_df['max_drawdown'].abs() > 25]
        if len(high_dd_scenarios) > 0:
            recommendations.append({
                'title': '⚠️ Risk Warning',
                'message': f"{len(high_dd_scenarios)} scenario(s) have drawdown > 25%. "
                          f"Consider adding money management rules.",
                'action': 'Test money management scenarios to reduce risk'
            })
    
    # Recommendation 3: Sample size
    if 'total_trades' in scenarios_df.columns:
        low_trade_scenarios = scenarios_df[scenarios_df['total_trades'] < 30]
        if len(low_trade_scenarios) > 0:
            recommendations.append({
                'title': '📊 Sample Size Concern',
                'message': f"{len(low_trade_scenarios)} scenario(s) have < 30 trades. "
                          f"Results may not be statistically significant.",
                'action': 'Relax filters or test on longer time period'
            })
    
    # Recommendation 4: Further testing
    scenario_types_tested = set([s['type'] for s in scenarios_data])
    all_types = {'position_sizing', 'sl_tp', 'filter', 'time', 'market_condition', 'money_management'}
    untested_types = all_types - scenario_types_tested
    
    if untested_types:
        type_names = [t.replace('_', ' ').title() for t in list(untested_types)[:2]]
        recommendations.append({
            'title': '💡 Explore More',
            'message': f"You haven't tested {', '.join(type_names)} scenarios yet.",
            'action': 'Add these scenario types for comprehensive optimization'
        })
    
    if not recommendations:
        return html.Div()
    
    recommendation_items = []
    for rec in recommendations[:4]:  # Show top 4
        from dash import dcc
        item = dbc.ListGroupItem([
            html.H6(rec['title'], className="mb-2"),
            html.Div(dcc.Markdown(rec['message'], dangerously_allow_html=True), className="mb-1 small"),
            html.Small([
                html.I(className="bi bi-arrow-right me-1"),
                html.Em(rec['action'])
            ], className="text-muted")
        ])
        recommendation_items.append(item)
    
    return dbc.Card([
        dbc.CardHeader(html.H6([
            html.I(className="bi bi-lightbulb me-2"),
            "Recommendations"
        ], className="mb-0")),
        dbc.CardBody([
            dbc.ListGroup(recommendation_items, flush=True)
        ])
    ], className="mb-3")
