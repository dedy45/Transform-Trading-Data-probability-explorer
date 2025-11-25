"""
Scenario Comparison Table Component
Side-by-side comparison of scenario metrics with color coding
"""
import dash_bootstrap_components as dbc
from dash import html, dash_table
import pandas as pd


def create_comparison_table(comparison_df):
    """
    Create formatted comparison table with color-coded cells
    
    Args:
        comparison_df: DataFrame with scenarios as rows and metrics as columns
    
    Returns:
        Dash DataTable component with conditional formatting
    """
    if comparison_df is None or len(comparison_df) == 0:
        return dbc.Alert(
            "No scenarios to compare. Add scenarios using the builder panel.",
            color="info"
        )
    
    # Define columns to display
    display_columns = [
        'scenario_name',
        'scenario_type',
        'total_trades',
        'win_rate',
        'avg_r',
        'expectancy',
        'total_profit',
        'max_drawdown',
        'profit_factor',
        'sharpe_ratio'
    ]
    
    # Filter to available columns
    available_columns = [col for col in display_columns if col in comparison_df.columns]
    display_df = comparison_df[available_columns].copy()
    
    # Format numeric columns
    numeric_formats = {
        'total_trades': '{:.0f}',
        'win_rate': '{:.2f}%',
        'avg_r': '{:.2f}',
        'expectancy': '${:.2f}',
        'total_profit': '${:,.2f}',
        'max_drawdown': '{:.2f}%',
        'profit_factor': '{:.2f}',
        'sharpe_ratio': '{:.2f}'
    }
    
    # Apply formatting
    for col, fmt in numeric_formats.items():
        if col in display_df.columns:
            if col == 'total_profit':
                display_df[col] = display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
            elif col in ['win_rate', 'max_drawdown']:
                display_df[col] = display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
            else:
                display_df[col] = display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
    
    # Rename columns for display
    column_names = {
        'scenario_name': 'Scenario',
        'scenario_type': 'Type',
        'total_trades': 'Trades',
        'win_rate': 'Win Rate',
        'avg_r': 'Avg R',
        'expectancy': 'Expectancy',
        'total_profit': 'Total Profit',
        'max_drawdown': 'Max DD',
        'profit_factor': 'Profit Factor',
        'sharpe_ratio': 'Sharpe'
    }
    
    display_df = display_df.rename(columns=column_names)
    
    # Create conditional formatting rules
    # Baseline row should be highlighted
    style_data_conditional = [
        {
            'if': {'row_index': 0},
            'backgroundColor': '#e3f2fd',
            'fontWeight': 'bold'
        }
    ]
    
    # Create DataTable
    table = dash_table.DataTable(
        id='scenario-comparison-datatable',
        data=display_df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in display_df.columns],
        style_table={
            'overflowX': 'auto',
            'minWidth': '100%'
        },
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontSize': '14px',
            'fontFamily': 'Arial, sans-serif'
        },
        style_header={
            'backgroundColor': '#f8f9fa',
            'fontWeight': 'bold',
            'borderBottom': '2px solid #dee2e6'
        },
        style_data_conditional=style_data_conditional,
        sort_action='native',
        filter_action='native',
        page_action='none',
        style_as_list_view=True
    )
    
    return table


def create_comparison_summary_cards(comparison_df):
    """
    Create summary cards showing best scenarios for key metrics
    
    Args:
        comparison_df: DataFrame with scenario comparison data
    
    Returns:
        Dash component with summary cards
    """
    if comparison_df is None or len(comparison_df) <= 1:
        return html.Div()
    
    # Exclude baseline (first row)
    scenarios_df = comparison_df.iloc[1:].copy()
    
    if len(scenarios_df) == 0:
        return html.Div()
    
    # Find best scenarios for each metric
    best_cards = []
    
    metrics = [
        ('win_rate', 'Highest Win Rate', 'success', 'arrow-up-circle', ''),
        ('avg_r', 'Best Avg R', 'primary', 'graph-up', ''),
        ('expectancy', 'Best Expectancy', 'info', 'currency-dollar', '$'),
        ('sharpe_ratio', 'Best Sharpe Ratio', 'warning', 'star', '')
    ]
    
    for metric, title, color, icon, prefix in metrics:
        if metric in scenarios_df.columns:
            best_idx = scenarios_df[metric].idxmax()
            best_scenario = scenarios_df.loc[best_idx]
            best_value = best_scenario[metric]
            
            card = dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className=f"bi bi-{icon} me-2"),
                        html.Span(title, className="fw-bold")
                    ], className="mb-2"),
                    html.H5(f"{prefix}{best_value:.2f}", className="mb-1"),
                    html.Small(best_scenario['scenario_name'], className="text-muted")
                ])
            ], color=color, outline=True, className="mb-2")
            
            best_cards.append(dbc.Col(card, md=3))
    
    return dbc.Row(best_cards, className="mb-3")


def format_scenario_type(scenario_type):
    """
    Format scenario type for display
    
    Args:
        scenario_type: Raw scenario type string
    
    Returns:
        Formatted display string
    """
    type_map = {
        'position_sizing': 'Position Sizing',
        'sl_tp': 'SL/TP',
        'filter': 'Filtering',
        'time': 'Time Restrictions',
        'market_condition': 'Market Conditions',
        'money_management': 'Money Management',
        'baseline': 'Baseline'
    }
    
    return type_map.get(scenario_type, scenario_type.replace('_', ' ').title())


def calculate_percentage_changes(comparison_df):
    """
    Calculate percentage changes from baseline for all metrics
    
    Args:
        comparison_df: DataFrame with scenario comparison data
    
    Returns:
        DataFrame with additional percentage change columns
    """
    if comparison_df is None or len(comparison_df) <= 1:
        return comparison_df
    
    result_df = comparison_df.copy()
    baseline = result_df.iloc[0]
    
    # Metrics to calculate changes for
    metrics = ['win_rate', 'avg_r', 'expectancy', 'total_profit', 
               'max_drawdown', 'profit_factor', 'sharpe_ratio']
    
    for metric in metrics:
        if metric in result_df.columns:
            baseline_value = baseline[metric]
            if baseline_value != 0:
                result_df[f'{metric}_change'] = (
                    (result_df[metric] - baseline_value) / abs(baseline_value) * 100
                )
            else:
                result_df[f'{metric}_change'] = 0
    
    return result_df


def get_color_for_change(value, metric):
    """
    Get color based on percentage change value
    
    Args:
        value: Percentage change value
        metric: Metric name
    
    Returns:
        Color string (green, red, or gray)
    """
    # For max_drawdown, lower is better (inverse)
    if metric == 'max_drawdown':
        if value < -5:
            return 'success'  # Green (improvement)
        elif value > 5:
            return 'danger'   # Red (worse)
        else:
            return 'secondary'  # Gray (neutral)
    else:
        # For other metrics, higher is better
        if value > 5:
            return 'success'  # Green (improvement)
        elif value < -5:
            return 'danger'   # Red (worse)
        else:
            return 'secondary'  # Gray (neutral)
