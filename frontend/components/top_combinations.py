"""
Top Combinations Panel Component
Display top probability combinations with filtering capabilities
"""
import dash_bootstrap_components as dbc
from dash import html
import pandas as pd


def create_combination_card(rank, conditions, probability, sample_size, lift_ratio, mean_r):
    """
    Create a card for a single combination
    
    Parameters:
    -----------
    rank : int
        Ranking position
    conditions : str
        Description of conditions
    probability : float
        Win probability
    sample_size : int
        Number of trades
    lift_ratio : float
        Lift over base rate
    mean_r : float
        Mean R-multiple
    
    Returns:
    --------
    dbc.Card
        Combination card component
    """
    # Color based on probability
    if probability >= 0.7:
        color = "success"
        icon = "bi-trophy-fill"
    elif probability >= 0.6:
        color = "info"
        icon = "bi-star-fill"
    elif probability >= 0.5:
        color = "warning"
        icon = "bi-check-circle-fill"
    else:
        color = "secondary"
        icon = "bi-circle-fill"
    
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5([
                        html.I(className=f"bi {icon} me-2"),
                        f"#{rank}"
                    ], className="mb-2")
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.Strong(conditions),
                        html.Br(),
                        html.Small([
                            html.Span(f"Win Rate: ", className="text-muted"),
                            html.Span(f"{probability:.1%}", className=f"text-{color} fw-bold"),
                            html.Span(f" | n={sample_size}", className="text-muted ms-2"),
                            html.Span(f" | Lift: {lift_ratio:.2f}x", className="text-muted ms-2"),
                            html.Span(f" | Avg R: {mean_r:.2f}", className="text-muted ms-2")
                        ])
                    ])
                ], width=10)
            ])
        ], className="py-2")
    ], className="mb-2", color=color, outline=True)


def create_top_combinations_list(prob_results, top_n=10):
    """
    Create list of top probability combinations
    
    Parameters:
    -----------
    prob_results : pd.DataFrame
        Probability results (1D or 2D)
    top_n : int
        Number of top combinations to show
    
    Returns:
    --------
    list
        List of combination cards
    """
    if prob_results is None or len(prob_results) == 0:
        return [
            dbc.Alert(
                "No combinations available. Calculate probabilities first.",
                color="info",
                className="mb-0"
            )
        ]
    
    # Sort by probability descending
    sorted_results = prob_results.sort_values('p_est', ascending=False).head(top_n)
    
    # Filter out low sample size
    min_samples = 20
    sorted_results = sorted_results[sorted_results['n'] >= min_samples]
    
    if len(sorted_results) == 0:
        return [
            dbc.Alert(
                "No reliable combinations found (all have insufficient sample size).",
                color="warning",
                className="mb-0"
            )
        ]
    
    # Calculate base rate for lift
    base_rate = prob_results['p_est'].mean()
    
    # Create cards
    cards = []
    for rank, (idx, row) in enumerate(sorted_results.iterrows(), 1):
        # Build condition description
        if 'label' in row:
            # 1D result
            conditions = f"{row['label']}"
        else:
            # 2D result
            conditions = f"X: bin {row['bin_x']}, Y: bin {row['bin_y']}"
        
        lift_ratio = row['p_est'] / base_rate if base_rate > 0 else 1.0
        
        card = create_combination_card(
            rank=rank,
            conditions=conditions,
            probability=row['p_est'],
            sample_size=int(row['n']),
            lift_ratio=lift_ratio,
            mean_r=row['mean_R']
        )
        cards.append(card)
    
    return cards


def create_combinations_summary(prob_results):
    """
    Create summary statistics for combinations
    
    Parameters:
    -----------
    prob_results : pd.DataFrame
        Probability results
    
    Returns:
    --------
    dbc.Alert
        Summary alert component
    """
    if prob_results is None or len(prob_results) == 0:
        return None
    
    min_samples = 20
    reliable_results = prob_results[prob_results['n'] >= min_samples]
    
    if len(reliable_results) == 0:
        return dbc.Alert(
            "No reliable combinations found (insufficient sample sizes).",
            color="warning",
            className="mb-3"
        )
    
    best_prob = reliable_results['p_est'].max()
    worst_prob = reliable_results['p_est'].min()
    avg_prob = reliable_results['p_est'].mean()
    total_combinations = len(reliable_results)
    
    return dbc.Alert([
        html.H6("Summary", className="alert-heading"),
        html.Hr(),
        html.P([
            html.Strong("Total Combinations: "),
            f"{total_combinations}",
            html.Br(),
            html.Strong("Best Win Rate: "),
            html.Span(f"{best_prob:.1%}", className="text-success fw-bold"),
            html.Br(),
            html.Strong("Worst Win Rate: "),
            html.Span(f"{worst_prob:.1%}", className="text-danger fw-bold"),
            html.Br(),
            html.Strong("Average Win Rate: "),
            f"{avg_prob:.1%}"
        ], className="mb-0")
    ], color="light", className="mb-3")


def format_combination_for_export(prob_results, top_n=20):
    """
    Format combinations for CSV export
    
    Parameters:
    -----------
    prob_results : pd.DataFrame
        Probability results
    top_n : int
        Number of top combinations to export
    
    Returns:
    --------
    pd.DataFrame
        Formatted dataframe for export
    """
    if prob_results is None or len(prob_results) == 0:
        return pd.DataFrame()
    
    # Sort and filter
    min_samples = 20
    sorted_results = prob_results[prob_results['n'] >= min_samples].sort_values(
        'p_est', ascending=False
    ).head(top_n)
    
    # Calculate lift
    base_rate = prob_results['p_est'].mean()
    sorted_results['lift_ratio'] = sorted_results['p_est'] / base_rate
    
    # Select columns for export
    export_cols = [
        'p_est', 'ci_lower', 'ci_upper', 'n', 
        'mean_R', 'p_hit_1R', 'p_hit_2R', 'lift_ratio'
    ]
    
    # Add bin/label columns if they exist
    if 'label' in sorted_results.columns:
        export_cols.insert(0, 'label')
    if 'bin_x' in sorted_results.columns:
        export_cols.insert(0, 'bin_x')
    if 'bin_y' in sorted_results.columns:
        export_cols.insert(0, 'bin_y')
    
    export_df = sorted_results[export_cols].copy()
    
    # Rename columns for clarity
    export_df.columns = [
        col.replace('p_est', 'win_rate')
           .replace('ci_lower', 'ci_lower_95')
           .replace('ci_upper', 'ci_upper_95')
           .replace('n', 'sample_size')
           .replace('mean_R', 'avg_r_multiple')
           .replace('p_hit_1R', 'prob_hit_1r')
           .replace('p_hit_2R', 'prob_hit_2r')
        for col in export_df.columns
    ]
    
    return export_df
