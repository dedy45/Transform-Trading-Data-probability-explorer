"""
Calibration Metrics Cards Component
Summary cards displaying key calibration metrics
"""
import dash_bootstrap_components as dbc
from dash import html


def create_calibration_metrics_cards(brier_score=None, ece=None, n_samples=None, n_bins=None):
    """
    Create summary cards for calibration metrics
    
    Parameters:
    -----------
    brier_score : float or None
        Brier score (0-1, lower is better)
    ece : float or None
        Expected Calibration Error (0-1, lower is better)
    n_samples : int or None
        Total number of samples
    n_bins : int or None
        Number of calibration bins
    
    Returns:
    --------
    dash_bootstrap_components.Row
        Row containing 4 metric cards
    
    Features:
    ---------
    - Brier Score card with interpretation
    - ECE card with interpretation
    - Sample count card
    - Bin count card
    - Color coding based on metric quality
    """
    
    # Determine Brier Score quality
    if brier_score is not None:
        if brier_score < 0.05:
            brier_color = "success"
            brier_quality = "Excellent"
        elif brier_score < 0.10:
            brier_color = "info"
            brier_quality = "Good"
        elif brier_score < 0.15:
            brier_color = "warning"
            brier_quality = "Fair"
        else:
            brier_color = "danger"
            brier_quality = "Poor"
        brier_text = f"{brier_score:.4f}"
    else:
        brier_color = "secondary"
        brier_quality = "N/A"
        brier_text = "—"
    
    # Determine ECE quality
    if ece is not None:
        if ece < 0.05:
            ece_color = "success"
            ece_quality = "Excellent"
        elif ece < 0.10:
            ece_color = "info"
            ece_quality = "Good"
        elif ece < 0.15:
            ece_color = "warning"
            ece_quality = "Fair"
        else:
            ece_color = "danger"
            ece_quality = "Poor"
        ece_text = f"{ece:.4f}"
    else:
        ece_color = "secondary"
        ece_quality = "N/A"
        ece_text = "—"
    
    # Format sample count
    if n_samples is not None:
        samples_text = f"{n_samples:,}"
        samples_color = "primary"
    else:
        samples_text = "—"
        samples_color = "secondary"
    
    # Format bin count
    if n_bins is not None:
        bins_text = f"{n_bins}"
        bins_color = "primary"
    else:
        bins_text = "—"
        bins_color = "secondary"
    
    return dbc.Row([
        # Brier Score Card
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-bullseye fs-1 mb-2", style={"color": f"var(--bs-{brier_color})"}),
                        html.H6("Brier Score", className="text-muted mb-1"),
                        html.H3(brier_text, className="mb-1"),
                        dbc.Badge(brier_quality, color=brier_color, className="mb-2"),
                        html.P(
                            "Measures accuracy of probability predictions. "
                            "Lower is better (0 = perfect).",
                            className="small text-muted mb-0"
                        )
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm")
        ], md=3, className="mb-3"),
        
        # ECE Card
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-graph-up fs-1 mb-2", style={"color": f"var(--bs-{ece_color})"}),
                        html.H6("Expected Calibration Error", className="text-muted mb-1"),
                        html.H3(ece_text, className="mb-1"),
                        dbc.Badge(ece_quality, color=ece_color, className="mb-2"),
                        html.P(
                            "Weighted average deviation from perfect calibration. "
                            "Lower is better (0 = perfect).",
                            className="small text-muted mb-0"
                        )
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm")
        ], md=3, className="mb-3"),
        
        # Sample Count Card
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-collection fs-1 mb-2", style={"color": f"var(--bs-{samples_color})"}),
                        html.H6("Total Samples", className="text-muted mb-1"),
                        html.H3(samples_text, className="mb-1"),
                        dbc.Badge("Data Points", color=samples_color, className="mb-2"),
                        html.P(
                            "Number of predictions used for calibration analysis.",
                            className="small text-muted mb-0"
                        )
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm")
        ], md=3, className="mb-3"),
        
        # Bin Count Card
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-grid-3x3 fs-1 mb-2", style={"color": f"var(--bs-{bins_color})"}),
                        html.H6("Calibration Bins", className="text-muted mb-1"),
                        html.H3(bins_text, className="mb-1"),
                        dbc.Badge("Bins", color=bins_color, className="mb-2"),
                        html.P(
                            "Number of bins used for reliability diagram.",
                            className="small text-muted mb-0"
                        )
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm")
        ], md=3, className="mb-3")
    ])


def create_empty_calibration_metrics_cards():
    """Create empty calibration metrics cards"""
    return create_calibration_metrics_cards(
        brier_score=None,
        ece=None,
        n_samples=None,
        n_bins=None
    )
