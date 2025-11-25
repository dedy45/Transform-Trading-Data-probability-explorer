"""
Feature Importance Display Component

This component creates visualizations for feature importance analysis:
1. Bar chart showing top 10 features from LightGBM classifier
2. SHAP summary plot if SHAP values are available
3. Feature contribution display for individual predictions
4. Comparison chart between classifier and quantile models
5. Export functionality for feature importance to CSV

**Feature: ml-prediction-engine**
**Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5**
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_feature_importance_bar_chart(
    feature_importance_df: pd.DataFrame,
    top_n: int = 10,
    title: str = "Feature Importance"
) -> go.Figure:
    """
    Create bar chart for top N features.
    
    Parameters
    ----------
    feature_importance_df : pd.DataFrame
        DataFrame with columns: feature, importance, rank
    top_n : int, default=10
        Number of top features to display
    title : str, default="Feature Importance"
        Chart title
    
    Returns
    -------
    plotly.graph_objects.Figure
        Bar chart figure
    """
    # Get top N features
    top_features = feature_importance_df.head(top_n).copy()
    
    # Sort by importance for better visualization
    top_features = top_features.sort_values('importance', ascending=True)
    
    # Create color scale based on importance
    max_importance = top_features['importance'].max()
    colors = [
        f'rgba(0, 100, 200, {0.3 + 0.7 * (imp / max_importance)})'
        for imp in top_features['importance']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0, 100, 200, 1)', width=1)
        ),
        text=[f"{imp:.0f}" for imp in top_features['importance']],
        textposition='outside',
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Importance: %{x:.2f}<br>'
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=max(400, top_n * 40),  # Dynamic height based on number of features
        showlegend=False,
        template='plotly_white',
        margin=dict(l=150, r=50, t=50, b=50)
    )
    
    return fig


def create_shap_summary_plot(
    shap_values: np.ndarray,
    feature_values: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 10
) -> go.Figure:
    """
    Create SHAP summary plot showing feature contributions.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values for each feature (shape: n_samples x n_features)
    feature_values : pd.DataFrame
        Feature values for each sample
    feature_names : list
        Names of features
    top_n : int, default=10
        Number of top features to display
    
    Returns
    -------
    plotly.graph_objects.Figure
        SHAP summary plot figure
    """
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame for sorting
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    # Get top N features
    top_features = shap_importance.head(top_n)['feature'].tolist()
    top_indices = [feature_names.index(f) for f in top_features]
    
    # Create figure
    fig = go.Figure()
    
    # For each feature, create a scatter plot of SHAP values colored by feature value
    for idx, feature in enumerate(top_features):
        feature_idx = top_indices[idx]
        
        # Get SHAP values and feature values for this feature
        shap_vals = shap_values[:, feature_idx]
        feat_vals = feature_values[feature].values
        
        # Normalize feature values for color scale
        feat_vals_norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-10)
        
        fig.add_trace(go.Scatter(
            x=shap_vals,
            y=[feature] * len(shap_vals),
            mode='markers',
            marker=dict(
                size=6,
                color=feat_vals_norm,
                colorscale='RdBu',
                showscale=(idx == 0),  # Show colorbar only for first trace
                colorbar=dict(
                    title='Feature<br>Value',
                    x=1.1
                ),
                line=dict(color='white', width=0.5),
                opacity=0.6
            ),
            name=feature,
            showlegend=False,
            hovertemplate=(
                f'<b>{feature}</b><br>'
                'SHAP: %{x:.3f}<br>'
                'Value: %{customdata:.3f}<br>'
                '<extra></extra>'
            ),
            customdata=feat_vals
        ))
    
    fig.update_layout(
        title='SHAP Summary Plot',
        xaxis_title='SHAP Value (Impact on Model Output)',
        yaxis_title='Feature',
        height=max(400, top_n * 40),
        template='plotly_white',
        margin=dict(l=150, r=100, t=50, b=50)
    )
    
    # Add vertical line at x=0
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    
    return fig


def create_feature_contribution_display(
    feature_values: Dict[str, float],
    feature_importance: pd.DataFrame,
    shap_values: Optional[Dict[str, float]] = None,
    top_n: int = 10
) -> go.Figure:
    """
    Display feature contribution for a single prediction.
    
    Parameters
    ----------
    feature_values : dict
        Feature values for the sample being predicted
    feature_importance : pd.DataFrame
        Global feature importance from model
    shap_values : dict, optional
        SHAP values for this specific sample
    top_n : int, default=10
        Number of top features to display
    
    Returns
    -------
    plotly.graph_objects.Figure
        Feature contribution chart
    """
    # Get top N features by importance
    top_features = feature_importance.head(top_n)
    
    # Create data for visualization
    features = []
    values = []
    importances = []
    contributions = []
    
    for _, row in top_features.iterrows():
        feature = row['feature']
        if feature in feature_values:
            features.append(feature)
            values.append(feature_values[feature])
            importances.append(row['importance'])
            
            # Use SHAP value if available, otherwise use importance * value
            if shap_values and feature in shap_values:
                contributions.append(shap_values[feature])
            else:
                # Approximate contribution as normalized importance * value
                contributions.append(row['importance'] * feature_values[feature] / 100)
    
    # Sort by absolute contribution
    sorted_indices = np.argsort(np.abs(contributions))
    features = [features[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    contributions = [contributions[i] for i in sorted_indices]
    
    # Create colors based on contribution direction
    colors = ['green' if c > 0 else 'red' for c in contributions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=contributions,
        y=features,
        orientation='h',
        marker=dict(color=colors, opacity=0.7),
        text=[f"{c:+.3f}" for c in contributions],
        textposition='outside',
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Value: %{customdata[0]:.3f}<br>'
            'Contribution: %{x:+.3f}<br>'
            '<extra></extra>'
        ),
        customdata=[[v] for v in values]
    ))
    
    fig.update_layout(
        title='Feature Contribution for This Prediction',
        xaxis_title='Contribution to Prediction',
        yaxis_title='Feature',
        height=max(400, len(features) * 40),
        showlegend=False,
        template='plotly_white',
        margin=dict(l=150, r=50, t=50, b=50)
    )
    
    # Add vertical line at x=0
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    
    return fig


def create_comparison_chart(
    classifier_importance: pd.DataFrame,
    quantile_importance_dict: Dict[str, pd.DataFrame],
    top_n: int = 10
) -> go.Figure:
    """
    Create comparison chart between classifier and quantile models.
    
    Parameters
    ----------
    classifier_importance : pd.DataFrame
        Feature importance from classifier
    quantile_importance_dict : dict
        Dictionary with keys 'p10', 'p50', 'p90' and values as DataFrames
    top_n : int, default=10
        Number of top features to display
    
    Returns
    -------
    plotly.graph_objects.Figure
        Comparison chart figure
    """
    # Get union of top features from all models
    all_features = set()
    
    # Add top features from classifier
    all_features.update(classifier_importance.head(top_n)['feature'].tolist())
    
    # Add top features from quantile models
    for quantile_df in quantile_importance_dict.values():
        all_features.update(quantile_df.head(top_n)['feature'].tolist())
    
    # Limit to top_n overall
    all_features = list(all_features)[:top_n]
    
    # Create data for each model
    fig = go.Figure()
    
    # Classifier importance
    classifier_data = []
    for feature in all_features:
        row = classifier_importance[classifier_importance['feature'] == feature]
        if len(row) > 0:
            classifier_data.append(row['importance'].values[0])
        else:
            classifier_data.append(0)
    
    fig.add_trace(go.Bar(
        name='Classifier',
        x=all_features,
        y=classifier_data,
        marker=dict(color='rgba(0, 100, 200, 0.7)')
    ))
    
    # Quantile model importances
    colors = {
        'p10': 'rgba(200, 0, 0, 0.7)',
        'p50': 'rgba(0, 200, 0, 0.7)',
        'p90': 'rgba(200, 100, 0, 0.7)'
    }
    
    for quantile_name, quantile_df in quantile_importance_dict.items():
        quantile_data = []
        for feature in all_features:
            row = quantile_df[quantile_df['feature'] == feature]
            if len(row) > 0:
                quantile_data.append(row['importance'].values[0])
            else:
                quantile_data.append(0)
        
        fig.add_trace(go.Bar(
            name=f'Quantile {quantile_name.upper()}',
            x=all_features,
            y=quantile_data,
            marker=dict(color=colors.get(quantile_name, 'gray'))
        ))
    
    fig.update_layout(
        title='Feature Importance Comparison Across Models',
        xaxis_title='Feature',
        yaxis_title='Importance Score',
        barmode='group',
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        template='plotly_white',
        xaxis=dict(tickangle=-45),
        margin=dict(l=50, r=50, t=50, b=150)
    )
    
    return fig


def export_feature_importance_to_csv(
    feature_importance_df: pd.DataFrame,
    filename: str = 'feature_importance.csv'
) -> str:
    """
    Export feature importance to CSV file.
    
    Parameters
    ----------
    feature_importance_df : pd.DataFrame
        Feature importance DataFrame
    filename : str, default='feature_importance.csv'
        Output filename
    
    Returns
    -------
    str
        Path to saved file
    """
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Save to CSV
    output_path = output_dir / filename
    feature_importance_df.to_csv(output_path, index=False)
    
    return str(output_path)


def create_feature_importance_section(
    classifier_importance: pd.DataFrame,
    quantile_importance_dict: Optional[Dict[str, pd.DataFrame]] = None,
    shap_values: Optional[np.ndarray] = None,
    feature_values: Optional[pd.DataFrame] = None,
    feature_names: Optional[List[str]] = None,
    sample_feature_values: Optional[Dict[str, float]] = None,
    sample_shap_values: Optional[Dict[str, float]] = None,
    top_n: int = 10
) -> dbc.Container:
    """
    Create complete feature importance section with all visualizations.
    
    Parameters
    ----------
    classifier_importance : pd.DataFrame
        Feature importance from classifier
    quantile_importance_dict : dict, optional
        Feature importance from quantile models
    shap_values : np.ndarray, optional
        SHAP values for all samples
    feature_values : pd.DataFrame, optional
        Feature values for all samples
    feature_names : list, optional
        Names of features
    sample_feature_values : dict, optional
        Feature values for single sample prediction
    sample_shap_values : dict, optional
        SHAP values for single sample prediction
    top_n : int, default=10
        Number of top features to display
    
    Returns
    -------
    dash_bootstrap_components.Container
        Container with all feature importance visualizations
    """
    # Create bar chart
    bar_chart = dcc.Graph(
        figure=create_feature_importance_bar_chart(classifier_importance, top_n),
        config={'displayModeBar': True, 'toImageButtonOptions': {'filename': 'feature_importance'}}
    )
    
    # Create SHAP summary plot if available
    shap_plot = None
    if shap_values is not None and feature_values is not None and feature_names is not None:
        shap_plot = dcc.Graph(
            figure=create_shap_summary_plot(shap_values, feature_values, feature_names, top_n),
            config={'displayModeBar': True, 'toImageButtonOptions': {'filename': 'shap_summary'}}
        )
    
    # Create feature contribution display if sample data available
    contribution_plot = None
    if sample_feature_values is not None:
        contribution_plot = dcc.Graph(
            figure=create_feature_contribution_display(
                sample_feature_values,
                classifier_importance,
                sample_shap_values,
                top_n
            ),
            config={'displayModeBar': True, 'toImageButtonOptions': {'filename': 'feature_contribution'}}
        )
    
    # Create comparison chart if quantile importance available
    comparison_plot = None
    if quantile_importance_dict is not None:
        comparison_plot = dcc.Graph(
            figure=create_comparison_chart(classifier_importance, quantile_importance_dict, top_n),
            config={'displayModeBar': True, 'toImageButtonOptions': {'filename': 'model_comparison'}}
        )
    
    # Create export button
    export_button = dbc.Button(
        [html.I(className="bi bi-download me-2"), "Export to CSV"],
        id='export-feature-importance-btn',
        color='primary',
        size='sm',
        className='mb-3'
    )
    
    # Assemble section
    section = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4([
                    html.I(className="bi bi-bar-chart-fill me-2"),
                    "Feature Importance Analysis"
                ], className="mb-3"),
                html.P(
                    "Understanding which features drive model predictions helps identify key market conditions.",
                    className="text-muted mb-3"
                ),
                export_button
            ])
        ]),
        
        # Bar chart
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Top Features - Classifier", className="mb-3"),
                        bar_chart
                    ])
                ], className="shadow-sm mb-4")
            ])
        ]),
        
        # SHAP summary plot (if available)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("SHAP Summary Plot", className="mb-3"),
                        html.P(
                            "SHAP values show how each feature contributes to predictions. "
                            "Red indicates high feature values, blue indicates low values.",
                            className="text-muted small mb-3"
                        ),
                        shap_plot if shap_plot else html.Div([
                            html.I(className="bi bi-info-circle fs-1 text-muted"),
                            html.P("SHAP values not available", className="text-muted mt-2")
                        ], className="text-center py-5")
                    ])
                ], className="shadow-sm mb-4")
            ])
        ]) if shap_plot or True else None,  # Always show section, even if empty
        
        # Feature contribution for sample (if available)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Feature Contribution - Current Prediction", className="mb-3"),
                        html.P(
                            "Shows how each feature contributes to the current prediction. "
                            "Green bars push prediction higher, red bars push it lower.",
                            className="text-muted small mb-3"
                        ),
                        contribution_plot if contribution_plot else html.Div([
                            html.I(className="bi bi-info-circle fs-1 text-muted"),
                            html.P("No sample prediction available", className="text-muted mt-2")
                        ], className="text-center py-5")
                    ])
                ], className="shadow-sm mb-4")
            ])
        ]) if contribution_plot or True else None,  # Always show section
        
        # Model comparison (if available)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Model Comparison", className="mb-3"),
                        html.P(
                            "Comparing feature importance across classifier and quantile regression models. "
                            "Different models may prioritize different features.",
                            className="text-muted small mb-3"
                        ),
                        comparison_plot if comparison_plot else html.Div([
                            html.I(className="bi bi-info-circle fs-1 text-muted"),
                            html.P("Quantile model importance not available", className="text-muted mt-2")
                        ], className="text-center py-5")
                    ])
                ], className="shadow-sm mb-4")
            ])
        ]) if comparison_plot or True else None  # Always show section
    ], fluid=True)
    
    return section


def create_empty_feature_importance_section() -> dbc.Container:
    """
    Create empty feature importance section (no data state).
    
    Returns
    -------
    dash_bootstrap_components.Container
        Empty container with placeholder
    """
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4([
                    html.I(className="bi bi-bar-chart-fill me-2"),
                    "Feature Importance Analysis"
                ], className="mb-3"),
                html.P(
                    "Understanding which features drive model predictions helps identify key market conditions.",
                    className="text-muted mb-3"
                )
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="bi bi-info-circle fs-1 text-muted"),
                            html.H5("No Feature Importance Data", className="text-muted mt-3"),
                            html.P(
                                "Train a model to see feature importance analysis.",
                                className="text-muted"
                            )
                        ], className="text-center py-5")
                    ])
                ], className="shadow-sm")
            ])
        ])
    ], fluid=True)
