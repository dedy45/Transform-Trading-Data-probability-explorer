"""
Demo script for Feature Importance Chart Component

This script demonstrates the feature importance visualization component
with sample data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from dash import Dash, html
import dash_bootstrap_components as dbc

from frontend.components.feature_importance_chart import (
    create_feature_importance_bar_chart,
    create_shap_summary_plot,
    create_feature_contribution_display,
    create_comparison_chart,
    create_feature_importance_section,
    create_empty_feature_importance_section
)


def generate_sample_data():
    """Generate sample feature importance data."""
    np.random.seed(42)
    
    # Sample features
    features = [
        'trend_strength_tf',
        'swing_position',
        'volatility_regime',
        'support_distance',
        'momentum_score',
        'time_of_day',
        'spread_ratio',
        'volume_profile',
        'rsi_14',
        'macd_signal',
        'bollinger_width',
        'atr_ratio'
    ]
    
    # Classifier importance
    classifier_importance = pd.DataFrame({
        'feature': features,
        'importance': np.random.exponential(scale=100, size=len(features))
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    classifier_importance['rank'] = range(1, len(classifier_importance) + 1)
    
    # Quantile model importances
    quantile_importance_dict = {}
    for quantile in ['p10', 'p50', 'p90']:
        quantile_importance = pd.DataFrame({
            'feature': features,
            'importance': np.random.exponential(scale=100, size=len(features))
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        quantile_importance['rank'] = range(1, len(quantile_importance) + 1)
        quantile_importance_dict[quantile] = quantile_importance
    
    # SHAP values (100 samples x 12 features)
    n_samples = 100
    shap_values = np.random.randn(n_samples, len(features)) * 0.1
    
    # Feature values for SHAP plot
    feature_values = pd.DataFrame(
        np.random.randn(n_samples, len(features)),
        columns=features
    )
    
    # Sample prediction feature values
    sample_feature_values = {
        feature: np.random.randn() for feature in features[:8]
    }
    
    # Sample SHAP values
    sample_shap_values = {
        feature: np.random.randn() * 0.1 for feature in features[:8]
    }
    
    return {
        'classifier_importance': classifier_importance,
        'quantile_importance_dict': quantile_importance_dict,
        'shap_values': shap_values,
        'feature_values': feature_values,
        'feature_names': features,
        'sample_feature_values': sample_feature_values,
        'sample_shap_values': sample_shap_values
    }


def demo_bar_chart():
    """Demo: Feature importance bar chart."""
    print("\n" + "="*60)
    print("DEMO 1: Feature Importance Bar Chart")
    print("="*60)
    
    data = generate_sample_data()
    
    fig = create_feature_importance_bar_chart(
        data['classifier_importance'],
        top_n=10
    )
    
    print(f"✓ Created bar chart with {len(fig.data)} traces")
    print(f"✓ Showing top 10 features")
    print(f"✓ Chart title: {fig.layout.title.text}")
    
    return fig


def demo_shap_summary():
    """Demo: SHAP summary plot."""
    print("\n" + "="*60)
    print("DEMO 2: SHAP Summary Plot")
    print("="*60)
    
    data = generate_sample_data()
    
    fig = create_shap_summary_plot(
        data['shap_values'],
        data['feature_values'],
        data['feature_names'],
        top_n=10
    )
    
    print(f"✓ Created SHAP summary plot with {len(fig.data)} traces")
    print(f"✓ Showing top 10 features")
    print(f"✓ Chart title: {fig.layout.title.text}")
    
    return fig


def demo_feature_contribution():
    """Demo: Feature contribution display."""
    print("\n" + "="*60)
    print("DEMO 3: Feature Contribution Display")
    print("="*60)
    
    data = generate_sample_data()
    
    fig = create_feature_contribution_display(
        data['sample_feature_values'],
        data['classifier_importance'],
        data['sample_shap_values'],
        top_n=8
    )
    
    print(f"✓ Created feature contribution chart with {len(fig.data)} traces")
    print(f"✓ Showing contributions for single prediction")
    print(f"✓ Chart title: {fig.layout.title.text}")
    
    return fig


def demo_comparison_chart():
    """Demo: Model comparison chart."""
    print("\n" + "="*60)
    print("DEMO 4: Model Comparison Chart")
    print("="*60)
    
    data = generate_sample_data()
    
    fig = create_comparison_chart(
        data['classifier_importance'],
        data['quantile_importance_dict'],
        top_n=10
    )
    
    print(f"✓ Created comparison chart with {len(fig.data)} traces")
    print(f"✓ Comparing classifier and 3 quantile models")
    print(f"✓ Chart title: {fig.layout.title.text}")
    
    return fig


def demo_full_section():
    """Demo: Complete feature importance section."""
    print("\n" + "="*60)
    print("DEMO 5: Complete Feature Importance Section")
    print("="*60)
    
    data = generate_sample_data()
    
    section = create_feature_importance_section(
        classifier_importance=data['classifier_importance'],
        quantile_importance_dict=data['quantile_importance_dict'],
        shap_values=data['shap_values'],
        feature_values=data['feature_values'],
        feature_names=data['feature_names'],
        sample_feature_values=data['sample_feature_values'],
        sample_shap_values=data['sample_shap_values'],
        top_n=10
    )
    
    print(f"✓ Created complete feature importance section")
    print(f"✓ Includes: bar chart, SHAP plot, contribution display, comparison chart")
    
    return section


def demo_empty_section():
    """Demo: Empty feature importance section."""
    print("\n" + "="*60)
    print("DEMO 6: Empty Feature Importance Section")
    print("="*60)
    
    section = create_empty_feature_importance_section()
    
    print(f"✓ Created empty feature importance section")
    print(f"✓ Shows placeholder when no data available")
    
    return section


def create_demo_app():
    """Create Dash app with all demos."""
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
    
    data = generate_sample_data()
    
    app.layout = dbc.Container([
        html.H1("Feature Importance Chart Component Demo", className="my-4"),
        
        html.Hr(),
        
        # Demo 1: Bar Chart
        html.H3("1. Feature Importance Bar Chart", className="mt-4 mb-3"),
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    create_feature_importance_bar_chart(
                        data['classifier_importance'],
                        top_n=10
                    )
                ], style={'height': '500px'})
            ])
        ], className="mb-4"),
        
        # Demo 2: SHAP Summary
        html.H3("2. SHAP Summary Plot", className="mt-4 mb-3"),
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    create_shap_summary_plot(
                        data['shap_values'],
                        data['feature_values'],
                        data['feature_names'],
                        top_n=10
                    )
                ], style={'height': '500px'})
            ])
        ], className="mb-4"),
        
        # Demo 3: Feature Contribution
        html.H3("3. Feature Contribution Display", className="mt-4 mb-3"),
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    create_feature_contribution_display(
                        data['sample_feature_values'],
                        data['classifier_importance'],
                        data['sample_shap_values'],
                        top_n=8
                    )
                ], style={'height': '400px'})
            ])
        ], className="mb-4"),
        
        # Demo 4: Comparison Chart
        html.H3("4. Model Comparison Chart", className="mt-4 mb-3"),
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    create_comparison_chart(
                        data['classifier_importance'],
                        data['quantile_importance_dict'],
                        top_n=10
                    )
                ], style={'height': '500px'})
            ])
        ], className="mb-4"),
        
        # Demo 5: Full Section
        html.H3("5. Complete Feature Importance Section", className="mt-4 mb-3"),
        create_feature_importance_section(
            classifier_importance=data['classifier_importance'],
            quantile_importance_dict=data['quantile_importance_dict'],
            shap_values=data['shap_values'],
            feature_values=data['feature_values'],
            feature_names=data['feature_names'],
            sample_feature_values=data['sample_feature_values'],
            sample_shap_values=data['sample_shap_values'],
            top_n=10
        ),
        
        # Demo 6: Empty Section
        html.H3("6. Empty Feature Importance Section", className="mt-4 mb-3"),
        create_empty_feature_importance_section(),
        
    ], fluid=True, className="py-4")
    
    return app


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Feature Importance Chart Component Demo")
    print("="*60)
    
    # Run individual demos
    demo_bar_chart()
    demo_shap_summary()
    demo_feature_contribution()
    demo_comparison_chart()
    demo_full_section()
    demo_empty_section()
    
    print("\n" + "="*60)
    print("Starting Dash app...")
    print("Open http://127.0.0.1:8050 in your browser")
    print("="*60 + "\n")
    
    # Create and run app
    app = create_demo_app()
    app.run_server(debug=True, port=8050)
