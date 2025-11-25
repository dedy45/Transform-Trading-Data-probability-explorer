"""
ML Help Modal Component

Provides comprehensive help documentation for ML Prediction Engine with 4 tabs:
1. Overview - Explain 4 ML components and workflow
2. How to Use - Step-by-step guide
3. Interpretation Guide - How to read predictions
4. FAQ - Common questions and answers
"""

import dash_bootstrap_components as dbc
from dash import html


def create_overview_tab_content():
    """Create Overview tab content explaining 4 ML components and workflow"""
    return html.Div([
        # Introduction
        html.Div([
            html.H5([
                html.I(className="bi bi-info-circle me-2"),
                "What is ML Prediction Engine?"
            ], className="mb-3"),
            html.P([
                "The ML Prediction Engine is a sophisticated machine learning system that provides ",
                html.Strong("calibrated probability predictions"), " and ",
                html.Strong("uncertainty-aware forecasts"), " for trading setups. ",
                "It combines 4 specialized ML components to deliver reliable predictions with confidence intervals."
            ], className="lead"),
        ], className="mb-4"),
        
        html.Hr(),
        
        # Component 1: LightGBM Classifier
        html.Div([
            html.H6([
                html.Span("1️⃣", className="me-2"),
                "LightGBM Binary Classifier"
            ], className="mb-2"),
            html.P([
                html.Strong("Purpose: "), "Predicts the probability of a trade being profitable (win/loss)."
            ]),
            html.Ul([
                html.Li("Uses gradient boosting to learn patterns from historical trades"),
                html.Li("Outputs raw probability (0-1) based on setup features"),
                html.Li("Trained with 5-fold time-series cross-validation"),
                html.Li("Provides feature importance to understand key drivers"),
            ]),
            dbc.Alert([
                html.I(className="bi bi-lightbulb me-2"),
                html.Strong("Key Insight: "), 
                "This component answers: 'What's the chance this trade will be profitable?'"
            ], color="info", className="mb-0"),
        ], className="mb-4"),
        
        # Component 2: Isotonic Calibration
        html.Div([
            html.H6([
                html.Span("2️⃣", className="me-2"),
                "Isotonic Calibration"
            ], className="mb-2"),
            html.P([
                html.Strong("Purpose: "), "Adjusts raw probabilities to be more accurate and trustworthy."
            ]),
            html.Ul([
                html.Li("Corrects systematic over/under-confidence in predictions"),
                html.Li("Ensures that a 70% prediction truly means ~70% win rate"),
                html.Li("Uses isotonic regression on calibration set"),
                html.Li("Improves Brier score and Expected Calibration Error (ECE)"),
            ]),
            dbc.Alert([
                html.I(className="bi bi-lightbulb me-2"),
                html.Strong("Key Insight: "), 
                "This component ensures: 'The probabilities you see are honest and reliable.'"
            ], color="info", className="mb-0"),
        ], className="mb-4"),
        
        # Component 3: Quantile Regression
        html.Div([
            html.H6([
                html.Span("3️⃣", className="me-2"),
                "Quantile Regression (P10/P50/P90)"
            ], className="mb-2"),
            html.P([
                html.Strong("Purpose: "), "Predicts the distribution of R_multiple outcomes (not just average)."
            ]),
            html.Ul([
                html.Li("Trains 3 separate models for P10 (pessimistic), P50 (typical), P90 (optimistic)"),
                html.Li("Shows worst-case, expected, and best-case scenarios"),
                html.Li("Captures asymmetry in profit/loss distribution"),
                html.Li("Helps understand risk-reward profile of each setup"),
            ]),
            dbc.Alert([
                html.I(className="bi bi-lightbulb me-2"),
                html.Strong("Key Insight: "), 
                "This component reveals: 'What range of outcomes can I expect?'"
            ], color="info", className="mb-0"),
        ], className="mb-4"),
        
        # Component 4: Conformal Prediction
        html.Div([
            html.H6([
                html.Span("4️⃣", className="me-2"),
                "Conformal Prediction"
            ], className="mb-2"),
            html.P([
                html.Strong("Purpose: "), "Provides honest uncertainty intervals with coverage guarantees."
            ]),
            html.Ul([
                html.Li("Adjusts P10-P90 intervals to achieve target coverage (90%)"),
                html.Li("Computes nonconformity scores from calibration set"),
                html.Li("Widens intervals when model is uncertain"),
                html.Li("Provides statistical guarantee: ~90% of actual outcomes fall within interval"),
            ]),
            dbc.Alert([
                html.I(className="bi bi-lightbulb me-2"),
                html.Strong("Key Insight: "), 
                "This component guarantees: 'These intervals are honest about uncertainty.'"
            ], color="info", className="mb-0"),
        ], className="mb-4"),
        
        html.Hr(),
        
        # Workflow Diagram
        html.Div([
            html.H6([
                html.I(className="bi bi-diagram-3 me-2"),
                "Prediction Workflow"
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.Div("📊 Input Features", className="text-center fw-bold mb-2"),
                            html.Small("(5-8 selected features)", className="text-muted d-block text-center")
                        ], className="p-3 bg-light rounded"),
                        
                        html.Div("⬇️", className="text-center my-2 fs-4"),
                        
                        html.Div([
                            html.Div("🤖 LightGBM Classifier", className="text-center fw-bold mb-2"),
                            html.Small("Raw Probability: 0.68", className="text-muted d-block text-center")
                        ], className="p-3 bg-light rounded"),
                        
                        html.Div("⬇️", className="text-center my-2 fs-4"),
                        
                        html.Div([
                            html.Div("📈 Isotonic Calibration", className="text-center fw-bold mb-2"),
                            html.Small("Calibrated Probability: 0.65", className="text-muted d-block text-center")
                        ], className="p-3 bg-light rounded"),
                        
                        html.Div("⬇️", className="text-center my-2 fs-4"),
                        
                        html.Div([
                            html.Div("📊 Quantile Regression", className="text-center fw-bold mb-2"),
                            html.Small("P10: -0.3R | P50: 1.2R | P90: 2.8R", className="text-muted d-block text-center")
                        ], className="p-3 bg-light rounded"),
                        
                        html.Div("⬇️", className="text-center my-2 fs-4"),
                        
                        html.Div([
                            html.Div("🎯 Conformal Prediction", className="text-center fw-bold mb-2"),
                            html.Small("Adjusted: P10: -0.5R | P90: 3.1R", className="text-muted d-block text-center")
                        ], className="p-3 bg-light rounded"),
                        
                        html.Div("⬇️", className="text-center my-2 fs-4"),
                        
                        html.Div([
                            html.Div("✅ Final Prediction", className="text-center fw-bold mb-2"),
                            html.Small("Quality: A | Recommendation: TRADE", className="text-success d-block text-center fw-bold")
                        ], className="p-3 bg-success bg-opacity-10 rounded border border-success"),
                    ])
                ])
            ], className="shadow-sm"),
        ], className="mb-4"),
    ])


def create_how_to_use_tab_content():
    """Create How to Use tab with step-by-step guide"""
    return html.Div([
        html.H5([
            html.I(className="bi bi-book me-2"),
            "Step-by-Step Guide"
        ], className="mb-4"),
        
        # Step 1: Training Models
        html.Div([
            html.H6([
                html.Span("Step 1:", className="badge bg-primary me-2"),
                "Train Models (First Time Setup)"
            ], className="mb-3"),
            
            html.Ol([
                html.Li([
                    "Click the ", html.Strong("'Train Models'"), " button in the ML Prediction Engine page"
                ]),
                html.Li([
                    "Configure training settings:",
                    html.Ul([
                        html.Li("Select 5-8 features (or use Auto Feature Selection results)"),
                        html.Li("Adjust train/calibration/test split ratios (default: 60/20/20)"),
                        html.Li("Set cross-validation folds (default: 5)"),
                    ])
                ]),
                html.Li("Click 'Start Training' and wait for completion (typically 30-60 seconds)"),
                html.Li("Review training metrics (AUC, Brier score, MAE, Coverage)"),
                html.Li("Models are saved automatically to data_processed/models/"),
            ]),
            
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                html.Strong("Note: "), 
                "You only need to train once. Retrain when you have new data or want to adjust features."
            ], color="info"),
        ], className="mb-4"),
        
        # Step 2: Single Prediction
        html.Div([
            html.H6([
                html.Span("Step 2:", className="badge bg-primary me-2"),
                "Make a Single Prediction"
            ], className="mb-3"),
            
            html.Ol([
                html.Li([
                    "Navigate to the ", html.Strong("'Input Controls'"), " section"
                ]),
                html.Li([
                    "Enter feature values for your trading setup:",
                    html.Ul([
                        html.Li("Trend strength, swing position, volatility, etc."),
                        html.Li("Or load from a saved setup"),
                    ])
                ]),
                html.Li("Click 'Predict' button"),
                html.Li([
                    "View results in ", html.Strong("'Prediction Summary'"), " cards:",
                    html.Ul([
                        html.Li("Probability Win (calibrated)"),
                        html.Li("Expected R (P50)"),
                        html.Li("Interval R (P10 to P90)"),
                        html.Li("Setup Quality (A+/A/B/C)"),
                        html.Li("Recommendation (TRADE/SKIP)"),
                    ])
                ]),
            ]),
        ], className="mb-4"),
        
        # Step 3: Batch Prediction
        html.Div([
            html.H6([
                html.Span("Step 3:", className="badge bg-primary me-2"),
                "Batch Prediction (Multiple Setups)"
            ], className="mb-3"),
            
            html.Ol([
                html.Li("Prepare a CSV file with feature columns"),
                html.Li("Click 'Upload CSV' in the Batch Prediction section"),
                html.Li("System predicts all rows automatically"),
                html.Li([
                    "View results in sortable table:",
                    html.Ul([
                        html.Li("Sort by probability, expected R, or quality"),
                        html.Li("Filter by quality (show only A+/A setups)"),
                        html.Li("Export filtered results to CSV"),
                    ])
                ]),
            ]),
            
            dbc.Alert([
                html.I(className="bi bi-lightbulb me-2"),
                html.Strong("Tip: "), 
                "Use batch prediction to screen many setups and focus on high-quality opportunities."
            ], color="success"),
        ], className="mb-4"),
        
        # Step 4: Interpret Results
        html.Div([
            html.H6([
                html.Span("Step 4:", className="badge bg-primary me-2"),
                "Interpret Predictions"
            ], className="mb-3"),
            
            html.Ol([
                html.Li([
                    html.Strong("Probability Analysis:"),
                    html.Ul([
                        html.Li("Check reliability diagram to verify calibration quality"),
                        html.Li("Review Brier score (< 0.20 is good)"),
                        html.Li("Check ECE (< 0.05 is well-calibrated)"),
                    ])
                ]),
                html.Li([
                    html.Strong("Distribution Analysis:"),
                    html.Ul([
                        html.Li("View P10-P90 fan chart to see outcome range"),
                        html.Li("Check coverage percentage (should be ~90%)"),
                        html.Li("Analyze skewness (positive = upside potential)"),
                    ])
                ]),
                html.Li([
                    html.Strong("Feature Importance:"),
                    html.Ul([
                        html.Li("Identify which features drive predictions"),
                        html.Li("Use SHAP values for detailed explanations"),
                        html.Li("Focus on controllable features for improvement"),
                    ])
                ]),
            ]),
        ], className="mb-4"),
        
        # Step 5: Adjust Settings
        html.Div([
            html.H6([
                html.Span("Step 5:", className="badge bg-primary me-2"),
                "Customize Settings (Optional)"
            ], className="mb-3"),
            
            html.Ol([
                html.Li([
                    "Click ", html.Strong("'Settings'"), " button to open configuration modal"
                ]),
                html.Li([
                    "Adjust in 4 tabs:",
                    html.Ul([
                        html.Li(html.Strong("Features: "), "Select/deselect features (5-15)"),
                        html.Li(html.Strong("Hyperparameters: "), "Tune model complexity"),
                        html.Li(html.Strong("Thresholds: "), "Adjust A+/A/B/C quality criteria"),
                        html.Li(html.Strong("Display: "), "Customize visualizations"),
                    ])
                ]),
                html.Li("Click 'Save Settings' to apply changes"),
                html.Li("Retrain models if you changed features or hyperparameters"),
            ]),
        ], className="mb-4"),
        
        # Best Practices
        html.Div([
            html.H6([
                html.I(className="bi bi-star me-2"),
                "Best Practices"
            ], className="mb-3"),
            
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.I(className="bi bi-check-circle text-success me-2"),
                    html.Strong("Start with Auto Feature Selection: "),
                    "Use top-ranked features for best results"
                ]),
                dbc.ListGroupItem([
                    html.I(className="bi bi-check-circle text-success me-2"),
                    html.Strong("Monitor model performance: "),
                    "Check metrics regularly and retrain if degradation > 10%"
                ]),
                dbc.ListGroupItem([
                    html.I(className="bi bi-check-circle text-success me-2"),
                    html.Strong("Use quality filters: "),
                    "Focus on A+ and A setups for higher win rates"
                ]),
                dbc.ListGroupItem([
                    html.I(className="bi bi-check-circle text-success me-2"),
                    html.Strong("Consider intervals: "),
                    "Don't just look at P50, understand the full P10-P90 range"
                ]),
                dbc.ListGroupItem([
                    html.I(className="bi bi-check-circle text-success me-2"),
                    html.Strong("Retrain periodically: "),
                    "Update models with new data every 1-3 months"
                ]),
            ], flush=True),
        ]),
    ])


def create_interpretation_guide_tab_content():
    """Create Interpretation Guide tab explaining how to read predictions"""
    return html.Div([
        html.H5([
            html.I(className="bi bi-compass me-2"),
            "How to Interpret Predictions"
        ], className="mb-4"),
        
        # Probability Win
        html.Div([
            html.H6([
                html.I(className="bi bi-percent me-2"),
                "Probability Win (Calibrated)"
            ], className="mb-3"),
            
            html.P([
                "This is the ", html.Strong("calibrated probability"), " that the trade will be profitable. ",
                "Unlike raw model outputs, this probability is adjusted to be honest and reliable."
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div("0.70", className="display-4 text-center text-success"),
                            html.P("70% Win Probability", className="text-center text-muted mb-0")
                        ])
                    ], className="shadow-sm")
                ], md=4),
                
                dbc.Col([
                    html.Div([
                        html.P([html.Strong("Interpretation:")]),
                        html.Ul([
                            html.Li([html.Strong("0.65+: "), "High confidence - Strong setup"]),
                            html.Li([html.Strong("0.55-0.65: "), "Good confidence - Solid setup"]),
                            html.Li([html.Strong("0.45-0.55: "), "Neutral - Coin flip"]),
                            html.Li([html.Strong("< 0.45: "), "Low confidence - Avoid"]),
                        ], className="mb-0")
                    ])
                ], md=8),
            ]),
            
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                html.Strong("Key Point: "), 
                "A 0.70 probability means if you take 100 similar setups, ~70 will be profitable. ",
                "This is NOT a guarantee for any single trade!"
            ], color="info", className="mt-3"),
        ], className="mb-4"),
        
        html.Hr(),
        
        # R_multiple Distribution
        html.Div([
            html.H6([
                html.I(className="bi bi-bar-chart me-2"),
                "R_multiple Distribution (P10/P50/P90)"
            ], className="mb-3"),
            
            html.P([
                "These values show the ", html.Strong("range of possible outcomes"), " in R units ",
                "(where 1R = your initial risk). They represent the 10th, 50th, and 90th percentiles."
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.Div("P10: -0.3R", className="text-danger mb-2"),
                                html.Div("P50: 1.2R", className="text-primary mb-2 fw-bold fs-5"),
                                html.Div("P90: 2.8R", className="text-success"),
                            ], className="text-center")
                        ])
                    ], className="shadow-sm")
                ], md=4),
                
                dbc.Col([
                    html.Div([
                        html.P([html.Strong("What Each Means:")]),
                        html.Ul([
                            html.Li([
                                html.Strong("P10 (-0.3R): "), 
                                "Pessimistic case - 10% of outcomes are worse than this"
                            ]),
                            html.Li([
                                html.Strong("P50 (1.2R): "), 
                                "Expected/typical outcome - median result"
                            ]),
                            html.Li([
                                html.Strong("P90 (2.8R): "), 
                                "Optimistic case - 10% of outcomes are better than this"
                            ]),
                        ], className="mb-0")
                    ])
                ], md=8),
            ]),
            
            dbc.Alert([
                html.I(className="bi bi-lightbulb me-2"),
                html.Strong("Trading Insight: "), 
                "The P10-P90 range shows your risk-reward profile. ",
                "A wider range means more uncertainty. Positive skew (P90-P50 > P50-P10) indicates upside potential."
            ], color="success", className="mt-3"),
        ], className="mb-4"),
        
        html.Hr(),
        
        # Conformal Intervals
        html.Div([
            html.H6([
                html.I(className="bi bi-arrows-expand me-2"),
                "Conformal Intervals (P10_conf / P90_conf)"
            ], className="mb-3"),
            
            html.P([
                "These are ", html.Strong("adjusted intervals"), " that provide a ",
                html.Strong("statistical guarantee"), ": approximately 90% of actual outcomes ",
                "will fall within this range."
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.Div("Raw: [-0.3R, 2.8R]", className="text-muted mb-2"),
                                html.Div("Adjusted: [-0.5R, 3.1R]", className="text-primary fw-bold fs-5"),
                                html.Div("Width: 3.6R", className="text-muted mt-2"),
                            ], className="text-center")
                        ])
                    ], className="shadow-sm")
                ], md=4),
                
                dbc.Col([
                    html.Div([
                        html.P([html.Strong("Why Adjusted?")]),
                        html.Ul([
                            html.Li("Raw quantile predictions may be overconfident"),
                            html.Li("Conformal prediction adds margin based on calibration errors"),
                            html.Li("Ensures honest uncertainty quantification"),
                            html.Li("Wider intervals = more uncertainty, narrower = more confidence"),
                        ], className="mb-0")
                    ])
                ], md=8),
            ]),
            
            dbc.Alert([
                html.I(className="bi bi-shield-check me-2"),
                html.Strong("Coverage Guarantee: "), 
                "If the system says 90% coverage, then ~90% of your actual trade outcomes ",
                "will fall within the predicted interval. This is mathematically guaranteed!"
            ], color="primary", className="mt-3"),
        ], className="mb-4"),
        
        html.Hr(),
        
        # Setup Quality Categories
        html.Div([
            html.H6([
                html.I(className="bi bi-award me-2"),
                "Setup Quality Categories (A+/A/B/C)"
            ], className="mb-3"),
            
            html.P([
                "Setups are categorized based on ", html.Strong("both probability and expected return"), ". ",
                "This helps you quickly identify high-quality opportunities."
            ]),
            
            dbc.Row([
                # A+ Category
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("A+", className="mb-0 text-white")
                        ], style={'backgroundColor': '#006400'}),
                        dbc.CardBody([
                            html.P([html.Strong("Excellent Setup")], className="mb-2"),
                            html.Ul([
                                html.Li("Prob Win > 0.65"),
                                html.Li("Expected R > 1.5"),
                            ], className="mb-2"),
                            dbc.Badge("TRADE", color="success", className="w-100")
                        ])
                    ], className="shadow-sm mb-3")
                ], md=6),
                
                # A Category
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("A", className="mb-0 text-white")
                        ], style={'backgroundColor': '#32CD32'}),
                        dbc.CardBody([
                            html.P([html.Strong("Good Setup")], className="mb-2"),
                            html.Ul([
                                html.Li("Prob Win > 0.55"),
                                html.Li("Expected R > 1.0"),
                            ], className="mb-2"),
                            dbc.Badge("TRADE", color="success", className="w-100")
                        ])
                    ], className="shadow-sm mb-3")
                ], md=6),
            ]),
            
            dbc.Row([
                # B Category
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("B", className="mb-0 text-dark")
                        ], style={'backgroundColor': '#FFD700'}),
                        dbc.CardBody([
                            html.P([html.Strong("Fair Setup")], className="mb-2"),
                            html.Ul([
                                html.Li("Prob Win > 0.45"),
                                html.Li("Expected R > 0.5"),
                            ], className="mb-2"),
                            dbc.Badge("SKIP", color="warning", className="w-100")
                        ])
                    ], className="shadow-sm mb-3")
                ], md=6),
                
                # C Category
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("C", className="mb-0 text-white")
                        ], style={'backgroundColor': '#DC143C'}),
                        dbc.CardBody([
                            html.P([html.Strong("Poor Setup")], className="mb-2"),
                            html.Ul([
                                html.Li("Below B thresholds"),
                                html.Li("Low probability or negative expected R"),
                            ], className="mb-2"),
                            dbc.Badge("SKIP", color="danger", className="w-100")
                        ])
                    ], className="shadow-sm mb-3")
                ], md=6),
            ]),
            
            dbc.Alert([
                html.I(className="bi bi-gear me-2"),
                html.Strong("Customizable: "), 
                "You can adjust these thresholds in Settings to match your risk tolerance. ",
                "Conservative traders might only trade A+ setups."
            ], color="secondary", className="mt-3"),
        ], className="mb-4"),
    ])


def create_faq_tab_content():
    """Create FAQ tab with common questions and answers"""
    return html.Div([
        html.H5([
            html.I(className="bi bi-question-circle me-2"),
            "Frequently Asked Questions"
        ], className="mb-4"),
        
        # FAQ 1
        dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="bi bi-1-circle me-2"),
                    "What is a 'calibrated probability'?"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.P([
                    "A calibrated probability is one that matches empirical frequencies. ",
                    "If the model predicts 0.70 for 100 setups, approximately 70 of them should be profitable. ",
                    "Raw ML models often output overconfident or underconfident probabilities. ",
                    "Isotonic calibration corrects this to make predictions more trustworthy."
                ])
            ])
        ], className="mb-3"),
        
        # FAQ 2
        dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="bi bi-2-circle me-2"),
                    "How do I interpret P10, P50, P90?"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.P([
                    html.Strong("P10: "), "10% of outcomes are worse than this (pessimistic scenario)"
                ]),
                html.P([
                    html.Strong("P50: "), "50% of outcomes are worse/better (median/expected outcome)"
                ]),
                html.P([
                    html.Strong("P90: "), "90% of outcomes are worse than this (optimistic scenario)"
                ]),
                html.P([
                    "Example: P10=-0.5R, P50=1.2R, P90=3.0R means you can expect to lose up to 0.5R in bad cases, ",
                    "gain 1.2R typically, and gain up to 3.0R in good cases."
                ], className="mb-0")
            ])
        ], className="mb-3"),
        
        # FAQ 3
        dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="bi bi-3-circle me-2"),
                    "What does 'coverage' mean?"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.P([
                    "Coverage is the percentage of actual outcomes that fall within the predicted interval. ",
                    "A 90% coverage target means that 90% of your actual trade results should fall between P10_conf and P90_conf. ",
                    "This is a statistical guarantee provided by conformal prediction."
                ], className="mb-0")
            ])
        ], className="mb-3"),
        
        # FAQ 4
        dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="bi bi-4-circle me-2"),
                    "When should I retrain the models?"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.P("Retrain models when:"),
                html.Ul([
                    html.Li("You have significant new data (e.g., 1000+ new trades)"),
                    html.Li("Model performance degrades by >10% (check Performance Monitoring)"),
                    html.Li("Market conditions change significantly (new regime)"),
                    html.Li("You want to add/remove features"),
                    html.Li("Every 1-3 months as a best practice"),
                ], className="mb-0")
            ])
        ], className="mb-3"),
        
        # FAQ 5
        dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="bi bi-5-circle me-2"),
                    "How many features should I use?"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.P([
                    "Recommended: 5-8 features. Use Auto Feature Selection to identify the most important ones. ",
                    "Too few features (<5) may miss important patterns. ",
                    "Too many features (>15) can lead to overfitting and slower predictions."
                ], className="mb-0")
            ])
        ], className="mb-3"),
        
        # FAQ 6
        dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="bi bi-6-circle me-2"),
                    "What's a good AUC score?"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.Ul([
                    html.Li([html.Strong("0.50: "), "Random guessing (no predictive power)"]),
                    html.Li([html.Strong("0.55-0.60: "), "Weak but useful signal"]),
                    html.Li([html.Strong("0.60-0.70: "), "Good predictive power"]),
                    html.Li([html.Strong("0.70-0.80: "), "Strong predictive power"]),
                    html.Li([html.Strong(">0.80: "), "Excellent (but verify not overfitting)"]),
                ], className="mb-0")
            ])
        ], className="mb-3"),
        
        # FAQ 7
        dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="bi bi-7-circle me-2"),
                    "What's a good Brier score?"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.P([
                    "Brier score measures probability prediction accuracy (0-1, lower is better):"
                ]),
                html.Ul([
                    html.Li([html.Strong("< 0.15: "), "Excellent calibration"]),
                    html.Li([html.Strong("0.15-0.20: "), "Good calibration"]),
                    html.Li([html.Strong("0.20-0.25: "), "Acceptable calibration"]),
                    html.Li([html.Strong("> 0.25: "), "Poor calibration (consider retraining)"]),
                ], className="mb-0")
            ])
        ], className="mb-3"),
        
        # FAQ 8
        dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="bi bi-8-circle me-2"),
                    "Should I always follow the TRADE/SKIP recommendation?"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.P([
                    "No. The recommendation is a guideline based on statistical analysis. ",
                    "You should also consider:",
                ]),
                html.Ul([
                    html.Li("Your personal risk tolerance"),
                    html.Li("Current market conditions"),
                    html.Li("Portfolio diversification"),
                    html.Li("Other fundamental/technical factors"),
                    html.Li("Your trading plan and rules"),
                ], className="mb-2"),
                html.P([
                    "Use ML predictions as ", html.Strong("one input"), " in your decision-making process, ",
                    "not the sole determinant."
                ], className="mb-0 fst-italic")
            ])
        ], className="mb-3"),
        
        # FAQ 9
        dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="bi bi-9-circle me-2"),
                    "Why are conformal intervals wider than quantile predictions?"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.P([
                    "Conformal prediction adds a margin to account for model uncertainty. ",
                    "Quantile models (P10/P50/P90) give point estimates but may be overconfident. ",
                    "Conformal prediction adjusts these intervals to provide a statistical guarantee ",
                    "that ~90% of actual outcomes will fall within the interval. ",
                    "Wider intervals = more honest about uncertainty."
                ], className="mb-0")
            ])
        ], className="mb-3"),
        
        # FAQ 10
        dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="bi bi-dash-circle me-2"),
                    "Can I use this for live trading?"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.P([
                    "Yes, but with caution:"
                ]),
                html.Ul([
                    html.Li("Backtest thoroughly on out-of-sample data first"),
                    html.Li("Start with small position sizes"),
                    html.Li("Monitor model performance continuously"),
                    html.Li("Have a plan for when predictions are wrong"),
                    html.Li("Never risk more than you can afford to lose"),
                ], className="mb-2"),
                dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    html.Strong("Disclaimer: "), 
                    "Past performance does not guarantee future results. ",
                    "ML predictions are probabilistic, not deterministic. ",
                    "Always use proper risk management."
                ], color="warning", className="mb-0")
            ])
        ], className="mb-3"),
        
        # FAQ 11
        dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="bi bi-dash-circle me-2"),
                    "What if my predictions seem inaccurate?"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.P("Troubleshooting steps:"),
                html.Ol([
                    html.Li("Check if you're using the right features (use Auto Feature Selection)"),
                    html.Li("Verify data quality (no missing values, outliers handled)"),
                    html.Li("Ensure sufficient training data (minimum 1000 samples)"),
                    html.Li("Check for data leakage (future information in features)"),
                    html.Li("Review training metrics (AUC, Brier, Coverage)"),
                    html.Li("Consider market regime changes (retrain with recent data)"),
                    html.Li("Adjust hyperparameters in Settings"),
                ], className="mb-0")
            ])
        ], className="mb-3"),
        
        # FAQ 12
        dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="bi bi-dash-circle me-2"),
                    "How is this different from traditional technical analysis?"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.P([
                    html.Strong("Traditional TA: "), "Rule-based, subjective interpretation, no probability estimates"
                ]),
                html.P([
                    html.Strong("ML Prediction Engine: "), "Data-driven, objective, provides calibrated probabilities ",
                    "and uncertainty intervals"
                ]),
                html.P([
                    "Best approach: Combine both. Use TA for setup identification, ",
                    "ML for probability assessment and risk-reward estimation."
                ], className="mb-0 fst-italic")
            ])
        ], className="mb-3"),
    ])



def get_help_content(active_tab):
    """
    Get content for the active help tab
    
    Parameters
    ----------
    active_tab : str
        Active tab ID (help-overview, help-howto, help-interpretation, help-faq)
    
    Returns
    -------
    dash component
        Content for the active tab
    """
    if active_tab == 'help-overview':
        return create_overview_tab_content()
    elif active_tab == 'help-howto':
        return create_how_to_use_tab_content()
    elif active_tab == 'help-interpretation':
        return create_interpretation_guide_tab_content()
    elif active_tab == 'help-faq':
        return create_faq_tab_content()
    else:
        return html.Div("Select a tab to view help content", className="text-muted")
