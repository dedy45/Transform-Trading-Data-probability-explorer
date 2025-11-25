"""
Training Modal Component

Modal interface for training ML models with:
1. Configuration form (features, train_size, cv_folds)
2. Progress bar with status per component
3. Training metrics display after completion
4. Warning modal for overwriting existing models
5. Error handling and rollback on failure

**Feature: ml-prediction-engine**
**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go


def create_training_config_form(
    available_features=None,
    selected_features=None,
    train_ratio=0.60,
    calib_ratio=0.20,
    cv_folds=5,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5
):
    """
    Create training configuration form.
    
    Parameters
    ----------
    available_features : list, optional
        List of available feature names
    selected_features : list, optional
        List of currently selected features
    train_ratio : float, default=0.60
        Training set ratio
    calib_ratio : float, default=0.20
        Calibration set ratio
    cv_folds : int, default=5
        Number of cross-validation folds
    n_estimators : int, default=100
        Number of estimators for LightGBM
    learning_rate : float, default=0.05
        Learning rate for LightGBM
    max_depth : int, default=5
        Max depth for LightGBM
    
    Returns
    -------
    dash_bootstrap_components.Form
        Configuration form
    """
    if available_features is None:
        available_features = []
    
    if selected_features is None:
        selected_features = available_features[:8] if len(available_features) >= 8 else available_features
    
    test_ratio = 1.0 - train_ratio - calib_ratio
    
    return dbc.Form([
        # Feature Selection Section
        html.H6([
            html.I(className="bi bi-list-check me-2"),
            "Feature Selection"
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Label("Select Features (5-15)", html_for="training-feature-select"),
                dcc.Dropdown(
                    id='training-feature-select',
                    options=[{'label': f, 'value': f} for f in available_features],
                    value=selected_features,
                    multi=True,
                    placeholder="Select 5-15 features...",
                    className="mb-3"
                ),
                html.Small(
                    f"Selected: {len(selected_features)} features (min: 5, max: 15)",
                    className="text-muted",
                    id="training-feature-count"
                )
            ], md=12)
        ]),
        
        html.Hr(className="my-4"),
        
        # Data Split Section
        html.H6([
            html.I(className="bi bi-pie-chart me-2"),
            "Data Split Configuration"
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Label("Train Ratio", html_for="training-train-ratio"),
                dbc.Input(
                    id='training-train-ratio',
                    type='number',
                    min=0.4,
                    max=0.8,
                    step=0.05,
                    value=train_ratio,
                    className="mb-2"
                ),
                html.Small(f"{train_ratio*100:.0f}% for training", className="text-muted")
            ], md=4),
            
            dbc.Col([
                dbc.Label("Calibration Ratio", html_for="training-calib-ratio"),
                dbc.Input(
                    id='training-calib-ratio',
                    type='number',
                    min=0.1,
                    max=0.4,
                    step=0.05,
                    value=calib_ratio,
                    className="mb-2"
                ),
                html.Small(f"{calib_ratio*100:.0f}% for calibration", className="text-muted")
            ], md=4),
            
            dbc.Col([
                dbc.Label("Test Ratio (Auto)", html_for="training-test-ratio"),
                dbc.Input(
                    id='training-test-ratio',
                    type='number',
                    value=test_ratio,
                    disabled=True,
                    className="mb-2"
                ),
                html.Small(f"{test_ratio*100:.0f}% for testing", className="text-muted")
            ], md=4)
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Label("Cross-Validation Folds", html_for="training-cv-folds"),
                dbc.Input(
                    id='training-cv-folds',
                    type='number',
                    min=3,
                    max=10,
                    step=1,
                    value=cv_folds,
                    className="mb-2"
                ),
                html.Small(f"{cv_folds}-fold time-series CV", className="text-muted")
            ], md=12)
        ]),
        
        html.Hr(className="my-4"),
        
        # Model Hyperparameters Section
        html.H6([
            html.I(className="bi bi-sliders me-2"),
            "Model Hyperparameters"
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Label("Number of Estimators", html_for="training-n-estimators"),
                dbc.Input(
                    id='training-n-estimators',
                    type='number',
                    min=50,
                    max=500,
                    step=50,
                    value=n_estimators,
                    className="mb-2"
                ),
                html.Small("More trees = better fit but slower", className="text-muted")
            ], md=4),
            
            dbc.Col([
                dbc.Label("Learning Rate", html_for="training-learning-rate"),
                dbc.Input(
                    id='training-learning-rate',
                    type='number',
                    min=0.01,
                    max=0.3,
                    step=0.01,
                    value=learning_rate,
                    className="mb-2"
                ),
                html.Small("Lower = more conservative", className="text-muted")
            ], md=4),
            
            dbc.Col([
                dbc.Label("Max Depth", html_for="training-max-depth"),
                dbc.Input(
                    id='training-max-depth',
                    type='number',
                    min=3,
                    max=10,
                    step=1,
                    value=max_depth,
                    className="mb-2"
                ),
                html.Small("Deeper = more complex", className="text-muted")
            ], md=4)
        ]),
        
        html.Hr(className="my-4"),
        
        # Info Alert
        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            html.Strong("Training Info: "),
            "Training will take 2-5 minutes depending on data size. ",
            "All 4 components (Classifier, Calibration, Quantile, Conformal) will be trained sequentially."
        ], color="info", className="mb-0")
    ])


def create_progress_display(
    current_component=None,
    current_step=None,
    progress_percentage=0.0,
    component_status=None
):
    """
    Create progress display with status per component.
    
    Parameters
    ----------
    current_component : str, optional
        Currently training component
    current_step : str, optional
        Current step description
    progress_percentage : float, default=0.0
        Overall progress (0-100)
    component_status : dict, optional
        Status for each component {'preprocessing': 'pending', ...}
    
    Returns
    -------
    html.Div
        Progress display
    """
    if component_status is None:
        component_status = {
            'preprocessing': 'pending',
            'classifier': 'pending',
            'calibration': 'pending',
            'quantile': 'pending',
            'conformal': 'pending'
        }
    
    # Component display names and icons
    component_info = {
        'preprocessing': {'name': 'Data Preprocessing', 'icon': 'bi-funnel'},
        'classifier': {'name': 'LightGBM Classifier', 'icon': 'bi-diagram-3'},
        'calibration': {'name': 'Isotonic Calibration', 'icon': 'bi-speedometer2'},
        'quantile': {'name': 'Quantile Regression', 'icon': 'bi-bar-chart-steps'},
        'conformal': {'name': 'Conformal Prediction', 'icon': 'bi-shield-check'}
    }
    
    # Status colors and icons
    status_config = {
        'pending': {'color': 'secondary', 'icon': 'bi-circle', 'text': 'Pending'},
        'in_progress': {'color': 'primary', 'icon': 'bi-arrow-repeat', 'text': 'In Progress'},
        'completed': {'color': 'success', 'icon': 'bi-check-circle-fill', 'text': 'Completed'},
        'failed': {'color': 'danger', 'icon': 'bi-x-circle-fill', 'text': 'Failed'}
    }
    
    # Create component status list
    component_items = []
    for comp_key, comp_data in component_info.items():
        status = component_status.get(comp_key, 'pending')
        status_cfg = status_config[status]
        
        is_current = comp_key == current_component
        
        component_items.append(
            dbc.ListGroupItem([
                dbc.Row([
                    dbc.Col([
                        html.I(className=f"{comp_data['icon']} me-2"),
                        html.Strong(comp_data['name']) if is_current else comp_data['name']
                    ], width=8),
                    dbc.Col([
                        dbc.Badge([
                            html.I(className=f"{status_cfg['icon']} me-1"),
                            status_cfg['text']
                        ], color=status_cfg['color'])
                    ], width=4, className="text-end")
                ], align="center")
            ], active=is_current, color="light" if is_current else None)
        )
    
    return html.Div([
        # Overall Progress Bar
        html.Div([
            html.H6("Overall Progress", className="mb-2"),
            dbc.Progress(
                value=progress_percentage,
                label=f"{progress_percentage:.0f}%",
                color="success" if progress_percentage == 100 else "primary",
                striped=progress_percentage < 100,
                animated=progress_percentage < 100,
                className="mb-3",
                style={"height": "30px"}
            )
        ]),
        
        # Current Step
        html.Div([
            html.H6("Current Step", className="mb-2"),
            dbc.Alert(
                current_step or "Waiting to start...",
                color="info",
                className="mb-3"
            )
        ]) if current_step else None,
        
        # Component Status List
        html.Div([
            html.H6("Component Status", className="mb-2"),
            dbc.ListGroup(component_items, flush=True)
        ])
    ])


def create_metrics_display(metrics=None):
    """
    Create training metrics display.
    
    Parameters
    ----------
    metrics : dict, optional
        Training metrics from all components
    
    Returns
    -------
    html.Div
        Metrics display
    """
    if metrics is None:
        return html.Div([
            html.I(className="bi bi-hourglass-split fs-1 text-muted"),
            html.P("Training not completed yet", className="text-muted mt-2")
        ], className="text-center py-5")
    
    # Extract metrics
    classifier_metrics = metrics.get('classifier', {})
    calibration_metrics = metrics.get('calibration', {})
    quantile_metrics = metrics.get('quantile', {})
    conformal_metrics = metrics.get('conformal', {})
    metadata = metrics.get('metadata', {})
    
    return html.Div([
        # Success Alert
        dbc.Alert([
            html.I(className="bi bi-check-circle-fill me-2"),
            html.Strong("Training Completed Successfully! "),
            f"Trained on {metadata.get('n_train', 0)} samples with {metadata.get('n_features', 0)} features."
        ], color="success", className="mb-4"),
        
        # Metrics Cards
        dbc.Row([
            # Classifier Metrics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-diagram-3 me-2"),
                        "Classifier Performance"
                    ]),
                    dbc.CardBody([
                        html.Div([
                            html.H4(f"{classifier_metrics.get('auc_val', classifier_metrics.get('auc_train', 0)):.4f}", className="mb-1"),
                            html.Small("AUC Score", className="text-muted")
                        ], className="mb-3"),
                        html.Div([
                            html.H5(f"{classifier_metrics.get('brier_score_val', classifier_metrics.get('brier_score_train', 0)):.4f}", className="mb-1"),
                            html.Small("Brier Score", className="text-muted")
                        ]),
                        html.Hr(),
                        html.Small([
                            html.I(className="bi bi-info-circle me-1"),
                            "AUC > 0.55 = better than random"
                        ], className="text-muted")
                    ])
                ], className="h-100")
            ], md=6, lg=3, className="mb-3"),
            
            # Calibration Metrics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-speedometer2 me-2"),
                        "Calibration Quality"
                    ]),
                    dbc.CardBody([
                        html.Div([
                            html.H4(f"{calibration_metrics.get('brier_improvement', 0):.4f}", className="mb-1"),
                            html.Small("Brier Improvement", className="text-muted")
                        ], className="mb-3"),
                        html.Div([
                            html.H5(f"{calibration_metrics.get('ece_after', 0):.4f}", className="mb-1"),
                            html.Small("ECE After", className="text-muted")
                        ]),
                        html.Hr(),
                        html.Small([
                            html.I(className="bi bi-info-circle me-1"),
                            "ECE < 0.05 = well calibrated"
                        ], className="text-muted")
                    ])
                ], className="h-100")
            ], md=6, lg=3, className="mb-3"),
            
            # Quantile Metrics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-bar-chart-steps me-2"),
                        "Quantile Accuracy"
                    ]),
                    dbc.CardBody([
                        html.Div([
                            html.H4(f"{quantile_metrics.get('mae_p50_val', quantile_metrics.get('mae_p50_train', 0)):.4f}", className="mb-1"),
                            html.Small("MAE P50", className="text-muted")
                        ], className="mb-3"),
                        html.Div([
                            html.Small(f"P10: {quantile_metrics.get('mae_p10_val', quantile_metrics.get('mae_p10_train', 0)):.4f}", className="d-block"),
                            html.Small(f"P90: {quantile_metrics.get('mae_p90_val', quantile_metrics.get('mae_p90_train', 0)):.4f}", className="d-block")
                        ]),
                        html.Hr(),
                        html.Small([
                            html.I(className="bi bi-info-circle me-1"),
                            "Lower MAE = better accuracy"
                        ], className="text-muted")
                    ])
                ], className="h-100")
            ], md=6, lg=3, className="mb-3"),
            
            # Conformal Metrics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-shield-check me-2"),
                        "Conformal Coverage"
                    ]),
                    dbc.CardBody([
                        html.Div([
                            html.H4(f"{conformal_metrics.get('actual_coverage_calib', 0)*100:.1f}%", className="mb-1"),
                            html.Small("Actual Coverage", className="text-muted")
                        ], className="mb-3"),
                        html.Div([
                            html.H5(f"{conformal_metrics.get('target_coverage', 0.9)*100:.0f}%", className="mb-1"),
                            html.Small("Target Coverage", className="text-muted")
                        ]),
                        html.Hr(),
                        html.Small([
                            html.I(className="bi bi-info-circle me-1"),
                            "Should be close to target"
                        ], className="text-muted")
                    ])
                ], className="h-100")
            ], md=6, lg=3, className="mb-3")
        ]),
        
        # Data Split Info
        dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-pie-chart me-2"),
                "Data Split Information"
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H5(f"{metadata.get('n_train', 0):,}", className="mb-1"),
                            html.Small("Training Samples", className="text-muted")
                        ], className="text-center")
                    ], md=3),
                    dbc.Col([
                        html.Div([
                            html.H5(f"{metadata.get('n_calib', 0):,}", className="mb-1"),
                            html.Small("Calibration Samples", className="text-muted")
                        ], className="text-center")
                    ], md=3),
                    dbc.Col([
                        html.Div([
                            html.H5(f"{metadata.get('n_test', 0):,}", className="mb-1"),
                            html.Small("Test Samples", className="text-muted")
                        ], className="text-center")
                    ], md=3),
                    dbc.Col([
                        html.Div([
                            html.H5(f"{metadata.get('n_features', 0)}", className="mb-1"),
                            html.Small("Features Used", className="text-muted")
                        ], className="text-center")
                    ], md=3)
                ])
            ])
        ], className="mb-3"),
        
        # Feature List
        dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-list-check me-2"),
                "Features Used"
            ]),
            dbc.CardBody([
                html.Div([
                    dbc.Badge(feature, color="primary", className="me-2 mb-2")
                    for feature in metadata.get('feature_columns', [])
                ])
            ])
        ])
    ])


def create_training_modal(
    is_open=False,
    available_features=None,
    selected_features=None,
    models_exist=False,
    show_warning=False,
    show_progress=False,
    show_metrics=False,
    progress_data=None,
    metrics_data=None,
    error_message=None
):
    """
    Create complete training modal with all states.
    
    Parameters
    ----------
    is_open : bool, default=False
        Whether modal is open
    available_features : list, optional
        Available features
    selected_features : list, optional
        Selected features
    models_exist : bool, default=False
        Whether models already exist
    show_warning : bool, default=False
        Show overwrite warning
    show_progress : bool, default=False
        Show training progress
    show_metrics : bool, default=False
        Show training metrics
    progress_data : dict, optional
        Progress data from trainer
    metrics_data : dict, optional
        Metrics data from training
    error_message : str, optional
        Error message if training failed
    
    Returns
    -------
    dash_bootstrap_components.Modal
        Training modal
    """
    # Determine modal content based on state
    if error_message:
        # Error State
        modal_body = dbc.ModalBody([
            dbc.Alert([
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                html.Strong("Training Failed! "),
                html.Br(),
                html.Pre(error_message, className="mt-2 mb-0", style={"fontSize": "0.85rem"})
            ], color="danger")
        ])
        
        modal_footer = dbc.ModalFooter([
            dbc.Button("Close", id="training-modal-close-btn", color="secondary")
        ])
        
    elif show_metrics:
        # Metrics Display State
        modal_body = dbc.ModalBody([
            create_metrics_display(metrics_data)
        ], style={"maxHeight": "70vh", "overflowY": "auto"})
        
        modal_footer = dbc.ModalFooter([
            dbc.Button([
                html.I(className="bi bi-check-circle me-2"),
                "Done"
            ], id="training-modal-close-btn", color="success")
        ])
        
    elif show_progress:
        # Progress Display State
        if progress_data is None:
            progress_data = {
                'current_component': None,
                'current_step': None,
                'progress_percentage': 0.0,
                'component_status': {
                    'preprocessing': 'pending',
                    'classifier': 'pending',
                    'calibration': 'pending',
                    'quantile': 'pending',
                    'conformal': 'pending'
                }
            }
        
        modal_body = dbc.ModalBody([
            create_progress_display(
                current_component=progress_data.get('current_component'),
                current_step=progress_data.get('current_step'),
                progress_percentage=progress_data.get('progress_percentage', 0.0),
                component_status=progress_data.get('component_status')
            )
        ])
        
        modal_footer = dbc.ModalFooter([
            dbc.Spinner(
                html.Div("Training in progress...", className="text-muted"),
                color="primary",
                size="sm"
            )
        ])
        
    elif show_warning:
        # Warning State (Overwrite Confirmation)
        modal_body = dbc.ModalBody([
            dbc.Alert([
                html.I(className="bi bi-exclamation-triangle-fill me-2 fs-3"),
                html.Div([
                    html.H5("Overwrite Existing Models?", className="mb-2"),
                    html.P([
                        "Trained models already exist. Starting new training will ",
                        html.Strong("overwrite the existing models"),
                        ". A backup will be created automatically."
                    ], className="mb-2"),
                    html.P([
                        "If training fails, the system will automatically rollback to the previous models."
                    ], className="mb-0 small text-muted")
                ])
            ], color="warning", className="d-flex align-items-start")
        ])
        
        modal_footer = dbc.ModalFooter([
            dbc.Button("Cancel", id="training-modal-cancel-btn", color="secondary", className="me-2"),
            dbc.Button([
                html.I(className="bi bi-arrow-repeat me-2"),
                "Continue Training"
            ], id="training-modal-confirm-btn", color="warning")
        ])
        
    else:
        # Configuration Form State
        modal_body = dbc.ModalBody([
            create_training_config_form(
                available_features=available_features,
                selected_features=selected_features
            )
        ], style={"maxHeight": "70vh", "overflowY": "auto"})
        
        modal_footer = dbc.ModalFooter([
            dbc.Button("Cancel", id="training-modal-cancel-btn", color="secondary", className="me-2"),
            dbc.Button([
                html.I(className="bi bi-play-circle me-2"),
                "Start Training"
            ], id="training-modal-start-btn", color="primary")
        ])
    
    return dbc.Modal([
        dbc.ModalHeader([
            html.I(className="bi bi-gear-fill me-2"),
            "Train ML Models"
        ], close_button=not show_progress),
        modal_body,
        modal_footer
    ], id="training-modal", is_open=is_open, size="xl", backdrop="static" if show_progress else True)


def create_training_button(disabled=False):
    """
    Create button to open training modal.
    
    Parameters
    ----------
    disabled : bool, default=False
        Whether button is disabled
    
    Returns
    -------
    dash_bootstrap_components.Button
        Training button
    """
    return dbc.Button([
        html.I(className="bi bi-gear-fill me-2"),
        "Train Models"
    ], id="open-training-modal-btn", color="primary", disabled=disabled, className="me-2")
