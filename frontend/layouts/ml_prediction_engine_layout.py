"""
ML Prediction Engine Layout

Machine learning-based prediction engine for trading decisions with:
1. LightGBM Classifier - Binary win/loss prediction
2. Isotonic Calibration - Probability calibration
3. Quantile Regression - R_multiple distribution (P10/P50/P90)
4. Conformal Prediction - Interval prediction with coverage guarantee
"""

import dash_bootstrap_components as dbc
from dash import html, dcc


def create_ml_prediction_engine_layout():
    """Create layout for ML Prediction Engine (Page 8)"""
    
    return dbc.Container([
        # Header Section
        dbc.Row([
            dbc.Col([
                html.H3([
                    html.I(className="bi bi-robot me-2"),
                    "ML Prediction Engine"
                ], className="mb-2"),
                html.P(
                    "Mesin prediksi berbasis machine learning untuk keputusan trading dengan probabilitas terkalibrasi",
                    className="text-muted"
                )
            ], md=8),
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button([
                        html.I(className="bi bi-arrow-left me-2"),
                        "Back to Dashboard"
                    ], id="ml-back-to-dashboard-btn", color="secondary", size="sm"),
                    dbc.Button([
                        html.I(className="bi bi-gear me-2"),
                        "Settings"
                    ], id="ml-settings-btn", color="info", size="sm"),
                    dbc.Button([
                        html.I(className="bi bi-question-circle me-2"),
                        "Help"
                    ], id="ml-help-btn", color="primary", size="sm"),
                ], className="float-end")
            ], md=4)
        ], className="mb-4"),
        
        # Model Status Alert
        dbc.Row([
            dbc.Col([
                html.Div(id='ml-model-status-alert')
            ])
        ], className="mb-3"),
        
        # Feature Configuration Info
        dbc.Row([
            dbc.Col([
                html.Div(id='ml-feature-config-info')
            ])
        ], className="mb-3"),
        
        # Section 1: Input Controls
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="bi bi-sliders me-2"),
                            "Input Controls"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Prediction Mode", className="fw-bold"),
                                dcc.Dropdown(
                                    id='ml-prediction-mode',
                                    options=[
                                        {'label': '🎯 Single Prediction', 'value': 'single'},
                                        {'label': '📊 Batch Prediction', 'value': 'batch'}
                                    ],
                                    value='single',
                                    clearable=False
                                ),
                            ], md=4),
                            dbc.Col([
                                html.Label("Data Source", className="fw-bold"),
                                dcc.Dropdown(
                                    id='ml-data-source',
                                    options=[
                                        {'label': 'Use Merged Data', 'value': 'merged'},
                                        {'label': 'Upload CSV', 'value': 'upload'}
                                    ],
                                    value='merged',
                                    clearable=False
                                ),
                            ], md=4),
                            dbc.Col([
                                html.Label("Actions", className="fw-bold"),
                                dbc.ButtonGroup([
                                    dbc.Button([
                                        html.I(className="bi bi-play-circle me-2"),
                                        "Run Prediction"
                                    ], id="ml-run-prediction-btn", color="primary", className="me-2"),
                                    dbc.Button([
                                        html.I(className="bi bi-arrow-repeat me-2"),
                                        "Train Models"
                                    ], id="ml-train-models-btn", color="success"),
                                ], className="w-100")
                            ], md=4)
                        ], className="mb-3"),
                        
                        # Upload area (hidden by default)
                        html.Div([
                            dcc.Upload(
                                id='ml-upload-data',
                                children=html.Div([
                                    html.I(className="bi bi-cloud-upload me-2"),
                                    'Drag and Drop or ',
                                    html.A('Select CSV File')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '2px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px 0'
                                },
                                multiple=False
                            ),
                        ], id='ml-upload-container', style={'display': 'none'})
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Section 2: Prediction Summary Cards
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5([
                        html.I(className="bi bi-card-checklist me-2"),
                        "Prediction Summary"
                    ], className="mb-3 d-inline-block"),
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className="bi bi-lightbulb me-2"),
                            "Add ML Prediction Scenario"
                        ], id="ml-add-whatif-scenario-btn", color="warning", size="sm", outline=True,
                           title="Send predictions to What-If Scenarios"),
                        dbc.Button([
                            html.I(className="bi bi-download me-2"),
                            "Export CSV"
                        ], id="ml-export-predictions-btn", color="primary", size="sm", outline=True),
                        dbc.Button([
                            html.I(className="bi bi-file-pdf me-2"),
                            "Export Report"
                        ], id="ml-export-report-btn", color="secondary", size="sm", outline=True),
                        dbc.Button([
                            html.I(className="bi bi-trash me-2"),
                            "Clear"
                        ], id="ml-clear-predictions-btn", color="danger", size="sm", outline=True,
                           title="Clear all predictions from cache"),
                    ], className="float-end")
                ], className="clearfix mb-3"),
                html.Div(id='ml-prediction-summary-cards')
            ])
        ], className="mb-4"),
        
        # Section 3: Probability Analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="bi bi-graph-up me-2"),
                            "Probability Analysis"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(label="📊 Reliability Diagram", tab_id="reliability"),
                            dbc.Tab(label="📈 Probability Distribution", tab_id="prob-dist"),
                            dbc.Tab(label="🎯 Calibration Metrics", tab_id="calib-metrics"),
                        ], id="ml-probability-tabs", active_tab="reliability"),
                        
                        html.Div(id="ml-probability-analysis-content", className="mt-4")
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Section 4: Distribution Analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="bi bi-bar-chart me-2"),
                            "Distribution Analysis"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(label="📊 R_multiple Fan Chart", tab_id="fan-chart"),
                            dbc.Tab(label="📈 Distribution Comparison", tab_id="dist-comp"),
                            dbc.Tab(label="🎯 Coverage Analysis", tab_id="coverage"),
                        ], id="ml-distribution-tabs", active_tab="fan-chart"),
                        
                        html.Div(id="ml-distribution-analysis-content", className="mt-4")
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Data Stores
        dcc.Store(id='ml-prediction-results-store'),
        dcc.Store(id='ml-model-metadata-store'),
        dcc.Store(id='ml-training-status-store'),
        dcc.Store(id='ml-data-sharing-status'),  # Track data sharing across pages
        
        # Download components
        dcc.Download(id="ml-download-predictions"),
        dcc.Download(id="ml-download-report"),
        
        # Settings Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle([
                html.I(className="bi bi-gear me-2"),
                "ML Prediction Settings"
            ])),
            dbc.ModalBody([
                dbc.Tabs([
                    dbc.Tab(label="🎯 Features", tab_id="settings-features"),
                    dbc.Tab(label="⚙️ Model Hyperparameters", tab_id="settings-hyperparams"),
                    dbc.Tab(label="📊 Thresholds", tab_id="settings-thresholds"),
                    dbc.Tab(label="🎨 Display", tab_id="settings-display"),
                ], id="ml-settings-tabs", active_tab="settings-features"),
                
                html.Div(id="ml-settings-content", className="mt-3", style={"maxHeight": "60vh", "overflowY": "auto"})
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="ml-settings-cancel-btn", color="secondary", className="me-2"),
                dbc.Button([
                    html.I(className="bi bi-save me-2"),
                    "Save Settings"
                ], id="ml-settings-save-btn", color="primary")
            ])
        ], id="ml-settings-modal", size="xl", is_open=False),
        
        # Help Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle([
                html.I(className="bi bi-question-circle me-2"),
                "ML Prediction Engine Help"
            ])),
            dbc.ModalBody([
                dbc.Tabs([
                    dbc.Tab(label="📖 Overview", tab_id="help-overview"),
                    dbc.Tab(label="🚀 How to Use", tab_id="help-usage"),
                    dbc.Tab(label="📊 Interpretation Guide", tab_id="help-interpretation"),
                    dbc.Tab(label="❓ FAQ", tab_id="help-faq"),
                ], id="ml-help-tabs", active_tab="help-overview"),
                
                html.Div(id="ml-help-content", className="mt-3", style={"maxHeight": "60vh", "overflowY": "auto"})
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id="ml-help-close-btn", color="secondary")
            ])
        ], id="ml-help-modal", size="xl", is_open=False),
        
        # Training Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle([
                html.I(className="bi bi-arrow-repeat me-2"),
                "Train ML Models"
            ])),
            dbc.ModalBody([
                html.Div(id="ml-training-modal-content")
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="ml-training-cancel-btn", color="secondary", className="me-2"),
                dbc.Button([
                    html.I(className="bi bi-play-circle me-2"),
                    "Start Training"
                ], id="ml-training-start-btn", color="success")
            ])
        ], id="ml-training-modal", size="lg", is_open=False),
        
    ], fluid=True)
