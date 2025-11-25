"""
Auto Feature Selection Layout

Advanced feature selection using multiple methods:
1. Boruta - Most accurate feature selection
2. SHAP - Feature contribution analysis
3. RFECV - Recursive feature elimination with CV
4. CatBoost - Feature importance
5. Random Forest + Permutation Importance
"""

import dash_bootstrap_components as dbc
from dash import html, dcc


def create_auto_feature_selection_layout():
    """Create layout for automatic feature selection analysis"""
    
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H3([
                    html.I(className="bi bi-cpu me-2"),
                    "Auto Feature Selection"
                ], className="mb-2"),
                html.P(
                    "Analisis otomatis untuk menemukan fitur terbaik menggunakan metode machine learning canggih",
                    className="text-muted"
                )
            ])
        ], className="mb-4"),
        
        # Data Info Panel
        dbc.Row([
            dbc.Col([
                html.Div(id='afs-data-info-panel')
            ])
        ], className="mb-3"),
        
        # Control Panel
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="bi bi-sliders me-2"),
                            "Kontrol Analisis"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Variabel Target", className="fw-bold"),
                                dcc.Dropdown(
                                    id='afs-target-variable',
                                    placeholder='Pilih variabel target (win/loss)',
                                    clearable=False
                                ),
                            ], md=4),
                            dbc.Col([
                                html.Label("Mode Analisis", className="fw-bold"),
                                dcc.Dropdown(
                                    id='afs-analysis-mode',
                                    options=[
                                        {'label': '⚡ Quick Analysis (30 detik)', 'value': 'quick'},
                                        {'label': '🎯 Deep Analysis (2-5 menit)', 'value': 'deep'}
                                    ],
                                    value='quick',
                                    clearable=False
                                ),
                                html.Small("Quick: RF + Permutation + SHAP | Deep: Boruta + RFECV + SHAP", 
                                          className="text-muted")
                            ], md=4),
                            dbc.Col([
                                html.Label("Jumlah Fitur Target", className="fw-bold"),
                                dcc.Slider(
                                    id='afs-n-features',
                                    min=5,
                                    max=20,
                                    step=1,
                                    value=8,
                                    marks={5: '5', 8: '8 ⭐', 10: '10', 15: '15', 20: '20'},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                html.Small("Rekomendasi: 5-8 fitur (optimal untuk trading)", 
                                          className="text-muted")
                            ], md=4)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Button([
                                    html.I(className="bi bi-play-circle me-2"),
                                    "Mulai Analisis"
                                ], id="afs-run-btn", color="primary", size="lg", className="w-100"),
                            ], md=3),
                            dbc.Col([
                                dbc.Button([
                                    html.I(className="bi bi-robot me-2"),
                                    "Use for ML"
                                ], id="afs-use-for-ml-btn", color="warning", size="lg", 
                                   className="w-100", disabled=True,
                                   title="Send selected features to ML Prediction Engine"),
                            ], md=3),
                            dbc.Col([
                                dbc.Button([
                                    html.I(className="bi bi-download me-2"),
                                    "Export Hasil"
                                ], id="afs-export-btn", color="success", size="lg", 
                                   className="w-100", disabled=True),
                            ], md=3),
                            dbc.Col([
                                dbc.Button([
                                    html.I(className="bi bi-book me-2"),
                                    "Panduan"
                                ], id="afs-guide-btn", color="info", size="lg", className="w-100"),
                            ], md=3)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Progress Indicator
        dbc.Row([
            dbc.Col([
                html.Div(id="afs-progress-container", children=[])
            ])
        ], className="mb-4"),
        
        # Results Tabs
        dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="📊 Ranking Fitur", tab_id="ranking"),
                    dbc.Tab(label="🎯 SHAP Analysis", tab_id="shap"),
                    dbc.Tab(label="⚡ Trading Rules", tab_id="trading-rules"),
                    dbc.Tab(label="📈 Perbandingan Metode", tab_id="comparison"),
                    dbc.Tab(label="🗑️ Fitur yang Dibuang", tab_id="rejected"),
                ], id="afs-result-tabs", active_tab="ranking"),
                
                html.Div(id="afs-result-content", className="mt-4")
            ])
        ]),
        
        # Data stores
        dcc.Store(id='afs-results-store'),
        dcc.Download(id="afs-download-data"),
        
        # Toast notification for ML integration
        html.Div(id='afs-ml-integration-toast-container'),
        
        # Guide Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle([
                html.I(className="bi bi-book me-2"),
                "Panduan Auto Feature Selection"
            ])),
            dbc.ModalBody([
                dbc.Tabs([
                    dbc.Tab(label="📖 Cara Membaca", tab_id="guide-reading"),
                    dbc.Tab(label="💡 Tips & Trik", tab_id="guide-tips"),
                    dbc.Tab(label="📈 Improve Win Rate", tab_id="guide-winrate"),
                    dbc.Tab(label="❓ FAQ", tab_id="guide-faq"),
                ], id="afs-guide-tabs", active_tab="guide-reading"),
                html.Div(id="afs-guide-content", className="mt-3", style={"maxHeight": "60vh", "overflowY": "auto"})
            ]),
            dbc.ModalFooter([
                dbc.Button("Tutup", id="afs-guide-close-btn", color="secondary")
            ])
        ], id="afs-guide-modal", size="xl", is_open=False)
        
    ], fluid=True)
