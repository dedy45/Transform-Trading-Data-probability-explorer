"""
Probability Explorer Layout
Main layout for probability analysis with interactive filtering and visualization
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
from frontend.components.composite_score_viz import create_composite_score_analyzer


def create_filter_panel():
    """
    Create collapsible filter panel component
    
    Features:
    - Date range filter
    - Session filter (ASIA, EUROPE, US)
    - Probability range filter
    - Composite score filter
    - Market condition filters
    - Filter summary and presets
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-funnel me-2"),
                "Filter"
            ], className="mb-0"),
            dbc.Button(
                html.I(className="bi bi-chevron-down"),
                id="filter-collapse-button",
                color="link",
                size="sm",
                className="float-end"
            )
        ]),
        dbc.Collapse([
            dbc.CardBody([
                # Date Range Filter
                dbc.Row([
                    dbc.Col([
                        html.Label("Rentang Tanggal", className="fw-bold"),
                        dcc.DatePickerRange(
                            id='date-range-filter',
                            display_format='YYYY-MM-DD',
                            className="w-100"
                        )
                    ], md=6),
                    dbc.Col([
                        html.Label("Pilihan Cepat", className="fw-bold"),
                        dbc.ButtonGroup([
                            dbc.Button("1M", id="quick-1m", size="sm", outline=True, color="primary"),
                            dbc.Button("3M", id="quick-3m", size="sm", outline=True, color="primary"),
                            dbc.Button("6M", id="quick-6m", size="sm", outline=True, color="primary"),
                            dbc.Button("1Y", id="quick-1y", size="sm", outline=True, color="primary"),
                            dbc.Button("All", id="quick-all", size="sm", outline=True, color="primary"),
                        ], className="w-100")
                    ], md=6)
                ], className="mb-3"),
                
                # Session Filter
                dbc.Row([
                    dbc.Col([
                        html.Label("Sesi Trading", className="fw-bold"),
                        dbc.Checklist(
                            id='session-filter',
                            options=[
                                {'label': ' ASIA', 'value': 'ASIA'},
                                {'label': ' EUROPE', 'value': 'EUROPE'},
                                {'label': ' US', 'value': 'US'},
                                {'label': ' OVERLAP', 'value': 'OVERLAP'},
                            ],
                            value=['ASIA', 'EUROPE', 'US', 'OVERLAP'],
                            inline=True,
                            switch=True
                        )
                    ])
                ], className="mb-3"),
                
                # Probability Range Filter
                dbc.Row([
                    dbc.Col([
                        html.Label("Rentang Probabilitas (%)", className="fw-bold"),
                        dcc.RangeSlider(
                            id='probability-range-filter',
                            min=0,
                            max=100,
                            step=5,
                            value=[0, 100],
                            marks={i: f'{i}%' for i in range(0, 101, 20)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ])
                ], className="mb-3"),
                
                # Composite Score Filter
                dbc.Row([
                    dbc.Col([
                        html.Label("Skor Komposit Minimum", className="fw-bold"),
                        dcc.Slider(
                            id='composite-score-filter',
                            min=0,
                            max=100,
                            step=5,
                            value=0,
                            marks={i: str(i) for i in range(0, 101, 20)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ])
                ], className="mb-3"),
                
                # Market Condition Filters
                dbc.Row([
                    dbc.Col([
                        html.Label("Regime Tren", className="fw-bold"),
                        dbc.Checklist(
                            id='trend-regime-filter',
                            options=[
                                {'label': ' Trending', 'value': 1},
                                {'label': ' Ranging', 'value': 0},
                            ],
                            value=[0, 1],
                            inline=True,
                            switch=True
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Regime Volatilitas", className="fw-bold"),
                        dbc.Checklist(
                            id='volatility-regime-filter',
                            options=[
                                {'label': ' Rendah', 'value': 0},
                                {'label': ' Sedang', 'value': 1},
                                {'label': ' Tinggi', 'value': 2},
                            ],
                            value=[0, 1, 2],
                            inline=True,
                            switch=True
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Regime Risiko", className="fw-bold"),
                        dbc.Checklist(
                            id='risk-regime-filter',
                            options=[
                                {'label': ' Risk-On', 'value': 0},
                                {'label': ' Risk-Off', 'value': 1},
                            ],
                            value=[0, 1],
                            inline=True,
                            switch=True
                        )
                    ], md=4)
                ], className="mb-3"),
                
                # Filter Summary and Actions
                dbc.Row([
                    dbc.Col([
                        html.Div(id='filter-summary', className="alert alert-info mb-0")
                    ], md=8),
                    dbc.Col([
                        dbc.ButtonGroup([
                            dbc.Button("Hapus Semua", id="clear-filters-btn", size="sm", color="warning"),
                            dbc.Button("Simpan Preset", id="save-preset-btn", size="sm", color="success"),
                            dbc.DropdownMenu(
                                label="Muat Preset",
                                id="load-preset-dropdown",
                                size="sm",
                                color="primary",
                                children=[
                                    dbc.DropdownMenuItem("Probabilitas Tinggi", id="preset-high-prob"),
                                    dbc.DropdownMenuItem("Pasar Trending", id="preset-trending"),
                                    dbc.DropdownMenuItem("Volatilitas Rendah", id="preset-low-vol"),
                                    dbc.DropdownMenuItem("Sesi Eropa", id="preset-europe"),
                                ]
                            )
                        ], className="w-100")
                    ], md=4)
                ])
            ])
        ], id="filter-collapse", is_open=True)
    ], className="mb-3")


def create_control_panel():
    """
    Create control panel component with dropdowns and sliders
    
    Features:
    - Target variable selector
    - Feature X and Y selectors
    - Confidence level slider
    - Bin size slider
    - Min samples slider
    - Calculate button
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Kontrol Analisis", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Variabel Target", className="fw-bold"),
                    dcc.Dropdown(
                        id='target-variable-dropdown',
                        options=[
                            {'label': 'Menang (y_win)', 'value': 'y_win'},
                            {'label': 'Capai 1R (y_hit_1R)', 'value': 'y_hit_1R'},
                            {'label': 'Capai 2R (y_hit_2R)', 'value': 'y_hit_2R'},
                            {'label': 'Menang Masa Depan K (y_future_win_k)', 'value': 'y_future_win_k'},
                        ],
                        value='y_win',
                        clearable=False
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Fitur X", className="fw-bold"),
                    dcc.Dropdown(
                        id='feature-x-dropdown',
                        placeholder="Pilih Fitur X...",
                        clearable=False
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Fitur Y (Opsional untuk 2D)", className="fw-bold"),
                    dcc.Dropdown(
                        id='feature-y-dropdown',
                        placeholder="Pilih Fitur Y (opsional)...",
                        clearable=True
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Tipe Visualisasi", className="fw-bold"),
                    dcc.Dropdown(
                        id='viz-type-dropdown',
                        options=[
                            {'label': 'Distribusi 1D', 'value': '1d'},
                            {'label': 'Heatmap 2D', 'value': '2d'},
                        ],
                        value='1d',
                        clearable=False
                    )
                ], md=3)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Tingkat Kepercayaan (%)", className="fw-bold"),
                    dcc.Slider(
                        id='confidence-level-slider',
                        min=80,
                        max=99,
                        step=1,
                        value=95,
                        marks={80: '80%', 85: '85%', 90: '90%', 95: '95%', 99: '99%'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Jumlah Bin", className="fw-bold"),
                    dcc.Slider(
                        id='bins-slider',
                        min=5,
                        max=50,
                        step=5,
                        value=20,
                        marks={5: '5', 15: '15', 25: '25', 35: '35', 50: '50'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Sampel Minimum per Bin", className="fw-bold"),
                    dcc.Slider(
                        id='min-samples-slider',
                        min=5,
                        max=100,
                        step=5,
                        value=20,
                        marks={5: '5', 25: '25', 50: '50', 75: '75', 100: '100'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=4)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button(
                            [html.I(className="bi bi-calculator me-2"), "Hitung Probabilitas"],
                            id="calculate-btn",
                            color="primary",
                            size="lg"
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-magic me-2"), "Pilih Fitur Terbaik Otomatis"],
                            id="auto-select-btn",
                            color="info",
                            size="lg",
                            outline=True
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-download me-2"), "Ekspor Hasil"],
                            id="export-results-btn",
                            color="success",
                            size="lg",
                            outline=True,
                            disabled=True
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-info-circle me-2"), "Panduan"],
                            id="prob-help-btn",
                            color="info",
                            size="lg",
                            outline=True
                        )
                    ], className="w-100")
                ])
            ])
        ])
    ], className="mb-3")


def create_heatmap_2d_component():
    """
    Create 2D probability heatmap component
    
    Features:
    - Interactive heatmap with hover info
    - Click to select cell
    - Color scale from red (low) to green (high)
    - Opacity for low sample size cells
    - Annotations for probability values
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Heatmap Probabilitas 2D", className="mb-0")),
        dbc.CardBody([
            dcc.Loading(
                id="loading-heatmap",
                type="default",
                children=[
                    dcc.Graph(
                        id='probability-heatmap-2d',
                        config={'displayModeBar': True, 'displaylogo': False},
                        style={'height': '600px'}
                    )
                ]
            ),
            html.Div(id='heatmap-click-data', className="mt-2 text-muted small")
        ])
    ])


def create_distribution_1d_component():
    """
    Create 1D probability distribution chart component
    
    Features:
    - Bar chart with confidence intervals
    - Color-coded by probability level
    - Sample size annotations
    - Mean R-multiple overlay
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Distribusi Probabilitas 1D", className="mb-0")),
        dbc.CardBody([
            dcc.Loading(
                id="loading-distribution",
                type="default",
                children=[
                    dcc.Graph(
                        id='probability-distribution-1d',
                        config={'displayModeBar': True, 'displaylogo': False},
                        style={'height': '500px'}
                    )
                ]
            )
        ])
    ])


def create_confidence_interval_chart():
    """
    Create confidence interval visualization component
    
    Features:
    - Error bars showing CI
    - Point estimates
    - Color-coded by reliability
    - Sample size indicators
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Interval Kepercayaan", className="mb-0")),
        dbc.CardBody([
            dcc.Loading(
                id="loading-ci-chart",
                type="default",
                children=[
                    dcc.Graph(
                        id='confidence-interval-chart',
                        config={'displayModeBar': True, 'displaylogo': False},
                        style={'height': '400px'}
                    )
                ]
            )
        ])
    ])


def create_top_combinations_panel():
    """
    Create top combinations panel component
    
    Features:
    - List of top probability combinations
    - Win rate, sample size, lift ratio
    - Click to filter data
    - Export combinations
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Kombinasi Probabilitas Teratas", className="mb-0 d-inline"),
            dbc.Badge("0", id="top-combinations-count", color="primary", className="ms-2")
        ]),
        dbc.CardBody([
            dcc.Loading(
                id="loading-top-combinations",
                type="default",
                children=[
                    html.Div(id='top-combinations-list', children=[
                        dbc.Alert(
                            "Calculate probabilities to see top combinations",
                            color="info",
                            className="mb-0"
                        )
                    ])
                ]
            )
        ]),
        dbc.CardFooter([
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className="bi bi-funnel me-2"), "Terapkan sebagai Filter"],
                    id="apply-combination-filter-btn",
                    size="sm",
                    color="primary",
                    outline=True
                ),
                dbc.Button(
                    [html.I(className="bi bi-download me-2"), "Ekspor"],
                    id="export-combinations-btn",
                    size="sm",
                    color="success",
                    outline=True
                )
            ], className="w-100")
        ])
    ])


def create_trade_details_panel():
    """
    Create trade details panel component (triggered by click)
    
    Features:
    - Selected cell/bin information
    - Trade count, win rate, avg R, expectancy
    - List of trades in selection
    - Actions: view trades, export, create scenario
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Detail Trading", className="mb-0")),
        dbc.CardBody([
            html.Div(id='trade-details-content', children=[
                dbc.Alert(
                    [
                        html.I(className="bi bi-info-circle me-2"),
                        "Klik pada sel heatmap atau bar distribusi untuk melihat detail trading"
                    ],
                    color="light",
                    className="mb-0"
                )
            ])
        ]),
        dbc.CardFooter([
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className="bi bi-table me-2"), "Lihat Trading"],
                    id="view-trades-btn",
                    size="sm",
                    color="primary",
                    outline=True,
                    disabled=True
                ),
                dbc.Button(
                    [html.I(className="bi bi-download me-2"), "Ekspor"],
                    id="export-trades-btn",
                    size="sm",
                    color="success",
                    outline=True,
                    disabled=True
                ),
                dbc.Button(
                    [html.I(className="bi bi-lightbulb me-2"), "Buat Skenario"],
                    id="create-scenario-btn",
                    size="sm",
                    color="info",
                    outline=True,
                    disabled=True
                )
            ], className="w-100")
        ])
    ], id="trade-details-panel", style={'display': 'none'})


def create_probability_explorer_layout():
    """
    Create complete Probability Explorer layout
    
    Layout structure:
    - Filter panel (collapsible)
    - Control panel
    - Results section with:
      - 2D heatmap or 1D distribution (main visualization)
      - Confidence interval chart
      - Top combinations panel
      - Trade details panel
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="bi bi-graph-up me-2"),
                    "Eksplorasi Probabilitas"
                ], className="mb-2"),
                html.P(
                    "Analisis probabilitas kondisional dan temukan setup trading dengan probabilitas tinggi",
                    className="text-muted"
                )
            ])
        ], className="mb-3"),
        
        # Info Panel
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6([
                            html.I(className="bi bi-info-circle me-2"),
                            "Info & Panduan Cepat"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.Div(id='prob-info-panel', children=[
                            dbc.Alert([
                                html.I(className="bi bi-lightbulb me-2"),
                                html.Strong("Cara Menggunakan: "),
                                "1) Pilih variabel target dan fitur, 2) Atur pengaturan analisis, 3) Klik 'Hitung Probabilitas', 4) Klik pada grafik untuk melihat detail trading"
                            ], color="info", className="mb-2"),
                            html.Div(id='prob-auto-select-result', className="mb-2"),
                            html.Div(id='prob-auto-insights', className="mb-2")
                        ])
                    ])
                ])
            ])
        ], className="mb-3"),
        
        # Filter Panel
        dbc.Row([
            dbc.Col([
                create_filter_panel()
            ])
        ]),
        
        # Control Panel
        dbc.Row([
            dbc.Col([
                create_control_panel()
            ])
        ]),
        
        # Results Section
        dbc.Row([
            # Main Visualization (Left - 8 columns)
            dbc.Col([
                # 2D Heatmap (shown when viz_type='2d')
                html.Div(
                    id='heatmap-container',
                    children=[create_heatmap_2d_component()],
                    style={'display': 'none'}
                ),
                # 1D Distribution (shown when viz_type='1d')
                html.Div(
                    id='distribution-container',
                    children=[
                        create_distribution_1d_component(),
                        html.Div(className="mt-3"),
                        create_confidence_interval_chart()
                    ],
                    style={'display': 'block'}
                ),
                
                # Interpretation Panel (shown after calculate)
                html.Div(id='prob-interpretation-panel', className="mt-3")
            ], md=8),
            
            # Side Panels (Right - 4 columns)
            dbc.Col([
                create_top_combinations_panel(),
                html.Div(className="mt-3"),
                create_trade_details_panel()
            ], md=4)
        ], className="mb-4"),
        
        # Composite Score Section (INTEGRATED)
        dbc.Row([
            dbc.Col([
                html.Hr(className="my-4"),
                html.H3([
                    html.I(className="bi bi-star-fill me-2"),
                    "Composite Score Analysis"
                ], className="mb-3"),
                html.P(
                    "Combine multiple probability indicators into a single master score for filtering high-probability setups.",
                    className="text-muted mb-4"
                ),
                create_composite_score_analyzer()
            ])
        ]),
        
        # Hidden stores for data
        dcc.Store(id='probability-results-store'),
        dcc.Store(id='selected-cell-store'),
        dcc.Store(id='filter-state-store'),
        
        # Download components
        dcc.Download(id='download-results'),
        dcc.Download(id='download-trades'),
        
        # Alert for scenario creation
        dbc.Alert(
            id='scenario-created-alert',
            is_open=False,
            color='success',
            dismissable=True,
            duration=8000,
            className="position-fixed top-0 end-0 m-3",
            style={'zIndex': 9999}
        ),
        
        # Help Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Panduan Eksplorasi Probabilitas")),
            dbc.ModalBody([
                html.H5("📚 Apa itu Eksplorasi Probabilitas?"),
                html.P(
                    "Tool ini membantu Anda menemukan kombinasi kondisi pasar yang memberikan probabilitas kemenangan tertinggi. "
                    "Anda dapat menganalisis probabilitas secara 1D (satu fitur) atau 2D (dua fitur) dengan visualisasi interaktif."
                ),
                html.Hr(),
                
                html.H5("🎯 Cara Menggunakan"),
                html.Ol([
                    html.Li("Pilih Variabel Target (y_win untuk win rate, y_hit_1R untuk capai 1R, dll)"),
                    html.Li("Pilih Fitur X (wajib) - fitur pasar yang ingin dianalisis"),
                    html.Li("[Opsional] Pilih Fitur Y untuk analisis 2D"),
                    html.Li("Pilih Tipe Visualisasi (1D untuk distribusi, 2D untuk heatmap)"),
                    html.Li("Atur Tingkat Kepercayaan (default 95%)"),
                    html.Li("Atur Jumlah Bin (default 20) - lebih banyak bin = analisis lebih detail"),
                    html.Li("Atur Sampel Minimum per Bin (default 20) - untuk reliability"),
                    html.Li("Klik 'Hitung Probabilitas' untuk menjalankan analisis"),
                    html.Li("Klik pada grafik (heatmap/bar) untuk melihat detail trading di sel tersebut"),
                    html.Li("[Opsional] Ekspor hasil atau buat skenario What-If")
                ], className="mb-3"),
                html.Hr(),
                
                html.H5("📊 Interpretasi Grafik"),
                
                html.H6("Heatmap 2D:", className="mt-3"),
                html.Ul([
                    html.Li([html.Strong("Warna Hijau: "), "Probabilitas tinggi (>60%) - setup bagus"]),
                    html.Li([html.Strong("Warna Kuning: "), "Probabilitas sedang (40-60%) - netral"]),
                    html.Li([html.Strong("Warna Merah: "), "Probabilitas rendah (<40%) - hindari"]),
                    html.Li([html.Strong("Opacity Rendah: "), "Sampel sedikit, kurang reliable"]),
                    html.Li([html.Strong("Angka di Cell: "), "Nilai probabilitas dalam persen"])
                ]),
                
                html.H6("Distribusi 1D:", className="mt-3"),
                html.Ul([
                    html.Li([html.Strong("Bar Hijau: "), "Win rate > 50% (menguntungkan)"]),
                    html.Li([html.Strong("Bar Merah: "), "Win rate < 50% (tidak menguntungkan)"]),
                    html.Li([html.Strong("Error Bars: "), "Interval kepercayaan (CI)"]),
                    html.Li([html.Strong("Angka di Atas Bar: "), "Jumlah sampel di bin"]),
                    html.Li([html.Strong("Tinggi Bar: "), "Probabilitas/win rate"])
                ]),
                
                html.H6("Grafik Interval Kepercayaan:", className="mt-3"),
                html.Ul([
                    html.Li([html.Strong("Garis: "), "Trend probabilitas"]),
                    html.Li([html.Strong("Shading: "), "Area interval kepercayaan"]),
                    html.Li([html.Strong("Marker Besar: "), "Bin reliable (sampel cukup)"]),
                    html.Li([html.Strong("Marker Kecil: "), "Bin unreliable (sampel kurang)"])
                ]),
                html.Hr(),
                
                html.H5("💡 Tips & Best Practices"),
                html.Ul([
                    html.Li([html.Strong("Pemilihan Fitur: "), "Pilih fitur yang relevan dengan strategi trading Anda (composite_score, trend_strength, volatility_regime, dll)"]),
                    html.Li([html.Strong("Jumlah Bin: "), "Data <1,000: gunakan 5-10 bins | Data 1,000-10,000: gunakan 10-20 bins | Data >10,000: gunakan 30-50 bins"]),
                    html.Li([html.Strong("Sampel Minimum: "), "Eksplorasi: 5-10 | Analisis: 20-30 | Production: 50-100"]),
                    html.Li([html.Strong("Tingkat Kepercayaan: "), "Eksplorasi: 90% | Standar: 95% | Konservatif: 99%"]),
                    html.Li([html.Strong("Filter: "), "Mulai tanpa filter untuk gambaran umum, lalu tambahkan filter bertahap"]),
                    html.Li([html.Strong("Validasi: "), "Selalu cek jumlah sampel (sample count) sebelum mengambil keputusan"])
                ]),
                html.Hr(),
                
                html.H5("🐛 Troubleshooting"),
                html.Ul([
                    html.Li([html.Strong("Feature dropdown kosong: "), "Pastikan data sudah dimuat di tab Dashboard Analisis Trading"]),
                    html.Li([html.Strong("Calculate tidak ada output: "), "Cek apakah target dan feature_x sudah dipilih"]),
                    html.Li([html.Strong("Semua bin unreliable: "), "Kurangi sampel minimum atau jumlah bin"]),
                    html.Li([html.Strong("Heatmap kosong: "), "Pastikan feature_y dipilih untuk mode 2D"]),
                    html.Li([html.Strong("Click tidak ada respon: "), "Pastikan sudah klik 'Hitung Probabilitas' terlebih dahulu"]),
                    html.Li([html.Strong("Export disabled: "), "Klik pada cell/bar grafik dulu untuk mengaktifkan tombol export"])
                ]),
                html.Hr(),
                
                html.H5("📋 Contoh Kasus"),
                dbc.Card([
                    dbc.CardBody([
                        html.P(html.Strong("Kasus 1: Mencari Setup Probabilitas Tinggi"), className="text-success"),
                        html.P([
                            "Target: y_win | Feature X: composite_score | Feature Y: trend_strength | Viz: 2D",
                            html.Br(),
                            html.Strong("Hasil: "),
                            "Heatmap menunjukkan area hijau di composite_score tinggi + trend_strength tinggi dengan win rate >70%"
                        ], className="small mb-3"),
                        
                        html.P(html.Strong("Kasus 2: Validasi Strategi per Sesi"), className="text-info"),
                        html.P([
                            "Filter: Session = EUROPE | Target: y_hit_1R | Feature X: volatility_regime | Viz: 1D",
                            html.Br(),
                            html.Strong("Hasil: "),
                            "Distribution chart menunjukkan volatility regime mana yang terbaik di sesi EUROPE"
                        ], className="small mb-3"),
                        
                        html.P(html.Strong("Kasus 3: Optimasi Entry Time"), className="text-warning"),
                        html.P([
                            "Filter: Probability Range = 60-100% | Target: y_win | Feature X: entry_hour | Viz: 1D",
                            html.Br(),
                            html.Strong("Hasil: "),
                            "Menemukan jam entry terbaik dengan probabilitas tinggi"
                        ], className="small mb-0")
                    ])
                ], className="mb-3"),
                html.Hr(),
                
                html.H5("✅ Action Items"),
                html.Ol([
                    html.Li("Identifikasi kombinasi fitur dengan probabilitas tertinggi"),
                    html.Li("Validasi dengan jumlah sampel yang cukup (min 20-50)"),
                    html.Li("Klik pada cell/bar untuk melihat detail trading"),
                    html.Li("Export hasil untuk dokumentasi"),
                    html.Li("Buat skenario What-If untuk testing lebih lanjut"),
                    html.Li("Combine dengan filter untuk analisis lebih spesifik"),
                    html.Li("Monitor perubahan probabilitas over time")
                ])
            ]),
            dbc.ModalFooter([
                html.P([
                    html.I(className="bi bi-lightbulb me-2"),
                    "Tip: Gunakan tombol 'Pilih Fitur Terbaik Otomatis' untuk menemukan fitur dengan korelasi tertinggi terhadap target."
                ], className="small text-muted me-auto mb-0"),
                dbc.Button("Tutup", id="prob-help-close", color="primary")
            ])
        ], id="prob-help-modal", size="xl", is_open=False),
        
    ], fluid=True)
