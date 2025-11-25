"""
What-If Scenarios Layout
Interactive scenario builder and comparison dashboard for trading strategy optimization
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from frontend.components.mae_mfe_optimizer import create_mae_mfe_optimizer
from frontend.components.monte_carlo_viz import create_monte_carlo_simulator


def create_scenario_builder_panel():
    """
    Create scenario builder panel (left sidebar)
    
    Features:
    - Starting equity input
    - Scenario type selector
    - Parameter inputs based on scenario type
    - Add scenario button
    - Active scenarios list
    - Scenario management (edit, delete, duplicate)
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-sliders me-2"),
                "Pembuat Skenario"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            # Starting Equity Input
            dbc.Row([
                dbc.Col([
                    html.Label([
                        html.I(className="bi bi-cash-stack me-1"),
                        "Ekuitas Awal ($)"
                    ], className="fw-bold"),
                    dbc.Input(
                        id='starting-equity-input',
                        type='number',
                        min=100,
                        step=100,
                        value=10000,
                        placeholder='Masukkan ekuitas awal...'
                    ),
                    html.Small("Default: $10,000 | Digunakan untuk perhitungan kurva ekuitas", 
                              className="text-muted")
                ])
            ], className="mb-3"),
            
            html.Hr(),
            
            # Scenario Name Input
            dbc.Row([
                dbc.Col([
                    html.Label("Nama Skenario", className="fw-bold"),
                    dbc.Input(
                        id='scenario-name-input',
                        type='text',
                        placeholder='Masukkan nama skenario...',
                        value=''
                    )
                ])
            ], className="mb-3"),
            
            # Scenario Type Selector
            dbc.Row([
                dbc.Col([
                    html.Label("Tipe Skenario", className="fw-bold"),
                    dcc.Dropdown(
                        id='scenario-type-dropdown',
                        options=[
                            {'label': '💰 Ukuran Posisi', 'value': 'position_sizing'},
                            {'label': '🎯 Penyesuaian SL/TP', 'value': 'sl_tp'},
                            {'label': '🔍 Filter Trading', 'value': 'filter'},
                            {'label': '⏰ Pembatasan Waktu', 'value': 'time'},
                            {'label': '📊 Kondisi Pasar', 'value': 'market_condition'},
                            {'label': '💵 Manajemen Uang', 'value': 'money_management'},
                            {'label': '🤖 Prediksi ML', 'value': 'ml_prediction'},
                        ],
                        value='position_sizing',
                        clearable=False
                    )
                ])
            ], className="mb-3"),
            
            # Dynamic Parameter Panel (changes based on scenario type)
            html.Div(id='scenario-parameters-panel'),
            
            # Add Scenario Button
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [html.I(className="bi bi-plus-circle me-2"), "Tambah Skenario"],
                        id="add-scenario-btn",
                        color="primary",
                        size="lg",
                        className="w-100"
                    )
                ])
            ], className="mb-3"),
            
            html.Hr(),
            
            # Active Scenarios List
            html.Div([
                html.Label("Skenario Aktif", className="fw-bold mb-2"),
                html.Div(id='active-scenarios-list', children=[
                    dbc.Alert(
                        "Belum ada skenario yang ditambahkan. Buat skenario pertama Anda di atas.",
                        color="light",
                        className="mb-0 small"
                    )
                ])
            ])
        ])
    ], className="h-100", style={'position': 'sticky', 'top': '20px'})


def create_scenario_comparison_table():
    """
    Create scenario comparison table component
    
    Features:
    - Side-by-side metrics comparison
    - Color-coded cells (green=better, red=worse)
    - Sortable columns
    - Percentage change from baseline
    - Export functionality
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Perbandingan Skenario", className="mb-0 d-inline"),
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className="bi bi-arrow-clockwise me-1"), "Segarkan"],
                    id="refresh-comparison-btn",
                    size="sm",
                    color="primary",
                    outline=True,
                    className="ms-2"
                ),
                dbc.Button(
                    [html.I(className="bi bi-download me-1"), "Ekspor"],
                    id="export-comparison-btn",
                    size="sm",
                    color="success",
                    outline=True
                )
            ], size="sm", className="float-end")
        ]),
        dbc.CardBody([
            dcc.Loading(
                id="loading-comparison-table",
                type="default",
                children=[
                    html.Div(id='scenario-comparison-table-container')
                ]
            )
        ])
    ], className="mb-3")


def create_equity_curve_comparison():
    """
    Create equity curve comparison component (fan chart)
    
    Features:
    - Overlaid equity curves for multiple scenarios
    - Color differentiation
    - Drawdown shading
    - Legend with scenario names
    - Zoom and pan
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Perbandingan Kurva Ekuitas", className="mb-0 d-inline"),
            dbc.ButtonGroup([
                dbc.Checklist(
                    id='show-drawdown-toggle',
                    options=[{'label': ' Tampilkan Drawdown', 'value': 'show'}],
                    value=['show'],
                    inline=True,
                    switch=True,
                    className="ms-3"
                )
            ], className="float-end")
        ]),
        dbc.CardBody([
            dcc.Loading(
                id="loading-equity-curves",
                type="default",
                children=[
                    dcc.Graph(
                        id='equity-curve-comparison-chart',
                        config={'displayModeBar': True, 'displaylogo': False},
                        style={'height': '500px'}
                    )
                ]
            )
        ])
    ], className="mb-3")


def create_metrics_radar_chart():
    """
    Create metrics radar chart component
    
    Features:
    - Multi-axis radar chart
    - Normalized metrics (0-100 scale)
    - Baseline highlighted
    - Multiple scenarios overlaid
    - Metrics: Win Rate, Avg R, Expectancy, Profit Factor, Sharpe, Recovery Factor
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Radar Metrik Performa", className="mb-0 d-inline"),
            dbc.Badge("Dinormalisasi 0-100", color="info", className="ms-2")
        ]),
        dbc.CardBody([
            dcc.Loading(
                id="loading-radar-chart",
                type="default",
                children=[
                    dcc.Graph(
                        id='metrics-radar-chart',
                        config={'displayModeBar': True, 'displaylogo': False},
                        style={'height': '450px'}
                    )
                ]
            )
        ])
    ], className="mb-3")


def create_trade_distribution_comparison():
    """
    Create trade distribution comparison component
    
    Features:
    - Histogram, box plots, or violin plots
    - R-multiple distribution per scenario
    - Win/loss distribution
    - Holding time distribution
    - Toggle between distribution types
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Perbandingan Distribusi Trading", className="mb-0 d-inline"),
            dbc.RadioItems(
                id='distribution-type-radio',
                options=[
                    {'label': ' Histogram', 'value': 'histogram'},
                    {'label': ' Plot Kotak', 'value': 'box'},
                    {'label': ' Plot Biola', 'value': 'violin'},
                ],
                value='histogram',
                inline=True,
                className="float-end"
            )
        ]),
        dbc.CardBody([
            dcc.Loading(
                id="loading-distribution-comparison",
                type="default",
                children=[
                    dcc.Graph(
                        id='trade-distribution-comparison-chart',
                        config={'displayModeBar': True, 'displaylogo': False},
                        style={'height': '400px'}
                    )
                ]
            )
        ])
    ], className="mb-3")


def create_scenario_save_load_panel():
    """
    Create scenario save/load functionality panel
    
    Features:
    - Save current scenario set
    - Load saved scenario set
    - Delete saved sets
    - Import/export scenario configurations
    """
    return dbc.Card([
        dbc.CardHeader(html.H6("Manajemen Skenario", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup([
                        dbc.Input(
                            id='scenario-set-name-input',
                            placeholder='Nama set skenario...',
                            type='text'
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-save me-1"), "Simpan Set"],
                            id='save-scenario-set-btn',
                            color="success",
                            outline=True
                        )
                    ], size="sm")
                ], md=6),
                dbc.Col([
                    dbc.InputGroup([
                        dcc.Dropdown(
                            id='load-scenario-set-dropdown',
                            placeholder='Pilih set tersimpan...',
                            options=[],
                            className="flex-grow-1"
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-folder-open me-1"), "Muat"],
                            id='load-scenario-set-btn',
                            color="primary",
                            outline=True
                        )
                    ], size="sm")
                ], md=6)
            ], className="mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button(
                            [html.I(className="bi bi-upload me-1"), "Impor JSON"],
                            id='import-scenarios-btn',
                            size="sm",
                            color="info",
                            outline=True
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-download me-1"), "Ekspor JSON"],
                            id='export-scenarios-btn',
                            size="sm",
                            color="info",
                            outline=True
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-trash me-1"), "Hapus Semua"],
                            id='clear-all-scenarios-btn',
                            size="sm",
                            color="danger",
                            outline=True
                        )
                    ], size="sm", className="w-100")
                ])
            ])
        ])
    ], className="mb-3")


def create_help_modal():
    """Create help modal with usage instructions"""
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="bi bi-question-circle me-2"),
            "Skenario What-If - Panduan Bantuan"
        ])),
        dbc.ModalBody([
            html.H5("🎯 Mulai Cepat", className="mb-3"),
            html.Ol([
                html.Li("Pilih tipe skenario dari dropdown"),
                html.Li("Konfigurasi parameter untuk skenario Anda"),
                html.Li("Klik 'Tambah Skenario' untuk menambahkannya ke daftar"),
                html.Li("Ulangi untuk menambahkan lebih banyak skenario"),
                html.Li("Klik 'Segarkan' untuk membandingkan semua skenario")
            ], className="mb-4"),
            
            html.H5("📊 Tipe Skenario", className="mb-3"),
            html.Ul([
                html.Li([html.Strong("Ukuran Posisi:"), " Uji persentase risiko dan ukuran lot yang berbeda"]),
                html.Li([html.Strong("Penyesuaian SL/TP:"), " Modifikasi level stop loss dan take profit"]),
                html.Li([html.Strong("Filter Trading:"), " Filter trading berdasarkan probabilitas, skor, atau kondisi"]),
                html.Li([html.Strong("Pembatasan Waktu:"), " Batasi trading pada jam atau hari tertentu"]),
                html.Li([html.Strong("Kondisi Pasar:"), " Filter berdasarkan tren, volatilitas, atau regime"]),
                html.Li([html.Strong("Manajemen Uang:"), " Terapkan compounding, martingale, atau batasan"])
            ], className="mb-4"),
            
            html.H5("💡 Tips", className="mb-3"),
            html.Ul([
                html.Li("Mulai dengan 2-3 skenario untuk perbandingan yang lebih mudah"),
                html.Li("Simpan set skenario yang berguna untuk referensi masa depan"),
                html.Li("Ekspor tabel perbandingan untuk dokumentasi"),
                html.Li("Perhatikan pengurangan jumlah trading saat filtering"),
                html.Li("Seimbangkan peningkatan profit dengan pengurangan risiko")
            ], className="mb-4"),
            
            html.H5("📈 Interpretasi Hasil", className="mb-3"),
            html.P([
                "Nilai hijau menunjukkan peningkatan dari baseline, merah menunjukkan performa lebih buruk. ",
                "Fokus pada skenario yang meningkatkan beberapa metrik secara bersamaan. ",
                "Pertimbangkan trade-off antara profit dan drawdown."
            ], className="mb-2"),
            
            html.Hr(),
            
            html.P([
                html.I(className="bi bi-book me-2"),
                "Untuk panduan lengkap, lihat ",
                html.Strong("WHATIF_SCENARIOS_COMPLETE_GUIDE.md")
            ], className="text-muted small mb-0")
        ]),
        dbc.ModalFooter(
            dbc.Button("Tutup", id="close-whatif-help-modal", className="ms-auto")
        )
    ], id="whatif-help-modal", size="lg", is_open=False)


def create_whatif_scenarios_layout():
    """
    Create complete What-If Scenarios layout
    
    Layout structure:
    - Left sidebar: Scenario builder panel (3 columns)
    - Right main area: Comparison and visualization (9 columns)
      - Scenario comparison table
      - Equity curve comparison
      - Metrics radar chart
      - Trade distribution comparison
      - Scenario save/load panel
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="bi bi-lightbulb me-2"),
                    "Skenario What-If"
                ], className="mb-2"),
                html.P(
                    "Simulasikan berbagai skenario trading dan bandingkan dampaknya terhadap metrik performa",
                    className="text-muted"
                )
            ], md=10),
            dbc.Col([
                dbc.Button([
                    html.I(className="bi bi-question-circle me-1"),
                    "Bantuan"
                ], id="open-whatif-help-btn", color="info", outline=True, size="sm")
            ], md=2, className="text-end")
        ], className="mb-3"),
        
        # Main Content
        dbc.Row([
            # Left Sidebar - Scenario Builder
            dbc.Col([
                create_scenario_builder_panel()
            ], md=3),
            
            # Right Main Area - Comparison and Visualization
            dbc.Col([
                # Summary Cards
                html.Div(id='whatif-summary-cards'),
                
                # MAE/MFE Optimizer Section (INTEGRATED)
                create_mae_mfe_optimizer(),
                
                # Monte Carlo Simulation Section (INTEGRATED)
                create_monte_carlo_simulator(),
                
                # Insights Panel
                dbc.Card([
                    dbc.CardHeader(html.H6([
                        html.I(className="bi bi-lightbulb me-2"),
                        "Wawasan Otomatis"
                    ], className="mb-0")),
                    dbc.CardBody(
                        html.Div(id='whatif-insights-panel')
                    )
                ], className="mb-3"),
                
                # Scenario Management
                create_scenario_save_load_panel(),
                
                # Comparison Table
                create_scenario_comparison_table(),
                
                # Visualizations
                dbc.Row([
                    dbc.Col([
                        create_equity_curve_comparison()
                    ], md=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        create_metrics_radar_chart()
                    ], md=6),
                    dbc.Col([
                        create_trade_distribution_comparison()
                    ], md=6)
                ]),
                
                # Recommendations Panel
                html.Div(id='whatif-recommendations-panel')
            ], md=9)
        ]),
        
        # Help Modal
        create_help_modal(),
        
        # Hidden stores for data
        dcc.Store(id='scenarios-store', data=[]),
        dcc.Store(id='baseline-metrics-store'),
        dcc.Store(id='scenario-results-store', data={}),
        dcc.Store(id='saved-scenario-sets-store', data={}),
        dcc.Store(id='starting-equity-store', data=10000),
        dcc.Store(id='ml-predictions-store', data=None),  # Store for ML predictions
        
        # Download components
        dcc.Download(id='download-comparison-csv'),
        dcc.Download(id='download-scenarios-json'),
        
        # Upload component (hidden)
        dcc.Upload(
            id='upload-scenarios-json',
            children=html.Div([]),
            style={'display': 'none'}
        )
        
    ], fluid=True)
