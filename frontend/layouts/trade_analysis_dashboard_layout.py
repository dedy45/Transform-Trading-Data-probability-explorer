"""
Trade Analysis Dashboard Layout
Main layout for comprehensive trade performance analysis
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
from frontend.components.expectancy_dashboard import create_expectancy_dashboard


def create_summary_cards():
    """
    Create 6 summary metric cards
    
    Cards:
    - Total Trades
    - Win Rate
    - Average R
    - Expectancy
    - Max Drawdown
    - Profit Factor
    """
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Total Trading", className="text-muted mb-2"),
                    html.H3(id="summary-total-trades", children="0", className="mb-0"),
                ])
            ], className="text-center")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Tingkat Kemenangan", className="text-muted mb-2"),
                    html.H3(id="summary-win-rate", children="0%", className="mb-0"),
                ])
            ], className="text-center")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Rata-rata R-Multiple", className="text-muted mb-2"),
                    html.H3(id="summary-avg-r", children="0.00", className="mb-0"),
                ])
            ], className="text-center")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Ekspektansi", className="text-muted mb-2"),
                    html.H3(id="summary-expectancy", children="$0", className="mb-0"),
                ])
            ], className="text-center")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Drawdown Maksimal", className="text-muted mb-2"),
                    html.H3(id="summary-max-dd", children="0%", className="mb-0 text-danger"),
                ])
            ], className="text-center")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Faktor Profit", className="text-muted mb-2"),
                    html.H3(id="summary-profit-factor", children="0.00", className="mb-0"),
                ])
            ], className="text-center")
        ], md=2),
    ], className="mb-4")


def create_expectancy_analysis_section():
    """
    Create expectancy analysis section using the expectancy dashboard component
    """
    try:
        return create_expectancy_dashboard()
    except Exception as e:
        print(f"Error creating expectancy dashboard: {e}")
        return dbc.Alert([
            html.I(className="bi bi-exclamation-triangle me-2"),
            html.Strong("Expectancy Analysis: "),
            "Load data untuk melihat analisis expectancy"
        ], color="info", className="mb-4")


def create_equity_curve_section():
    """
    Create equity curve visualization with drawdown shading
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Kurva Ekuitas", className="mb-0")),
        dbc.CardBody([
            dcc.Loading(
                id="loading-equity-curve",
                type="default",
                children=[
                    dcc.Graph(
                        id='equity-curve-chart',
                        config={'displayModeBar': True, 'displaylogo': False},
                        style={'height': '400px'}
                    )
                ]
            )
        ])
    ], className="mb-4")


def create_r_distribution_section():
    """
    Create R-multiple distribution with histogram + KDE
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Distribusi R-Multiple", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-r-distribution",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='r-distribution-chart',
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '400px'}
                            )
                        ]
                    )
                ], md=8),
                dbc.Col([
                    html.H6("Statistik", className="fw-bold mb-3"),
                    html.Div(id="r-statistics-table")
                ], md=4)
            ])
        ])
    ], className="mb-4")


def create_mae_mfe_section():
    """
    Create MAE/MFE scatter plot and analysis
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Analisis MAE/MFE", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-mae-mfe",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='mae-mfe-scatter',
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '400px'}
                            )
                        ]
                    )
                ], md=8),
                dbc.Col([
                    html.H6("Statistik Menang", className="fw-bold mb-2 text-success"),
                    html.Div(id="winners-stats-table", className="mb-3"),
                    html.H6("Statistik Kalah", className="fw-bold mb-2 text-danger"),
                    html.Div(id="losers-stats-table")
                ], md=4)
            ])
        ])
    ], className="mb-4")


def create_time_based_section():
    """
    Create time-based performance analysis
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Performa Berdasarkan Waktu", className="mb-0")),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab(label="Per Jam", tab_id="hourly-tab"),
                dbc.Tab(label="Harian", tab_id="daily-tab"),
                dbc.Tab(label="Mingguan", tab_id="weekly-tab"),
                dbc.Tab(label="Bulanan", tab_id="monthly-tab"),
                dbc.Tab(label="Sesi", tab_id="session-tab"),
            ], id="time-based-tabs", active_tab="hourly-tab"),
            html.Div(id="time-based-content", className="mt-3")
        ])
    ], className="mb-4")


def create_trade_type_section():
    """
    Create trade type analysis (BUY vs SELL, exit reasons)
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Analisis Tipe Trading", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Analisis Arah", className="fw-bold mb-3"),
                    dcc.Loading(
                        id="loading-direction-analysis",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='direction-analysis-chart',
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '300px'}
                            )
                        ]
                    )
                ], md=6),
                dbc.Col([
                    html.H6("Distribusi Alasan Exit", className="fw-bold mb-3"),
                    dcc.Loading(
                        id="loading-exit-reason",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='exit-reason-chart',
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '300px'}
                            )
                        ]
                    )
                ], md=6)
            ])
        ])
    ], className="mb-4")


def create_consecutive_section():
    """
    Create consecutive trades analysis (streaks, cumulative)
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Analisis Trading Berurutan", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Timeline Streak", className="fw-bold mb-3"),
                    dcc.Loading(
                        id="loading-streak-timeline",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='streak-timeline-chart',
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '300px'}
                            )
                        ]
                    )
                ], md=6),
                dbc.Col([
                    html.H6("Performa Kumulatif", className="fw-bold mb-3"),
                    dcc.Loading(
                        id="loading-cumulative-perf",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='cumulative-perf-chart',
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '300px'}
                            )
                        ]
                    )
                ], md=6)
            ])
        ])
    ], className="mb-4")


def create_risk_metrics_section():
    """
    Create comprehensive risk metrics table
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Metrik Risiko", className="mb-0")),
        dbc.CardBody([
            html.Div(id="risk-metrics-table")
        ])
    ], className="mb-4")


def create_trade_table_section():
    """
    Create sortable, filterable, paginated trade table
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Riwayat Trading", className="mb-0 d-inline"),
            dbc.Badge("0 trading", id="trade-count-badge", color="primary", className="ms-2")
        ]),
        dbc.CardBody([
            dcc.Loading(
                id="loading-trade-table",
                type="default",
                children=[
                    html.Div(id="trade-table-container")
                ]
            )
        ])
    ], className="mb-4")


def create_trade_analysis_dashboard_layout():
    """
    Create complete Trade Analysis Dashboard layout
    
    Layout structure:
    - File upload section
    - Summary cards (6 metrics)
    - Equity curve with drawdown shading
    - R-multiple distribution with statistics
    - MAE/MFE analysis
    - Time-based performance
    - Trade type analysis
    - Consecutive trades analysis
    - Risk metrics table
    - Trade history table
    - Navigation to next page
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="bi bi-bar-chart-line me-2"),
                    "Dashboard Analisis Trading"
                ], className="mb-2"),
                html.P(
                    "Analisis komprehensif performa trading dan metrik risiko",
                    className="text-muted"
                )
            ])
        ], className="mb-3"),
        
        # Info: Data dimuat dari Global Data Loader di atas
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.I(className="bi bi-info-circle me-2"),
                    html.Strong("Info: "),
                    "Data trading dimuat dari Global Data Loader di bagian atas halaman. "
                    "Pilih file trade dan feature CSV, lalu klik 'Muat Data Terpilih'."
                ], color="info", className="mb-3")
            ])
        ]),
        
        # Summary Cards
        create_summary_cards(),
        
        # Expectancy Analysis Section (NEW FEATURE)
        create_expectancy_analysis_section(),
        
        # Other Analysis Sections
        create_equity_curve_section(),
        create_r_distribution_section(),
        create_mae_mfe_section(),
        create_time_based_section(),
        create_trade_type_section(),
        create_consecutive_section(),
        create_risk_metrics_section(),
        create_trade_table_section(),
        
        # Navigation Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Siap untuk Analisis Lanjutan?", className="mb-2"),
                                html.P(
                                    "Jelajahi analisis probabilitas, deteksi regime, dan skenario what-if",
                                    className="text-muted mb-0"
                                )
                            ], md=6),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-cpu me-2"), "Prediksi dengan ML"],
                                    id="navigate-to-ml-prediction-btn",
                                    color="primary",
                                    size="lg",
                                    className="w-100 mb-2",
                                    disabled=True
                                )
                            ], md=3),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-arrow-right me-2"), "Selanjutnya: Analisis Fitur"],
                                    id="navigate-to-probability-btn",
                                    color="success",
                                    size="lg",
                                    className="w-100 mb-2",
                                    disabled=True
                                )
                            ], md=3)
                        ])
                    ])
                ], className="bg-light")
            ])
        ], className="mb-4"),
        
        # Export Section
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="bi bi-file-pdf me-2"), "Ekspor Laporan (PDF)"],
                        id="export-pdf-btn",
                        color="danger",
                        outline=True,
                        disabled=True
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-file-excel me-2"), "Ekspor Data (CSV)"],
                        id="export-csv-btn",
                        color="success",
                        outline=True,
                        disabled=True
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-image me-2"), "Ekspor Grafik (PNG)"],
                        id="export-charts-btn",
                        color="info",
                        outline=True,
                        disabled=True
                    )
                ], className="w-100")
            ])
        ], className="mb-4"),
        
        # Hidden stores for data
        dcc.Store(id='trade-data-loaded-store'),
        dcc.Store(id='selected-trade-store'),
        
        # Download components
        dcc.Download(id="download-report"),
        dcc.Download(id="download-data"),
        
    ], fluid=True)
