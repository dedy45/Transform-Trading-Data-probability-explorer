"""
Regime Explorer Layout
Layout for market regime probability analysis
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from frontend.components.regime_filter_controls import (
    create_regime_filter_controls,
    create_empty_regime_summary_cards
)
from frontend.components.regime_winrate_chart import create_empty_regime_winrate_chart
from frontend.components.regime_r_multiple_chart import create_empty_regime_r_multiple_chart
from frontend.components.regime_comparison_table import create_empty_regime_comparison_table


def create_regime_explorer_layout():
    """
    Create complete Regime Explorer layout
    
    Layout structure:
    - Header with description
    - Summary cards (4 key metrics: best/worst regime, highest R, total regimes)
    - Analysis controls and regime selection
    - Win rate bar chart per regime
    - R-multiple threshold probabilities chart (P(R>=1), P(R>=2))
    - Comprehensive comparison table
    - Insights and recommendations
    
    Requirements: 12.5
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="bi bi-bar-chart-line me-2"),
                    "Eksplorasi Regime"
                ], className="mb-2"),
                html.P(
                    "Analisis performa trading di berbagai regime pasar. "
                    "Identifikasi kondisi pasar mana yang memberikan keunggulan terbaik dan sesuaikan strategi Anda. "
                    "Bandingkan tingkat kemenangan, R-multiple, dan signifikansi statistik di berbagai regime.",
                    className="text-muted"
                )
            ])
        ], className="mb-3"),
        
        # Summary Cards
        dbc.Row([
            dbc.Col([
                html.Div(id='regime-summary-cards', children=[create_empty_regime_summary_cards()])
            ])
        ]),
        
        # Analysis Controls
        dbc.Row([
            dbc.Col([
                html.Div(id='regime-filter-controls', children=[create_regime_filter_controls()])
            ])
        ]),
        
        # Info Panel
        dbc.Row([
            dbc.Col([
                html.Div(id='regime-info-panel', children=[
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Alert([
                                html.I(className="bi bi-info-circle me-2"),
                                "Klik tombol 'Hitung Analisis Regime' untuk memulai analisis"
                            ], color="info", className="mb-0")
                        ])
                    ])
                ])
            ])
        ], className="mb-3"),
        
        # Charts Row 1: Win Rate and R-Multiple
        dbc.Row([
            # Win Rate Chart
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Tingkat Kemenangan per Regime", className="mb-0 d-inline"),
                        dbc.Badge("Dengan Interval Kepercayaan", color="info", className="ms-2")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-regime-winrate",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id='regime-winrate-chart',
                                    figure=create_empty_regime_winrate_chart(),
                                    config={'displayModeBar': True, 'displaylogo': False}
                                )
                            ]
                        ),
                        html.Div([
                            html.P([
                                html.Strong("Interpretasi: "),
                                "Bar hijau menunjukkan tingkat kemenangan > 50% (menguntungkan). "
                                "Bar merah menunjukkan tingkat kemenangan < 50% (tidak menguntungkan). "
                                "Error bar menunjukkan interval kepercayaan. "
                                "Fokus pada regime dengan tingkat kemenangan tinggi dan ukuran sampel yang memadai."
                            ], className="text-muted small mb-0")
                        ], className="mt-2")
                    ])
                ], className="mb-3")
            ], md=6),
            
            # R-Multiple Chart
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Ambang R-Multiple per Regime", className="mb-0 d-inline"),
                        dbc.Badge("P(R≥1) dan P(R≥2)", color="info", className="ms-2")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-regime-r-multiple",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id='regime-r-multiple-chart',
                                    figure=create_empty_regime_r_multiple_chart(),
                                    config={'displayModeBar': True, 'displaylogo': False}
                                )
                            ]
                        ),
                        html.Div([
                            html.P([
                                html.Strong("Interpretasi: "),
                                "Bar biru menunjukkan P(R ≥ 1) - probabilitas minimal impas. "
                                "Bar hijau menunjukkan P(R ≥ 2) - probabilitas menggandakan risiko. "
                                "Nilai yang lebih tinggi menunjukkan performa risk-reward yang lebih baik. "
                                "P(R ≥ 2) harus selalu ≤ P(R ≥ 1)."
                            ], className="text-muted small mb-0")
                        ], className="mt-2")
                    ])
                ], className="mb-3")
            ], md=6)
        ]),
        
        # Comparison Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Perbandingan Regime Komprehensif", className="mb-0 d-inline"),
                        dbc.Badge("Semua Metrik", color="info", className="ms-2")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-regime-comparison-table",
                            type="default",
                            children=[
                                html.Div(
                                    id='regime-comparison-table',
                                    children=[create_empty_regime_comparison_table()]
                                )
                            ]
                        )
                    ])
                ], className="mb-3")
            ])
        ]),
        
        # Insights and Recommendations
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Wawasan & Rekomendasi Regime", className="mb-0 d-inline"),
                        html.I(className="bi bi-lightbulb ms-2")
                    ]),
                    dbc.CardBody([
                        html.Div(id='regime-insights-content', children=[
                            dbc.Alert([
                                html.I(className="bi bi-info-circle me-2"),
                                "Hitung analisis regime untuk melihat wawasan dan rekomendasi yang dapat ditindaklanjuti"
                            ], color="info", className="mb-0")
                        ])
                    ])
                ], className="mb-3")
            ])
        ]),
        
        # Additional Analysis: Regime Transitions (Optional)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Analisis Transisi Regime", className="mb-0 d-inline"),
                        dbc.Badge("Opsional", color="secondary", className="ms-2")
                    ]),
                    dbc.CardBody([
                        html.Div([
                            html.P([
                                html.I(className="bi bi-diagram-3 me-2"),
                                "Analyze how market regimes transition from one to another. "
                                "This helps understand regime persistence and switching patterns."
                            ], className="text-muted mb-2"),
                            dbc.Button(
                                [html.I(className="bi bi-arrow-right-circle me-2"), "Calculate Regime Transitions"],
                                id="regime-transition-btn",
                                color="secondary",
                                outline=True
                            )
                        ]),
                        html.Div(id='regime-transition-content', className="mt-3")
                    ])
                ], className="mb-3")
            ])
        ]),
        
        # Hidden stores for data
        dcc.Store(id='regime-probs-store'),
        dcc.Store(id='regime-threshold-store'),
        dcc.Store(id='regime-comparison-store'),
        dcc.Store(id='regime-transition-store'),
        dcc.Store(id='available-regimes-store'),
        dcc.Store(id='regime-results-store'),
        dcc.Download(id='regime-download-data'),
        
        # Help Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Panduan Eksplorasi Regime")),
            dbc.ModalBody([
                html.H5("Apa itu Analisis Regime?"),
                html.P(
                    "Analisis regime memeriksa bagaimana performa trading bervariasi di berbagai kondisi pasar. "
                    "Dengan mengidentifikasi regime mana yang memberikan keunggulan terbaik, Anda dapat menyesuaikan strategi "
                    "untuk trading lebih agresif di kondisi yang menguntungkan dan mengurangi eksposur di kondisi yang tidak menguntungkan."
                ),
                html.Hr(),
                html.H5("Tipe Regime Umum"),
                html.Ul([
                    html.Li([
                        html.Strong("Regime Tren: "),
                        "Pasar Trending (terarah) vs Ranging (sideways). "
                        "Banyak strategi berkinerja lebih baik di satu regime daripada yang lain."
                    ]),
                    html.Li([
                        html.Strong("Regime Volatilitas: "),
                        "Volatilitas Rendah, Sedang, atau Tinggi. "
                        "Mempengaruhi penempatan stop loss dan ukuran posisi."
                    ]),
                    html.Li([
                        html.Strong("Regime Risiko: "),
                        "Risk-on (sentimen bullish) vs Risk-off (defensif). "
                        "Mempengaruhi korelasi dan perilaku pasar."
                    ]),
                    html.Li([
                        html.Strong("Sesi: "),
                        "Sesi trading Asia, Eropa, AS. "
                        "Karakteristik likuiditas dan volatilitas yang berbeda."
                    ])
                ]),
                html.Hr(),
                html.H5("Metrik Utama"),
                html.Ul([
                    html.Li([
                        html.Strong("Tingkat Kemenangan: "),
                        "Persentase trading yang menang di setiap regime. "
                        "Cari regime dengan tingkat kemenangan > 50%."
                    ]),
                    html.Li([
                        html.Strong("Mean R: "),
                        "Rata-rata R-multiple per regime. "
                        "Nilai positif menunjukkan profitabilitas. Semakin tinggi semakin baik."
                    ]),
                    html.Li([
                        html.Strong("P(R ≥ 1): "),
                        "Probabilitas minimal impas. "
                        "Harus > 50% untuk regime yang menguntungkan."
                    ]),
                    html.Li([
                        html.Strong("P(R ≥ 2): "),
                        "Probabilitas menggandakan risiko Anda. "
                        "Menunjukkan potensi untuk kemenangan besar."
                    ]),
                    html.Li([
                        html.Strong("Reliabel: "),
                        "Menunjukkan ukuran sampel yang cukup untuk kepercayaan statistik. "
                        "Hanya percaya hasil dari regime yang reliabel."
                    ])
                ]),
                html.Hr(),
                html.H5("Strategi yang Dapat Ditindaklanjuti"),
                html.Ul([
                    html.Li("Trading hanya di regime dengan Mean R positif dan Tingkat Kemenangan > 50%"),
                    html.Li("Tingkatkan ukuran posisi di regime dengan performa terbaik"),
                    html.Li("Kurangi atau hindari trading di regime dengan performa terburuk"),
                    html.Li("Gunakan filter regime sebagai kriteria entry di sistem trading Anda"),
                    html.Li("Monitor transisi regime untuk mengantisipasi perubahan keunggulan"),
                    html.Li("Gabungkan beberapa filter regime untuk setup dengan probabilitas lebih tinggi")
                ]),
                html.Hr(),
                html.H5("Tips Interpretasi"),
                html.Ul([
                    html.Li("Fokus pada regime dengan ukuran sampel yang memadai (Reliabel = Ya)"),
                    html.Li("Bandingkan interval kepercayaan - CI yang tumpang tindih menunjukkan tidak ada perbedaan signifikan"),
                    html.Li("Cari pola konsisten di tingkat kemenangan dan mean R"),
                    html.Li("Pertimbangkan signifikansi praktis, bukan hanya signifikansi statistik"),
                    html.Li("Validasi temuan dengan data out-of-sample sebelum trading live")
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Tutup", id="regime-help-close", className="ms-auto")
            )
        ], id="regime-help-modal", size="xl", is_open=False)
        
    ], fluid=True)
