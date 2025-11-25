"""
Sequential Analysis Layout
Layout for Markov chain and streak analysis
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from frontend.components.markov_summary_cards import create_empty_summary_cards
from frontend.components.transition_matrix import create_empty_transition_matrix
from frontend.components.streak_distribution import create_empty_streak_distribution
from frontend.components.conditional_streak_chart import create_empty_conditional_streak_chart


def create_sequential_analysis_layout():
    """
    Create complete Sequential Analysis layout
    
    Layout structure:
    - Header with description
    - Summary cards (4 key metrics)
    - Transition matrix heatmap
    - Streak distribution charts
    - Conditional win rate chart P(Win | loss_streak = k)
    - Analysis controls and settings
    
    Requirements: 12.3
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="bi bi-diagram-3 me-2"),
                    "Analisis Sekuensial"
                ], className="mb-2"),
                html.P(
                    "Analisis pola sekuensial dalam hasil trading menggunakan rantai Markov dan analisis streak. "
                    "Pahami bagaimana hasil masa lalu mempengaruhi probabilitas masa depan.",
                    className="text-muted"
                )
            ])
        ], className="mb-3"),
        
        # Summary Cards
        dbc.Row([
            dbc.Col([
                html.Div(id='markov-summary-cards', children=[create_empty_summary_cards()])
            ])
        ]),
        
        # Analysis Controls
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Pengaturan Analisis", className="mb-0")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Variabel Target", className="fw-bold"),
                                dcc.Dropdown(
                                    id='seq-target-variable-dropdown',
                                    options=[
                                        {'label': 'Sukses Trading (Menang/Kalah)', 'value': 'trade_success'},
                                        {'label': 'Capai 1R', 'value': 'y_hit_1R'},
                                        {'label': 'Capai 2R', 'value': 'y_hit_2R'},
                                    ],
                                    value='trade_success',
                                    clearable=False
                                )
                            ], md=3),
                            dbc.Col([
                                html.Label("Tingkat Kepercayaan (%)", className="fw-bold"),
                                dcc.Slider(
                                    id='seq-confidence-level-slider',
                                    min=80,
                                    max=99,
                                    step=1,
                                    value=95,
                                    marks={80: '80%', 85: '85%', 90: '90%', 95: '95%', 99: '99%'},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], md=3),
                            dbc.Col([
                                html.Label("Panjang Streak Maksimal", className="fw-bold"),
                                dcc.Slider(
                                    id='seq-max-streak-slider',
                                    min=5,
                                    max=100,
                                    step=5,
                                    value=20,
                                    marks={5: '5', 20: '20', 40: '40', 60: '60', 80: '80', 100: '100'},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], md=3),
                            dbc.Col([
                                html.Label("Sampel Minimum (0=Semua Data)", className="fw-bold"),
                                dcc.Slider(
                                    id='seq-min-samples-slider',
                                    min=0,
                                    max=100,
                                    step=5,
                                    value=5,
                                    marks={0: 'Semua', 5: '5', 20: '20', 50: '50', 100: '100'},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], md=3)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-calculator me-2"), "Hitung Analisis Sekuensial"],
                                    id="seq-calculate-btn",
                                    color="primary",
                                    size="lg",
                                    className="w-100"
                                )
                            ], md=6),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-download me-2"), "Ekspor Hasil"],
                                    id="seq-export-btn",
                                    color="success",
                                    size="lg",
                                    outline=True,
                                    className="w-100"
                                )
                            ], md=3),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-info-circle me-2"), "Panduan"],
                                    id="seq-help-btn",
                                    color="info",
                                    size="lg",
                                    outline=True,
                                    className="w-100"
                                )
                            ], md=3)
                        ])
                    ])
                ], className="mb-3")
            ])
        ]),
        
        # Transition Matrix
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Matriks Transisi Markov", className="mb-0 d-inline"),
                        dbc.Badge("Orde Pertama", color="info", className="ms-2")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-transition-matrix",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id='transition-matrix-heatmap',
                                    figure=create_empty_transition_matrix(),
                                    config={'displayModeBar': True, 'displaylogo': False}
                                )
                            ]
                        ),
                        html.Div([
                            html.P([
                                html.Strong("Interpretasi: "),
                                "Matriks ini menunjukkan probabilitas hasil trading berikutnya berdasarkan hasil saat ini. "
                                "Setiap baris berjumlah 100%. P(Menang|Menang) tinggi menunjukkan momentum, sedangkan P(Menang|Kalah) tinggi menunjukkan pemulihan yang baik."
                            ], className="text-muted small mb-0")
                        ], className="mt-2")
                    ])
                ], className="mb-3")
            ], md=6),
            
            # Streak Distribution
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Distribusi Streak", className="mb-0 d-inline"),
                        dbc.Badge("Urutan Menang/Kalah", color="info", className="ms-2")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-streak-distribution",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id='streak-distribution-chart',
                                    figure=create_empty_streak_distribution(),
                                    config={'displayModeBar': True, 'displaylogo': False}
                                )
                            ]
                        ),
                        html.Div([
                            html.P([
                                html.Strong("Interpretasi: "),
                                "Menunjukkan frekuensi kemenangan dan kekalahan berturut-turut. "
                                "Streak yang lebih panjang lebih jarang terjadi. Bandingkan streak maksimal untuk menilai toleransi risiko."
                            ], className="text-muted small mb-0")
                        ], className="mt-2")
                    ])
                ], className="mb-3")
            ], md=6)
        ]),
        
        # Conditional Win Rate Chart
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Tingkat Kemenangan Kondisional Setelah Streak Kalah", className="mb-0 d-inline"),
                        dbc.Badge("P(Menang | Streak Kalah = k)", color="info", className="ms-2")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-conditional-streak",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id='conditional-streak-chart',
                                    figure=create_empty_conditional_streak_chart(),
                                    config={'displayModeBar': True, 'displaylogo': False}
                                )
                            ]
                        ),
                        html.Div([
                            html.P([
                                html.Strong("Interpretasi: "),
                                "Menunjukkan bagaimana probabilitas menang berubah setelah kekalahan berturut-turut. "
                                "Jika garis naik, Anda lebih mungkin menang setelah streak kalah yang lebih panjang (mean reversion). "
                                "Jika garis turun, kekalahan cenderung berkelompok (momentum). "
                                "Garis datar menunjukkan independensi (random walk)."
                            ], className="text-muted small mb-0")
                        ], className="mt-2")
                    ])
                ], className="mb-3")
            ])
        ]),
        
        # Insights and Recommendations
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Wawasan Utama", className="mb-0 d-inline"),
                        html.I(className="bi bi-lightbulb ms-2")
                    ]),
                    dbc.CardBody([
                        html.Div(id='seq-insights-content', children=[
                            dbc.Alert([
                                html.I(className="bi bi-info-circle me-2"),
                                "Hitung analisis sekuensial untuk melihat wawasan dan rekomendasi"
                            ], color="info", className="mb-0")
                        ])
                    ])
                ], className="mb-3")
            ])
        ]),
        
        # Hidden stores for data
        dcc.Store(id='markov-results-store'),
        dcc.Store(id='streak-results-store'),
        dcc.Store(id='conditional-results-store'),
        
        # Download component
        dcc.Download(id='seq-download-data'),
        
        # Help Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Panduan Sequential Analysis")),
            dbc.ModalBody([
                html.H5("📚 Apa itu Sequential Analysis?"),
                html.P(
                    "Sequential Analysis menganalisis pola urutan hasil trading (win/loss) untuk memahami "
                    "dependensi temporal dan probabilitas kondisional. Tool ini membantu Anda mengidentifikasi "
                    "apakah hasil trading Anda independent atau ada pola streak yang signifikan."
                ),
                html.Hr(),
                
                html.H5("📊 Komponen Utama"),
                html.Ul([
                    html.Li([
                        html.Strong("Markov Transition Matrix: "),
                        "Menunjukkan probabilitas transisi dari satu state (Win/Loss) ke state berikutnya. ",
                        "Membantu memahami apakah hasil trading memiliki 'memory' atau independent."
                    ], className="mb-2"),
                    html.Li([
                        html.Strong("Streak Distribution: "),
                        "Distribusi panjang winning streaks dan losing streaks. ",
                        "Membantu memahami pola clustering wins/losses."
                    ], className="mb-2"),
                    html.Li([
                        html.Strong("Conditional Probability: "),
                        "Win rate setelah losing streak dengan panjang tertentu. ",
                        "Membantu identifikasi 'mean reversion' atau 'momentum' patterns."
                    ])
                ]),
                html.Hr(),
                
                html.H5("🎯 Cara Menggunakan"),
                html.Ol([
                    html.Li("Pilih Target Variable (trade_success, y_hit_1R, atau y_hit_2R)"),
                    html.Li("Atur Confidence Level untuk confidence intervals (default 95%)"),
                    html.Li("Atur Max Streak Length untuk analisis (5-100, default 20)"),
                    html.Li("Atur Min Samples untuk reliability threshold (0=All, atau 20-100)"),
                    html.Li("Klik 'Calculate Sequential Analysis' untuk menjalankan"),
                    html.Li("Review Markov matrix, streak distribution, dan conditional probabilities"),
                    html.Li("Baca insights yang di-generate otomatis"),
                    html.Li("Export hasil untuk dokumentasi")
                ], className="mb-3"),
                html.Hr(),
                
                html.H5("💡 Interpretasi Markov Matrix"),
                html.P(html.Strong("Cara Membaca:")),
                html.Ul([
                    html.Li("Baris = Current trade state (Win atau Loss)"),
                    html.Li("Kolom = Next trade state (Win atau Loss)"),
                    html.Li("Cell value = Probabilitas transisi"),
                    html.Li("Warna hijau = probabilitas tinggi, merah = rendah")
                ], className="mb-2"),
                html.P(html.Strong("Interpretasi:")),
                html.Ul([
                    html.Li([
                        html.Strong("P(Win|Win) ≈ P(Win|Loss): "),
                        "Hasil independent, tidak ada momentum atau mean reversion"
                    ]),
                    html.Li([
                        html.Strong("P(Win|Win) > P(Win|Loss): "),
                        "Momentum effect - winning streaks cenderung berlanjut"
                    ]),
                    html.Li([
                        html.Strong("P(Win|Win) < P(Win|Loss): "),
                        "Mean reversion - setelah win cenderung loss, dan sebaliknya"
                    ])
                ]),
                html.Hr(),
                
                html.H5("📈 Interpretasi Streak Distribution"),
                html.Ul([
                    html.Li("Histogram menunjukkan frekuensi setiap panjang streak"),
                    html.Li("Max streak: Panjang streak terpanjang yang pernah terjadi"),
                    html.Li("Avg streak: Rata-rata panjang streak"),
                    html.Li("Distribusi normal: Streaks random (independent)"),
                    html.Li("Long tail: Ada clustering (momentum atau mean reversion)")
                ]),
                html.Hr(),
                
                html.H5("🎲 Interpretasi Conditional Probability"),
                html.P([
                    "Chart menunjukkan win rate setelah losing streak dengan panjang tertentu. ",
                    "Error bars menunjukkan confidence interval."
                ]),
                html.Ul([
                    html.Li([
                        html.Strong("Win rate meningkat dengan streak length: "),
                        "Mean reversion - setelah banyak loss, probabilitas win meningkat"
                    ]),
                    html.Li([
                        html.Strong("Win rate menurun dengan streak length: "),
                        "Momentum - losing streaks cenderung berlanjut"
                    ]),
                    html.Li([
                        html.Strong("Win rate flat: "),
                        "Independent - panjang losing streak tidak mempengaruhi probabilitas win"
                    ]),
                    html.Li([
                        html.Strong("Wide error bars: "),
                        "Sample size kecil, kurang reliable"
                    ])
                ]),
                html.Hr(),
                
                html.H5("🔧 Tips & Troubleshooting"),
                html.Ul([
                    html.Li([
                        html.Strong("Sample size kecil: "),
                        "Tingkatkan Min Samples atau kumpulkan lebih banyak data"
                    ]),
                    html.Li([
                        html.Strong("Banyak streaks tidak reliable: "),
                        "Kurangi Max Streak Length atau tingkatkan Min Samples"
                    ]),
                    html.Li([
                        html.Strong("Confidence intervals lebar: "),
                        "Normal untuk sample kecil, atau turunkan confidence level"
                    ]),
                    html.Li([
                        html.Strong("Tidak ada pola jelas: "),
                        "Bisa jadi hasil trading memang independent (bagus!)"
                    ])
                ]),
                html.Hr(),
                
                html.H5("📋 Contoh Kasus"),
                dbc.Card([
                    dbc.CardBody([
                        html.P(html.Strong("Kasus 1: Independent Trading"), className="text-success"),
                        html.P([
                            "P(Win|Win) = 0.55, P(Win|Loss) = 0.54. ",
                            "Conditional probabilities flat. ",
                            html.Br(),
                            html.Strong("Interpretasi: "),
                            "Hasil trading independent, tidak ada momentum atau mean reversion. ",
                            "Ini adalah karakteristik trading yang baik."
                        ], className="small mb-3"),
                        
                        html.P(html.Strong("Kasus 2: Momentum Effect"), className="text-info"),
                        html.P([
                            "P(Win|Win) = 0.65, P(Win|Loss) = 0.45. ",
                            "Winning streaks panjang. ",
                            html.Br(),
                            html.Strong("Interpretasi: "),
                            "Ada momentum - wins cenderung diikuti wins. ",
                            "Strategi: Increase position size setelah wins, reduce setelah losses."
                        ], className="small mb-3"),
                        
                        html.P(html.Strong("Kasus 3: Mean Reversion"), className="text-warning"),
                        html.P([
                            "P(Win|Win) = 0.45, P(Win|Loss) = 0.65. ",
                            "Win rate meningkat setelah losing streaks. ",
                            html.Br(),
                            html.Strong("Interpretasi: "),
                            "Mean reversion - setelah losses, probabilitas win meningkat. ",
                            "Strategi: Increase position size setelah losing streaks (martingale-like, hati-hati!)."
                        ], className="small mb-0")
                    ])
                ], className="mb-3"),
                html.Hr(),
                
                html.H5("✅ Action Items"),
                html.Ol([
                    html.Li("Identifikasi apakah trading Anda independent atau ada pola"),
                    html.Li("Jika ada momentum: Capitalize dengan position sizing"),
                    html.Li("Jika ada mean reversion: Adjust strategy accordingly"),
                    html.Li("Monitor patterns over time untuk detect regime changes"),
                    html.Li("Combine dengan Calibration Lab untuk validate probabilities"),
                    html.Li("Document findings dan adjust trading plan")
                ])
            ]),
            dbc.ModalFooter([
                html.P([
                    html.I(className="bi bi-lightbulb me-2"),
                    "Tip: Sequential analysis membantu Anda memahami temporal dependencies dalam trading. ",
                    "Gunakan insights ini untuk optimize position sizing dan risk management."
                ], className="small text-muted me-auto mb-0"),
                dbc.Button("Tutup", id="seq-help-close", color="primary")
            ])
        ], id="seq-help-modal", size="lg", is_open=False)
        
    ], fluid=True)
