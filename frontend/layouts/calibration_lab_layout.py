"""
Calibration Lab Layout
Layout for probability calibration assessment
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from frontend.components.calibration_metrics_cards import create_empty_calibration_metrics_cards
from frontend.components.reliability_diagram import create_empty_reliability_diagram
from frontend.components.calibration_histogram import create_empty_calibration_histogram
from frontend.components.calibration_table import create_empty_calibration_table


def create_calibration_lab_layout():
    """
    Create complete Calibration Lab layout
    
    Layout structure:
    - Header with description
    - Calibration metrics cards (Brier Score, ECE, samples, bins)
    - Analysis controls and settings
    - Reliability diagram (calibration plot)
    - Histogram of predicted probabilities
    - Detailed calibration table per bin
    - Insights and recommendations
    
    Requirements: 12.4
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="bi bi-speedometer2 me-2"),
                    "Lab Kalibrasi"
                ], className="mb-2"),
                html.P(
                    "Nilai kualitas kalibrasi prediksi probabilitas. "
                    "Model yang terkalibrasi dengan baik memiliki probabilitas prediksi yang sesuai dengan frekuensi yang diamati. "
                    "Gunakan tool ini untuk memvalidasi dan meningkatkan estimasi probabilitas Anda.",
                    className="text-muted"
                )
            ])
        ], className="mb-3"),
        
        # Calibration Metrics Cards
        dbc.Row([
            dbc.Col([
                html.Div(id='calibration-metrics-cards', children=[create_empty_calibration_metrics_cards()])
            ])
        ], className="mb-3"),
        
        # Info Panel
        dbc.Row([
            dbc.Col([
                html.Div(id='calib-info-panel', children=[
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Alert([
                                html.I(className="bi bi-info-circle me-2"),
                                "Klik tombol 'Hitung Kalibrasi' untuk memulai analisis kalibrasi"
                            ], color="info", className="mb-0")
                        ])
                    ])
                ])
            ])
        ], className="mb-3"),
        
        # Analysis Controls
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Pengaturan Kalibrasi", className="mb-0")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Kolom Probabilitas", className="fw-bold"),
                                dcc.Dropdown(
                                    id='calib-prob-column-dropdown',
                                    options=[
                                        {'label': 'Probabilitas Model (model_prob_win)', 'value': 'model_prob_win'},
                                        {'label': 'Probabilitas Menang Global', 'value': 'prob_global_win'},
                                        {'label': 'Probabilitas Menang Sesi', 'value': 'prob_session_win'},
                                        {'label': 'Prob Menang Kekuatan Tren', 'value': 'prob_trend_strength_win'},
                                        {'label': 'Prob Menang Regime Volatilitas', 'value': 'prob_vol_regime_win'},
                                        {'label': 'Skor Komposit (dinormalisasi)', 'value': 'composite_score_norm'},
                                    ],
                                    value='model_prob_win',
                                    clearable=False,
                                    placeholder="Pilih kolom probabilitas untuk dikalibrasi"
                                )
                            ], md=4),
                            dbc.Col([
                                html.Label("Variabel Target", className="fw-bold"),
                                dcc.Dropdown(
                                    id='calib-target-variable-dropdown',
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
                                html.Label("Jumlah Bin", className="fw-bold"),
                                html.Div([
                                    dcc.Slider(
                                        id='calib-n-bins-slider',
                                        min=5,
                                        max=100,
                                        step=5,
                                        value=10,
                                        marks={
                                            5: '5', 
                                            10: '10', 
                                            20: '20', 
                                            30: '30',
                                            50: '50',
                                            75: '75',
                                            100: '100'
                                        },
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                    html.Small([
                                        html.I(className="bi bi-info-circle me-1"),
                                        "Bins membagi range probabilitas [0,1] untuk analisis detail. ",
                                        "Lebih banyak bins = analisis lebih granular. ",
                                        "Untuk data jutaan, gunakan 50-100 bins."
                                    ], className="text-muted d-block mt-1")
                                ])
                            ], md=3),
                            dbc.Col([
                                html.Label("Strategi Binning", className="fw-bold"),
                                dcc.Dropdown(
                                    id='calib-strategy-dropdown',
                                    options=[
                                        {'label': 'Kuantil (Sampel Sama)', 'value': 'quantile'},
                                        {'label': 'Seragam (Lebar Sama)', 'value': 'uniform'}
                                    ],
                                    value='quantile',
                                    clearable=False
                                )
                            ], md=2)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.ButtonGroup([
                                    dbc.Button(
                                        [html.I(className="bi bi-calculator me-2"), "Hitung Kalibrasi"],
                                        id="calib-calculate-btn",
                                        color="primary",
                                        size="lg"
                                    ),
                                    dbc.Button(
                                        [html.I(className="bi bi-download me-2"), "Ekspor Hasil"],
                                        id="calib-export-btn",
                                        color="success",
                                        size="lg",
                                        outline=True
                                    ),
                                    dbc.Button(
                                        [html.I(className="bi bi-info-circle me-2"), "Panduan Kalibrasi"],
                                        id="calib-help-btn",
                                        color="info",
                                        size="lg",
                                        outline=True
                                    )
                                ], className="w-100")
                            ])
                        ])
                    ])
                ], className="mb-3")
            ])
        ]),
        
        # Reliability Diagram and Histogram
        dbc.Row([
            # Reliability Diagram
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Diagram Reliabilitas", className="mb-0 d-inline"),
                        dbc.Badge("Plot Kalibrasi", color="info", className="ms-2")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-reliability-diagram",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id='reliability-diagram',
                                    figure=create_empty_reliability_diagram(),
                                    config={'displayModeBar': True, 'displaylogo': False}
                                )
                            ]
                        ),
                        html.Div([
                            html.P([
                                html.Strong("Interpretasi: "),
                                "Titik pada garis diagonal menunjukkan kalibrasi sempurna. "
                                "Titik di atas garis berarti model kurang percaya diri (prediksi lebih rendah dari aktual). "
                                "Titik di bawah berarti terlalu percaya diri (prediksi lebih tinggi dari aktual). "
                                "Ukuran titik mewakili jumlah sampel."
                            ], className="text-muted small mb-0")
                        ], className="mt-2")
                    ])
                ], className="mb-3")
            ], md=7),
            
            # Histogram
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Distribusi Probabilitas", className="mb-0 d-inline"),
                        dbc.Badge("Histogram", color="info", className="ms-2")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-calibration-histogram",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id='calibration-histogram',
                                    figure=create_empty_calibration_histogram(),
                                    config={'displayModeBar': True, 'displaylogo': False}
                                )
                            ]
                        ),
                        html.Div([
                            html.P([
                                html.Strong("Interpretasi: "),
                                "Menunjukkan bagaimana probabilitas prediksi didistribusikan. "
                                "Idealnya, prediksi harus mencakup rentang penuh [0, 1]. "
                                "Pengelompokan di dekat 0.5 menunjukkan kepercayaan rendah. "
                                "Pengelompokan di ekstrem (0 atau 1) menunjukkan kepercayaan tinggi."
                            ], className="text-muted small mb-0")
                        ], className="mt-2")
                    ])
                ], className="mb-3")
            ], md=5)
        ]),
        
        # Detailed Calibration Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Detail Kalibrasi per Bin", className="mb-0 d-inline"),
                        dbc.Badge("Statistik Per-Bin", color="info", className="ms-2")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-calibration-table",
                            type="default",
                            children=[
                                html.Div(id='calibration-table', children=[create_empty_calibration_table()])
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
                        html.H5("Wawasan Kalibrasi", className="mb-0 d-inline"),
                        html.I(className="bi bi-lightbulb ms-2")
                    ]),
                    dbc.CardBody([
                        html.Div(id='calib-insights-content', children=[
                            dbc.Alert([
                                html.I(className="bi bi-info-circle me-2"),
                                "Hitung kalibrasi untuk melihat wawasan dan rekomendasi"
                            ], color="info", className="mb-0")
                        ])
                    ])
                ], className="mb-3")
            ])
        ]),
        
        # Hidden stores for data
        dcc.Store(id='calibration-results-store'),
        dcc.Store(id='predicted-probs-store'),
        dcc.Download(id='calib-download-data'),
        
        # Help Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Panduan Kalibrasi")),
            dbc.ModalBody([
                html.H5("📚 Apa itu Kalibrasi?"),
                html.P(
                    "Kalibrasi mengukur seberapa baik prediksi probabilitas sesuai dengan frekuensi aktual. "
                    "Model yang terkalibrasi dengan baik memprediksi probabilitas 70% untuk kejadian yang terjadi 70% dari waktu. "
                    "Ini penting untuk membuat keputusan trading yang informed berdasarkan probabilitas."
                ),
                html.Hr(),
                
                html.H5("📊 Metrik Utama"),
                html.Ul([
                    html.Li([
                        html.Strong("Brier Score: "),
                        "Mean squared error antara prediksi dan hasil aktual. Range [0, 1], semakin rendah semakin baik. ",
                        html.Br(),
                        "• < 0.05 = Excellent (sangat akurat) ",
                        html.Br(),
                        "• 0.05-0.10 = Good (akurat) ",
                        html.Br(),
                        "• 0.10-0.15 = Fair (cukup) ",
                        html.Br(),
                        "• > 0.15 = Poor (perlu perbaikan)"
                    ], className="mb-2"),
                    html.Li([
                        html.Strong("Expected Calibration Error (ECE): "),
                        "Rata-rata tertimbang dari deviasi terhadap kalibrasi sempurna. Range [0, 1], semakin rendah semakin baik. ",
                        "Threshold kualitas sama dengan Brier Score."
                    ], className="mb-2"),
                    html.Li([
                        html.Strong("Reliability Diagram: "),
                        "Visualisasi kalibrasi. Titik pada garis diagonal menunjukkan kalibrasi sempurna. ",
                        "Ukuran marker menunjukkan jumlah sample di bin tersebut."
                    ])
                ]),
                html.Hr(),
                
                html.H5("🎯 Cara Menggunakan"),
                html.Ol([
                    html.Li("Pilih kolom probabilitas yang ingin dianalisis (misal: model_prob_win)"),
                    html.Li("Pilih target variable (trade_success, hit_1R, atau hit_2R)"),
                    html.Li("Atur jumlah bins (5-20, default 10) - lebih banyak bins = analisis lebih detail"),
                    html.Li("Pilih strategi binning: Quantile (sample merata) atau Uniform (range merata)"),
                    html.Li("Klik 'Calculate Calibration' untuk menjalankan analisis"),
                    html.Li("Review reliability diagram, histogram, dan tabel detail"),
                    html.Li("Baca insights dan rekomendasi yang dihasilkan otomatis"),
                    html.Li("Export hasil untuk dokumentasi dan tracking")
                ], className="mb-3"),
                html.Hr(),
                
                html.H5("💡 Interpretasi Chart"),
                html.P(html.Strong("Reliability Diagram:")),
                html.Ul([
                    html.Li("Titik di atas diagonal: Model under-confident (prediksi terlalu rendah)"),
                    html.Li("Titik di bawah diagonal: Model over-confident (prediksi terlalu tinggi)"),
                    html.Li("Titik dekat diagonal: Kalibrasi baik"),
                    html.Li("Ukuran marker: Jumlah sample (lebih besar = lebih reliable)"),
                    html.Li("Warna hijau: Kalibrasi excellent, kuning: fair, merah: poor")
                ], className="mb-3"),
                
                html.P(html.Strong("Histogram Distribusi:")),
                html.Ul([
                    html.Li("Menunjukkan distribusi prediksi probabilitas"),
                    html.Li("Idealnya tersebar di range [0, 1], tidak menumpuk di tengah"),
                    html.Li("Garis merah: mean probability"),
                    html.Li("Garis hijau: median probability"),
                    html.Li("Clustering di 0.5: Model kurang confident"),
                    html.Li("Spread luas: Model confident dengan prediksi beragam")
                ]),
                html.Hr(),
                
                html.H5("🔧 Tips & Troubleshooting"),
                html.Ul([
                    html.Li([
                        html.Strong("Bins kosong: "),
                        "Gunakan quantile binning atau kurangi jumlah bins"
                    ]),
                    html.Li([
                        html.Strong("ECE tinggi: "),
                        "Pertimbangkan recalibration dengan Platt scaling atau isotonic regression"
                    ]),
                    html.Li([
                        html.Strong("Brier Score tinggi: "),
                        "Review feature engineering dan model selection"
                    ]),
                    html.Li([
                        html.Strong("Prediksi terlalu konservatif: "),
                        "Std dev rendah, semua prediksi dekat 0.5 - model kurang discriminative"
                    ]),
                    html.Li([
                        html.Strong("Sample size kecil: "),
                        "Kumpulkan lebih banyak data untuk analisis yang lebih robust (min 100 samples)"
                    ]),
                    html.Li([
                        html.Strong("Systematic bias: "),
                        "Jika semua bins over/under-confident, ada bias sistematis dalam model"
                    ])
                ]),
                html.Hr(),
                
                html.H5("📈 Contoh Kasus"),
                dbc.Card([
                    dbc.CardBody([
                        html.P(html.Strong("Kasus 1: Kalibrasi Excellent"), className="text-success"),
                        html.P([
                            "Brier Score = 0.042, ECE = 0.038. ",
                            "Semua bins dekat diagonal. ",
                            html.Br(),
                            html.Strong("Action: "),
                            "Model siap digunakan, monitor berkala untuk memastikan kalibrasi tetap baik."
                        ], className="small mb-3"),
                        
                        html.P(html.Strong("Kasus 2: Over-Confident"), className="text-warning"),
                        html.P([
                            "Brier Score = 0.125, ECE = 0.118. ",
                            "Banyak bins di bawah diagonal. ",
                            html.Br(),
                            html.Strong("Action: "),
                            "Model terlalu yakin. Gunakan Platt scaling untuk recalibration. ",
                            "Pertimbangkan ensemble dengan model lain."
                        ], className="small mb-3"),
                        
                        html.P(html.Strong("Kasus 3: Under-Confident"), className="text-info"),
                        html.P([
                            "Brier Score = 0.095, ECE = 0.089. ",
                            "Bins di atas diagonal, std dev rendah. ",
                            html.Br(),
                            html.Strong("Action: "),
                            "Model terlalu konservatif. Review feature engineering, ",
                            "tambahkan features yang lebih discriminative."
                        ], className="small mb-0")
                    ])
                ], className="mb-3"),
                html.Hr(),
                
                html.H5("✅ Action Items"),
                html.Ol([
                    html.Li("Jalankan analisis kalibrasi secara berkala (misal: setiap bulan)"),
                    html.Li("Track Brier Score dan ECE over time untuk monitor degradasi"),
                    html.Li("Jika kalibrasi memburuk, retrain model dengan data terbaru"),
                    html.Li("Dokumentasikan hasil kalibrasi untuk audit trail"),
                    html.Li("Gunakan hasil kalibrasi untuk adjust position sizing"),
                    html.Li("Combine dengan Sequential Analysis untuk strategi yang lebih robust")
                ])
            ]),
            dbc.ModalFooter([
                html.P([
                    html.I(className="bi bi-lightbulb me-2"),
                    "Tip: Kalibrasi yang baik adalah fondasi untuk probability-based trading. ",
                    "Gunakan tool ini untuk memvalidasi dan improve model Anda."
                ], className="small text-muted me-auto mb-0"),
                dbc.Button("Tutup", id="calib-help-close", color="primary")
            ])
        ], id="calib-help-modal", size="lg", is_open=False)
        
    ], fluid=True)
