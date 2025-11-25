"""
Composite Score Dashboard Component
Provides UI components for composite score analysis and filtering
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_weight_control_panel():
    """
    Create weight adjustment control panel

    Features:
    - 6 weight sliders for each component
    - Real-time weight sum validation
    - Reset to defaults button
    """
    return dbc.Card([
        dbc.CardHeader(html.H6("Bobot Komponen", className="mb-0")),
        dbc.CardBody([
            # Win Rate Weight
            dbc.Row([
                dbc.Col([
                    html.Label("Win Rate", className="fw-bold small"),
                    dcc.Slider(
                        id='cs-weight-win-rate',
                        min=0,
                        max=1,
                        step=0.05,
                        value=0.30,
                        marks={0: '0%', 0.5: '50%', 1: '100%'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ])
            ], className="mb-2"),

            # Expected R Weight
            dbc.Row([
                dbc.Col([
                    html.Label("Expected R", className="fw-bold small"),
                    dcc.Slider(
                        id='cs-weight-expected-r',
                        min=0,
                        max=1,
                        step=0.05,
                        value=0.25,
                        marks={0: '0%', 0.5: '50%', 1: '100%'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ])
            ], className="mb-2"),

            # Structure Quality Weight
            dbc.Row([
                dbc.Col([
                    html.Label("Kualitas Struktur", className="fw-bold small"),
                    dcc.Slider(
                        id='cs-weight-structure',
                        min=0,
                        max=1,
                        step=0.05,
                        value=0.15,
                        marks={0: '0%', 0.5: '50%', 1: '100%'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ])
            ], className="mb-2"),

            # Time Based Weight
            dbc.Row([
                dbc.Col([
                    html.Label("Berbasis Waktu", className="fw-bold small"),
                    dcc.Slider(
                        id='cs-weight-time',
                        min=0,
                        max=1,
                        step=0.05,
                        value=0.10,
                        marks={0: '0%', 0.5: '50%', 1: '100%'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ])
            ], className="mb-2"),

            # Correlation Weight
            dbc.Row([
                dbc.Col([
                    html.Label("Korelasi", className="fw-bold small"),
                    dcc.Slider(
                        id='cs-weight-correlation',
                        min=0,
                        max=1,
                        step=0.05,
                        value=0.10,
                        marks={0: '0%', 0.5: '50%', 1: '100%'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ])
            ], className="mb-2"),

            # Entry Quality Weight
            dbc.Row([
                dbc.Col([
                    html.Label("Kualitas Entry", className="fw-bold small"),
                    dcc.Slider(
                        id='cs-weight-entry',
                        min=0,
                        max=1,
                        step=0.05,
                        value=0.10,
                        marks={0: '0%', 0.5: '50%', 1: '100%'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ])
            ], className="mb-3"),

            # Weight Sum Validation
            html.Div(id='cs-weight-sum-validation', className="mb-3"),

            # Action Buttons
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className="bi bi-arrow-clockwise me-2"), "Reset Default"],
                    id="cs-reset-weights-btn",
                    size="sm",
                    color="secondary",
                    outline=True
                ),
                dbc.Button(
                    [html.I(className="bi bi-calculator me-2"), "Hitung Ulang"],
                    id="cs-recalculate-btn",
                    size="sm",
                    color="primary"
                )
            ], className="w-100")
        ])
    ])


def create_composite_score_gauge():
    """
    Create composite score gauge visualization (0-100)

    Features:
    - Gauge chart showing average composite score
    - Color-coded by score level (red/yellow/green)
    - Score distribution indicator
    """
    return dbc.Card([
        dbc.CardHeader(html.H6("Skor Komposit Rata-rata", className="mb-0")),
        dbc.CardBody([
            dcc.Loading(
                id="loading-cs-gauge",
                type="default",
                children=[
                    dcc.Graph(
                        id='cs-gauge-chart',
                        config={'displayModeBar': False},
                        style={'height': '250px'}
                    )
                ]
            ),
            html.Div(id='cs-gauge-stats', className="text-center mt-2")
        ])
    ])


def create_component_breakdown_radar():
    """
    Create radar chart showing component breakdown

    Features:
    - 6-axis radar chart for each component
    - Shows average score for each component
    - Interactive hover info
    """
    return dbc.Card([
        dbc.CardHeader(html.H6("Breakdown Komponen", className="mb-0")),
        dbc.CardBody([
            dcc.Loading(
                id="loading-cs-radar",
                type="default",
                children=[
                    dcc.Graph(
                        id='cs-radar-chart',
                        config={'displayModeBar': False},
                        style={'height': '350px'}
                    )
                ]
            )
        ])
    ])


def create_score_distribution_histogram():
    """
    Create histogram showing score distribution

    Features:
    - Histogram of composite scores
    - Color-coded by recommendation level
    - Threshold line overlay
    """
    return dbc.Card([
        dbc.CardHeader(html.H6("Distribusi Skor", className="mb-0")),
        dbc.CardBody([
            dcc.Loading(
                id="loading-cs-histogram",
                type="default",
                children=[
                    dcc.Graph(
                        id='cs-histogram-chart',
                        config={'displayModeBar': True, 'displaylogo': False},
                        style={'height': '300px'}
                    )
                ]
            )
        ])
    ])


def create_threshold_optimizer():
    """
    Create threshold optimizer with slider and metrics

    Features:
    - Threshold slider (0-100)
    - Real-time metrics update (win rate, expectancy, trade count)
    - Recommendation based on threshold
    """
    return dbc.Card([
        dbc.CardHeader(html.H6("Optimasi Threshold", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Threshold Skor Minimum", className="fw-bold"),
                    dcc.Slider(
                        id='cs-threshold-slider',
                        min=0,
                        max=100,
                        step=5,
                        value=50,
                        marks={i: str(i) for i in range(0, 101, 20)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ])
            ], className="mb-3"),

            # Filtered Metrics Display
            html.Div(id='cs-threshold-metrics', children=[
                dbc.Alert(
                    "Sesuaikan threshold untuk melihat metrik trading yang difilter",
                    color="info",
                    className="mb-0"
                )
            ])
        ])
    ])


def create_backtest_results_table():
    """
    Create backtest results table

    Features:
    - Table showing performance at different thresholds
    - Columns: threshold, win_rate, expectancy, trade_frequency
    - Highlight optimal threshold
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H6("Hasil Backtest Threshold", className="mb-0 d-inline"),
            dbc.Button(
                [html.I(className="bi bi-play-fill me-2"), "Jalankan Backtest"],
                id="cs-backtest-btn",
                size="sm",
                color="primary",
                className="float-end"
            )
        ]),
        dbc.CardBody([
            dcc.Loading(
                id="loading-cs-backtest",
                type="default",
                children=[
                    html.Div(id='cs-backtest-table', children=[
                        dbc.Alert(
                            "Klik 'Jalankan Backtest' untuk menguji berbagai threshold",
                            color="info",
                            className="mb-0"
                        )
                    ])
                ]
            )
        ])
    ])


def create_recommendation_labels():
    """
    Create recommendation labels display

    Features:
    - Badge counts for each recommendation level
    - STRONG BUY, BUY, NEUTRAL, AVOID
    - Percentage distribution
    """
    return dbc.Card([
        dbc.CardHeader(html.H6("Distribusi Rekomendasi", className="mb-0")),
        dbc.CardBody([
            html.Div(id='cs-recommendation-badges', children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Badge("0", color="success", className="fs-4"),
                            html.Div("STRONG BUY", className="small text-muted mt-1"),
                            html.Div("(≥80)", className="small text-muted")
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            dbc.Badge("0", color="info", className="fs-4"),
                            html.Div("BUY", className="small text-muted mt-1"),
                            html.Div("(60-79)", className="small text-muted")
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            dbc.Badge("0", color="warning", className="fs-4"),
                            html.Div("NEUTRAL", className="small text-muted mt-1"),
                            html.Div("(40-59)", className="small text-muted")
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            dbc.Badge("0", color="danger", className="fs-4"),
                            html.Div("AVOID", className="small text-muted mt-1"),
                            html.Div("(<40)", className="small text-muted")
                        ], className="text-center")
                    ], width=3)
                ])
            ])
        ])
    ])


def create_composite_score_dashboard():
    """
    Create complete composite score dashboard layout

    Layout structure:
    - Weight control panel (left sidebar)
    - Main visualization area:
      - Gauge chart
      - Radar chart
      - Histogram
      - Threshold optimizer
      - Backtest results
      - Recommendation labels
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H4([
                    html.I(className="bi bi-speedometer2 me-2"),
                    "Analisis Skor Komposit"
                ], className="mb-2"),
                html.P(
                    "Kombinasikan multiple indikator probabilitas menjadi satu skor "
                    "master untuk filtering trading",
                    className="text-muted small"
                )
            ])
        ], className="mb-3"),

        # Info Alert
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.I(className="bi bi-info-circle me-2"),
                    html.Strong("Cara Menggunakan: "),
                    "1) Sesuaikan bobot komponen sesuai strategi Anda, "
                    "2) Lihat distribusi skor, 3) Atur threshold untuk filtering, "
                    "4) Jalankan backtest untuk menemukan threshold optimal"
                ], color="info", className="mb-0")
            ])
        ], className="mb-3"),

        # Main Content
        dbc.Row([
            # Left Sidebar - Weight Controls
            dbc.Col([
                create_weight_control_panel()
            ], md=3),

            # Main Visualization Area
            dbc.Col([
                # Top Row - Gauge and Radar
                dbc.Row([
                    dbc.Col([
                        create_composite_score_gauge()
                    ], md=6),
                    dbc.Col([
                        create_component_breakdown_radar()
                    ], md=6)
                ], className="mb-3"),

                # Middle Row - Histogram and Recommendations
                dbc.Row([
                    dbc.Col([
                        create_score_distribution_histogram()
                    ], md=8),
                    dbc.Col([
                        create_recommendation_labels()
                    ], md=4)
                ], className="mb-3"),

                # Bottom Row - Threshold Optimizer
                dbc.Row([
                    dbc.Col([
                        create_threshold_optimizer()
                    ])
                ], className="mb-3"),

                # Backtest Results
                dbc.Row([
                    dbc.Col([
                        create_backtest_results_table()
                    ])
                ])
            ], md=9)
        ])
    ], fluid=True)
