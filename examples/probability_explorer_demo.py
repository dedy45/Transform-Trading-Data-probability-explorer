"""
Probability Explorer Frontend Demo
Demonstrates how to integrate the Probability Explorer frontend components
"""
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
import numpy as np

from frontend.layouts import create_probability_explorer_layout
from frontend.callbacks import register_probability_explorer_callbacks

# Create sample data for demonstration
def create_sample_data(n_rows=1000):
    """Create sample merged dataset for testing"""
    np.random.seed(42)
    
    data = {
        # Target variables
        'y_win': np.random.choice([0, 1], n_rows, p=[0.45, 0.55]),
        'y_hit_1R': np.random.choice([0, 1], n_rows, p=[0.6, 0.4]),
        'y_hit_2R': np.random.choice([0, 1], n_rows, p=[0.75, 0.25]),
        'y_future_win_k': np.random.choice([0, 1], n_rows, p=[0.5, 0.5]),
        
        # Features
        'trend_strength_tf': np.random.uniform(0, 1, n_rows),
        'volatility_regime': np.random.choice([0, 1, 2], n_rows),
        'session': np.random.choice([0, 1, 2, 3], n_rows),
        'trend_regime': np.random.choice([0, 1], n_rows),
        'atr_tf_14': np.random.uniform(0.5, 2.0, n_rows),
        'bar_range_over_atr': np.random.uniform(0.3, 1.5, n_rows),
        'entropy': np.random.uniform(0, 1, n_rows),
        'hurst_exponent': np.random.uniform(0.3, 0.7, n_rows),
        
        # Trade data
        'R_multiple': np.random.normal(0.5, 1.5, n_rows),
        'trade_success': np.random.choice([0, 1], n_rows, p=[0.45, 0.55]),
        'net_profit': np.random.normal(50, 200, n_rows),
        'MAE_R': np.random.uniform(0, 1.5, n_rows),
        'MFE_R': np.random.uniform(0, 3.0, n_rows),
        'holding_minutes': np.random.randint(30, 480, n_rows),
        
        # Identifiers
        'Ticket_id': range(1, n_rows + 1),
        'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='15min'),
        'entry_time': pd.date_range('2024-01-01', periods=n_rows, freq='15min'),
        'Type': np.random.choice(['BUY', 'SELL'], n_rows),
        'entry_price': np.random.uniform(1800, 2000, n_rows),
        'ClosePrice': np.random.uniform(1800, 2000, n_rows),
    }
    
    return pd.DataFrame(data)


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.2/font/bootstrap-icons.css"
    ],
    suppress_callback_exceptions=True
)

app.title = "Probability Explorer Demo"

# Create sample data
sample_data = create_sample_data(1000)

# Main layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Probability Explorer Demo", className="text-center my-4"),
            html.P(
                "Interactive demonstration of the Probability Explorer frontend components",
                className="text-center text-muted mb-4"
            )
        ])
    ]),
    
    # Probability Explorer Layout
    create_probability_explorer_layout(),
    
    # Pre-load sample data into store
    dcc.Store(
        id='merged-data-store',
        data=sample_data.to_dict('records')
    )
    
], fluid=True)

# Register callbacks
register_probability_explorer_callbacks(app)


if __name__ == '__main__':
    print("=" * 60)
    print("Probability Explorer Demo")
    print("=" * 60)
    print("\nStarting Dash server...")
    print("Open your browser to: http://localhost:8050")
    print("\nFeatures to try:")
    print("  1. Expand/collapse the filter panel")
    print("  2. Select a target variable (e.g., y_win)")
    print("  3. Select Feature X (e.g., trend_strength_tf)")
    print("  4. Click 'Calculate Probabilities'")
    print("  5. Try selecting Feature Y for 2D heatmap")
    print("  6. Click on bars or heatmap cells to see trade details")
    print("  7. Adjust confidence level and bin size")
    print("  8. Apply filters and see real-time updates")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run_server(
        host='127.0.0.1',
        port=8050,
        debug=True
    )
