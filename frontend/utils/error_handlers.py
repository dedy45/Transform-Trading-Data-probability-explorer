"""
Error Handling Utilities

Centralized error handling functions for consistent error messages
and user feedback across all dashboard callbacks.

Validates: Requirements 5.4, 8.1, 8.2, 8.3, 8.4, 8.5
"""

import dash_bootstrap_components as dbc
from dash import html
import logging
import traceback
import os

# Configure logging
logger = logging.getLogger(__name__)


def create_file_not_found_error(file_path):
    """
    Create error alert for file not found.
    
    Validates: Requirements 8.1
    Property 21: Error Message Specificity
    
    Parameters:
    -----------
    file_path : str
        Path to the file that was not found
        
    Returns:
    --------
    dbc.Alert
        Alert component with specific error message
    """
    return dbc.Alert([
        html.I(className="bi bi-exclamation-triangle me-2"),
        html.Strong("File Tidak Ditemukan: "),
        html.Br(),
        html.Small(f"Path: {file_path}"),
        html.Hr(),
        html.P([
            "Pastikan file ada di folder ",
            html.Code("dataraw/"),
            " dan path sudah benar."
        ], className="mb-0")
    ], color="danger")


def create_invalid_csv_error(file_path, error_details):
    """
    Create error alert for invalid CSV format.
    
    Validates: Requirements 8.1
    Property 21: Error Message Specificity
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    error_details : str
        Details about the parsing error
        
    Returns:
    --------
    dbc.Alert
        Alert component with specific error message
    """
    return dbc.Alert([
        html.I(className="bi bi-file-earmark-x me-2"),
        html.Strong("Format CSV Tidak Valid: "),
        html.Br(),
        html.Small(f"File: {os.path.basename(file_path)}"),
        html.Hr(),
        html.P(f"Detail error: {error_details}", className="small"),
        html.P([
            "Pastikan file adalah CSV valid dengan delimiter yang benar ",
            "(comma, semicolon, atau tab)"
        ], className="mb-0 small")
    ], color="danger")


def create_missing_columns_error(missing_columns, file_path=None):
    """
    Create error alert for missing required columns.
    
    Validates: Requirements 5.4, 8.2
    Property 16: Column Validation Completeness
    
    Parameters:
    -----------
    missing_columns : list
        List of missing column names
    file_path : str, optional
        Path to the file (if available)
        
    Returns:
    --------
    dbc.Alert
        Alert component listing all missing columns
    """
    return dbc.Alert([
        html.H5("Data Validation Failed", className="alert-heading"),
        html.P("Kolom yang diperlukan tidak ditemukan:"),
        html.Ul([html.Li(f"❌ {col}") for col in missing_columns]),
        html.Hr(),
        html.P([
            html.Strong("Solusi: "),
            "Pastikan CSV Anda memiliki semua kolom yang diperlukan. ",
            "Periksa nama kolom dan pastikan tidak ada typo."
        ], className="mb-0")
    ] + ([html.P(f"File: {os.path.basename(file_path)}", className="small text-muted mt-2")] if file_path else []),
    color="danger")


def create_empty_data_message(context="analysis"):
    """
    Create message for empty DataFrame.
    
    Validates: Requirements 8.3
    
    Parameters:
    -----------
    context : str
        Context of where the empty data was encountered
        
    Returns:
    --------
    dbc.Alert
        Alert component with clear message
    """
    return dbc.Alert([
        html.I(className="bi bi-inbox me-2"),
        html.Strong("Tidak Ada Data: "),
        f"Tidak ada data tersedia untuk {context}.",
        html.Hr(),
        html.P("Silakan muat data trading terlebih dahulu.", className="mb-0 small")
    ], color="info")


def create_insufficient_data_message(current_count, minimum_required, context="analysis"):
    """
    Create message for insufficient data.
    
    Validates: Requirements 8.3
    
    Parameters:
    -----------
    current_count : int
        Current number of data points
    minimum_required : int
        Minimum required data points
    context : str
        Context of the analysis
        
    Returns:
    --------
    dbc.Alert
        Alert component with clear message
    """
    return dbc.Alert([
        html.I(className="bi bi-exclamation-circle me-2"),
        html.Strong("Data Tidak Cukup: "),
        f"Hanya {current_count} data tersedia untuk {context}.",
        html.Hr(),
        html.P([
            f"Minimum diperlukan: ",
            html.Strong(f"{minimum_required} data points"),
            ". Tambahkan lebih banyak data untuk analisis yang akurat."
        ], className="mb-0 small")
    ], color="warning")


def create_calculation_error_alert(error_message, context="calculation"):
    """
    Create error alert for calculation failures.
    
    Validates: Requirements 8.3
    
    Parameters:
    -----------
    error_message : str
        Error message from the exception
    context : str
        Context of the calculation
        
    Returns:
    --------
    dbc.Alert
        Alert component with error details
    """
    return dbc.Alert([
        html.I(className="bi bi-calculator me-2"),
        html.Strong(f"Error Perhitungan {context.title()}: "),
        html.Br(),
        html.Small(f"Detail: {error_message}"),
        html.Hr(),
        html.P([
            "Coba reload data atau hubungi support jika masalah berlanjut."
        ], className="mb-0 small")
    ], color="warning")


def create_visualization_error_placeholder(error_message=None):
    """
    Create placeholder for visualization errors.
    
    Validates: Requirements 8.3
    
    Parameters:
    -----------
    error_message : str, optional
        Specific error message
        
    Returns:
    --------
    dbc.Alert
        Alert component for visualization error
    """
    return dbc.Alert([
        html.I(className="bi bi-graph-down me-2"),
        html.Strong("Tidak Dapat Membuat Visualisasi: "),
        html.Br(),
        html.Small(error_message if error_message else "Data mungkin dalam format yang tidak sesuai."),
        html.Hr(),
        html.P("Coba refresh halaman atau muat ulang data.", className="mb-0 small")
    ], color="warning")


def create_success_message(row_count, data_type="trade"):
    """
    Create success message for data loading.
    
    Validates: Requirements 8.4, 8.5
    Property 22: Success Feedback Accuracy
    
    Parameters:
    -----------
    row_count : int
        Number of rows loaded
    data_type : str
        Type of data loaded (trade, feature, etc.)
        
    Returns:
    --------
    dbc.Alert
        Success alert with row count
    """
    return dbc.Alert([
        html.I(className="bi bi-check-circle me-2"),
        html.Strong("Data Berhasil Dimuat: "),
        f"{row_count:,} {data_type} records loaded successfully."
    ], color="success", dismissable=True)


def create_loading_spinner(message="Memproses data..."):
    """
    Create loading spinner component.
    
    Validates: Requirements 8.4, 8.5
    
    Parameters:
    -----------
    message : str
        Loading message to display
        
    Returns:
    --------
    html.Div
        Div containing spinner and message
    """
    return html.Div([
        html.Div([
            html.Span(className="spinner-border spinner-border-sm me-2"),
            html.Span(message)
        ], className="text-center text-muted")
    ], className="p-4")


def create_completion_message(operation, duration_seconds=None):
    """
    Create completion message for long operations.
    
    Validates: Requirements 8.4, 8.5
    
    Parameters:
    -----------
    operation : str
        Name of the operation completed
    duration_seconds : float, optional
        Duration of the operation in seconds
        
    Returns:
    --------
    dbc.Alert
        Success alert with completion message
    """
    duration_text = f" ({duration_seconds:.1f}s)" if duration_seconds else ""
    
    return dbc.Alert([
        html.I(className="bi bi-check-circle me-2"),
        html.Strong(f"{operation} Selesai"),
        duration_text
    ], color="success", dismissable=True, duration=4000)


def validate_required_columns(df, required_columns, file_path=None):
    """
    Validate that DataFrame has all required columns.
    
    Validates: Requirements 5.4, 8.2
    Property 16: Column Validation Completeness
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list
        List of required column names
    file_path : str, optional
        Path to the file (for error message)
        
    Returns:
    --------
    tuple
        (is_valid: bool, error_alert: dbc.Alert or None)
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        error_alert = create_missing_columns_error(missing_columns, file_path)
        return False, error_alert
    
    return True, None


def log_error(error, context="Unknown", include_traceback=True):
    """
    Log error with context and optional traceback.
    
    Parameters:
    -----------
    error : Exception
        The exception to log
    context : str
        Context where the error occurred
    include_traceback : bool
        Whether to include full traceback
    """
    error_msg = f"Error in {context}: {str(error)}"
    logger.error(error_msg)
    
    if include_traceback:
        logger.error(traceback.format_exc())


def handle_data_loading_error(error, file_path):
    """
    Handle data loading errors and return appropriate alert.
    
    Validates: Requirements 5.4, 8.1, 8.2
    Property 21: Error Message Specificity
    
    Parameters:
    -----------
    error : Exception
        The exception that occurred
    file_path : str
        Path to the file being loaded
        
    Returns:
    --------
    dbc.Alert
        Appropriate error alert based on error type
    """
    log_error(error, f"Data loading from {file_path}")
    
    error_type = type(error).__name__
    error_str = str(error)
    
    if "FileNotFoundError" in error_type or "No such file" in error_str:
        return create_file_not_found_error(file_path)
    elif "ParserError" in error_type or "CSV" in error_str:
        return create_invalid_csv_error(file_path, error_str)
    elif "KeyError" in error_type:
        # Extract column name from KeyError
        missing_col = error_str.strip("'\"")
        return create_missing_columns_error([missing_col], file_path)
    else:
        # Generic error
        return dbc.Alert([
            html.I(className="bi bi-exclamation-triangle me-2"),
            html.Strong("Error Loading Data: "),
            html.Br(),
            html.Small(f"File: {os.path.basename(file_path)}"),
            html.Br(),
            html.Small(f"Error: {error_str}"),
            html.Hr(),
            html.P("Periksa format file dan coba lagi.", className="mb-0 small")
        ], color="danger")


def handle_calculation_error(error, calculation_name, preserve_previous=True):
    """
    Handle calculation errors and return appropriate response.
    
    Validates: Requirements 8.3
    
    Parameters:
    -----------
    error : Exception
        The exception that occurred
    calculation_name : str
        Name of the calculation
    preserve_previous : bool
        Whether to preserve previous results (return no_update)
        
    Returns:
    --------
    Alert or no_update
        Error alert or no_update to preserve state
    """
    from dash import no_update
    
    log_error(error, f"{calculation_name} calculation")
    
    if preserve_previous:
        # Log error but preserve previous state
        logger.warning(f"Preserving previous state after error in {calculation_name}")
        return no_update
    else:
        # Return error alert
        return create_calculation_error_alert(str(error), calculation_name)


def handle_visualization_error(error, chart_name):
    """
    Handle visualization errors and return placeholder.
    
    Validates: Requirements 8.3
    
    Parameters:
    -----------
    error : Exception
        The exception that occurred
    chart_name : str
        Name of the chart/visualization
        
    Returns:
    --------
    dbc.Alert or plotly.Figure
        Error placeholder
    """
    log_error(error, f"{chart_name} visualization")
    
    return create_visualization_error_placeholder(f"Error creating {chart_name}: {str(error)}")


# Export all functions
__all__ = [
    'create_file_not_found_error',
    'create_invalid_csv_error',
    'create_missing_columns_error',
    'create_empty_data_message',
    'create_insufficient_data_message',
    'create_calculation_error_alert',
    'create_visualization_error_placeholder',
    'create_success_message',
    'create_loading_spinner',
    'create_completion_message',
    'validate_required_columns',
    'log_error',
    'handle_data_loading_error',
    'handle_calculation_error',
    'handle_visualization_error'
]
