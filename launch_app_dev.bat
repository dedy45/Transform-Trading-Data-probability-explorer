@echo off
REM Launch Trading Probability Explorer in DEVELOPMENT MODE

echo ========================================
echo Trading Probability Explorer
echo DEVELOPMENT MODE
echo ========================================
echo.

REM Set environment name
set ENV_NAME=trading-prob-explorer

echo Activating conda environment: %ENV_NAME%
call conda activate %ENV_NAME%

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate conda environment
    echo Please run setup_env.bat first
    pause
    exit /b 1
)

echo.
echo ========================================
echo WARNING: DEVELOPMENT MODE
echo ========================================
echo - Hot-reload is ENABLED
echo - Code changes will trigger auto-reload
echo - Data may be lost on reload
echo - Use for development only!
echo ========================================
echo.
echo Starting Dash application...
echo Browser will open automatically in 3 seconds
echo Application URL: http://127.0.0.1:8050
echo.
echo Press Ctrl+C to stop the application
echo.

REM Start browser in background after 3 seconds
start /B cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8050"

REM Start the application
python app.py

REM Deactivate environment on exit
call conda deactivate

pause
