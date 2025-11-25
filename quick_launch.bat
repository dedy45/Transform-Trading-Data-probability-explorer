@echo off
REM Quick Launch - Trading Probability Explorer
REM Langsung jalankan aplikasi dengan auto-open browser

cls
echo ========================================================================
echo    TRADING PROBABILITY EXPLORER - QUICK LAUNCH
echo ========================================================================
echo.
echo Starting application...
echo Browser will open automatically in 3 seconds
echo.
echo Access: http://localhost:8050
echo Press Ctrl+C to stop the server
echo.
echo ========================================================================
echo.

REM Start browser in background after 3 seconds
start /B cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8050"

REM Start the application
python app.py

echo.
echo Server stopped.
pause
