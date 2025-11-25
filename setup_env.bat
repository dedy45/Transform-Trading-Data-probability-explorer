@echo off
REM Setup conda environment for Trading Probability Explorer

echo ========================================
echo Trading Probability Explorer Setup
echo ========================================
echo.

REM Set environment name
set ENV_NAME=trading-prob-explorer

echo Creating conda environment: %ENV_NAME%
echo.

REM Create conda environment with Python 3.10
call conda create -n %ENV_NAME% python=3.10 -y

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create conda environment
    pause
    exit /b 1
)

echo.
echo Activating environment...
call conda activate %ENV_NAME%

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate conda environment
    pause
    exit /b 1
)

echo.
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To activate the environment, run:
echo   conda activate %ENV_NAME%
echo.
echo To launch the application, run:
echo   launch_app.bat
echo.
pause
