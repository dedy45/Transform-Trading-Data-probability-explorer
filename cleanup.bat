@echo off
REM Cleanup conda environment for Trading Probability Explorer

echo ========================================
echo Trading Probability Explorer Cleanup
echo ========================================
echo.

REM Set environment name
set ENV_NAME=trading-prob-explorer

echo This will remove the conda environment: %ENV_NAME%
echo.
set /p CONFIRM="Are you sure? (Y/N): "

if /i "%CONFIRM%" NEQ "Y" (
    echo Cleanup cancelled
    pause
    exit /b 0
)

echo.
echo Deactivating environment if active...
call conda deactivate

echo.
echo Removing conda environment: %ENV_NAME%
call conda env remove -n %ENV_NAME% -y

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to remove conda environment
    pause
    exit /b 1
)

echo.
echo ========================================
echo Cleanup completed successfully!
echo ========================================
echo.
pause
