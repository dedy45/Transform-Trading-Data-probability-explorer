@echo off
REM Trading Probability Explorer - Interactive Launcher
REM Enhanced with dependency checking and menu system

setlocal enabledelayedexpansion

:MENU
cls
echo ========================================================================
echo    TRADING PROBABILITY EXPLORER - LAUNCHER
echo ========================================================================
echo.
echo    [1] Check Dependencies
echo    [2] Verify Integration
echo    [3] Install Missing Dependencies
echo    [4] Run Application (Development Mode)
echo    [5] Run Application (Production Mode)
echo    [6] Run Tests
echo    [7] View Documentation
echo    [8] Open Data Folder
echo    [9] Clean Cache
echo    [0] Exit
echo.
echo ========================================================================
echo.

set /p choice="Select option (0-9): "

if "%choice%"=="1" goto CHECK_DEPS
if "%choice%"=="2" goto VERIFY_INT
if "%choice%"=="3" goto INSTALL_DEPS
if "%choice%"=="4" goto RUN_DEV
if "%choice%"=="5" goto RUN_PROD
if "%choice%"=="6" goto RUN_TESTS
if "%choice%"=="7" goto VIEW_DOCS
if "%choice%"=="8" goto OPEN_DATA
if "%choice%"=="9" goto CLEAN_CACHE
if "%choice%"=="0" goto EXIT

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto MENU

:CHECK_DEPS
cls
echo ========================================================================
echo    CHECKING DEPENDENCIES
echo ========================================================================
echo.
python check_dependencies.py
echo.
echo ========================================================================
pause
goto MENU

:VERIFY_INT
cls
echo ========================================================================
echo    VERIFYING INTEGRATION
echo ========================================================================
echo.
python verify_integration.py
echo.
echo ========================================================================
pause
goto MENU

:INSTALL_DEPS
cls
echo ========================================================================
echo    INSTALLING DEPENDENCIES
echo ========================================================================
echo.
echo Installing from requirements.txt...
echo.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo ========================================================================
echo Installation complete!
echo.
pause
goto MENU

:RUN_DEV
cls
echo ========================================================================
echo    RUNNING APPLICATION - DEVELOPMENT MODE
echo ========================================================================
echo.
echo Checking dependencies first...
python check_dependencies.py
if errorlevel 1 (
    echo.
    echo ERROR: Missing dependencies detected!
    echo Please install dependencies first (Option 3)
    echo.
    pause
    goto MENU
)
echo.
echo Verifying integration...
python verify_integration.py
if errorlevel 1 (
    echo.
    echo ERROR: Integration verification failed!
    echo Please check the errors above.
    echo.
    pause
    goto MENU
)
echo.
echo ========================================================================
echo Starting application in DEVELOPMENT mode...
echo.
echo Opening browser at: http://localhost:8050
echo Press Ctrl+C to stop the server
echo ========================================================================
echo.

REM Start browser in background after 3 seconds
start /B cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8050"

REM Start the application
python app.py

echo.
echo Server stopped.
pause
goto MENU

:RUN_PROD
cls
echo ========================================================================
echo    RUNNING APPLICATION - PRODUCTION MODE
echo ========================================================================
echo.
echo Checking dependencies first...
python check_dependencies.py
if errorlevel 1 (
    echo.
    echo ERROR: Missing dependencies detected!
    echo Please install dependencies first (Option 3)
    echo.
    pause
    goto MENU
)
echo.
echo ========================================================================
echo Starting application in PRODUCTION mode...
echo.
echo Opening browser at: http://localhost:8050
echo Press Ctrl+C to stop the server
echo ========================================================================
echo.

REM Start browser in background after 3 seconds
start /B cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8050"

REM Start the application in production mode
python app.py --production

echo.
echo Server stopped.
pause
goto MENU

:RUN_TESTS
cls
echo ========================================================================
echo    RUNNING TESTS
echo ========================================================================
echo.
if exist tests\ (
    echo Running pytest...
    python -m pytest tests/ -v
) else (
    echo No tests directory found.
    echo Tests will be added in future updates.
)
echo.
echo ========================================================================
pause
goto MENU

:VIEW_DOCS
cls
echo ========================================================================
echo    DOCUMENTATION
echo ========================================================================
echo.
echo Available documentation files:
echo.
echo [1] README.md - Main documentation
echo [2] INTEGRASI_FITUR_BARU_SELESAI.md - Integration guide
echo [3] ROADMAP_SPECIALIST_ACTIONABLE.md - Development roadmap
echo [4] DOCSAI\NEW_FEATURES_USER_GUIDE.md - User guide for new features
echo [5] PANDUAN_LENGKAP.md - Complete guide (Indonesian)
echo [6] Back to main menu
echo.
set /p doc_choice="Select document to open (1-6): "

if "%doc_choice%"=="1" start README.md
if "%doc_choice%"=="2" start INTEGRASI_FITUR_BARU_SELESAI.md
if "%doc_choice%"=="3" start ROADMAP_SPECIALIST_ACTIONABLE.md
if "%doc_choice%"=="4" start DOCSAI\NEW_FEATURES_USER_GUIDE.md
if "%doc_choice%"=="5" start PANDUAN_LENGKAP.md
if "%doc_choice%"=="6" goto MENU

timeout /t 1 >nul
goto VIEW_DOCS

:OPEN_DATA
cls
echo ========================================================================
echo    OPENING DATA FOLDER
echo ========================================================================
echo.
if exist dataraw\ (
    echo Opening dataraw folder...
    start explorer dataraw
) else (
    echo dataraw folder not found!
    echo Creating dataraw folder...
    mkdir dataraw
    start explorer dataraw
)
echo.
timeout /t 2 >nul
goto MENU

:CLEAN_CACHE
cls
echo ========================================================================
echo    CLEANING CACHE
echo ========================================================================
echo.
echo Cleaning Python cache files...
echo.

REM Clean __pycache__ directories
for /d /r %%d in (__pycache__) do (
    if exist "%%d" (
        echo Removing: %%d
        rd /s /q "%%d"
    )
)

REM Clean .pyc files
for /r %%f in (*.pyc) do (
    if exist "%%f" (
        echo Removing: %%f
        del /q "%%f"
    )
)

REM Clean .pytest_cache
if exist .pytest_cache\ (
    echo Removing: .pytest_cache
    rd /s /q .pytest_cache
)

REM Clean .hypothesis
if exist .hypothesis\tmp\ (
    echo Removing: .hypothesis\tmp
    rd /s /q .hypothesis\tmp
)

echo.
echo Cache cleaned successfully!
echo.
pause
goto MENU

:EXIT
cls
echo ========================================================================
echo    EXITING
echo ========================================================================
echo.
echo Thank you for using Trading Probability Explorer!
echo.
timeout /t 2 >nul
exit /b 0
