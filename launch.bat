@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title NutriVision AI
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "TF_CPP_MIN_LOG_LEVEL=2"

echo ============================================================
echo   NutriVision AI - Food Recognition and Calorie Counter
echo ============================================================
echo.

REM Prefer the Python launcher with 3.11; fall back to python on PATH.
set "PYTHON_CMD="
py -3.11 --version >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.11"
if not defined PYTHON_CMD (
    python --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD goto :no_python

%PYTHON_CMD% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 goto :old_python

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :venv_failed
) else (
    echo [1/4] Virtual environment found.
)

set "VENV_PY=.venv\Scripts\python.exe"

echo [2/4] Installing dependencies...
echo       This can take a few minutes on the first run.
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :install_failed
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :install_failed

echo [3/4] Checking models...
if not exist "saved_models\mobilenet_model.keras" (
    echo.
    echo WARNING: the MobileNet food classifier is missing.
    echo The app can still open, but recognition will be weaker.
    echo To train it:
    echo   "%VENV_PY%" training\download_dataset.py
    echo   "%VENV_PY%" training\train_mobilenet_linear.py
    echo.
)

echo [4/4] Launching NutriVision AI...
echo.
echo Open http://localhost:8501 in your browser.
echo Press Ctrl+C in this window to stop the app.
echo.
"%VENV_PY%" -m streamlit run app.py
echo.
echo App stopped.
pause
exit /b 0

:no_python
echo.
echo Python was not found.
echo Install 64-bit Python 3.11 from:
echo   https://www.python.org/downloads/windows/
echo During installation, tick "Add python.exe to PATH".
echo Then run launch.bat again.
echo.
pause
exit /b 1

:old_python
echo.
echo Python 3.11 or newer is required.
echo Install 64-bit Python 3.11 from:
echo   https://www.python.org/downloads/windows/
echo Then run launch.bat again.
echo.
pause
exit /b 1

:venv_failed
echo.
echo Could not create the virtual environment.
echo Check that you have write permission in this folder.
echo.
pause
exit /b 1

:install_failed
echo.
echo Dependency installation failed.
echo Check your internet connection, Python version, and the messages above.
echo If TensorFlow failed, install the Microsoft Visual C++ Redistributable:
echo   https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.
pause
exit /b 1
