@echo off
echo ============================================================
echo   FoodAI - Food Recognition and Calorie Estimator
echo ============================================================
cd /d "%~dp0"

echo.
echo [1/3] Checking dependencies...
python -c "import tensorflow, streamlit, plotly" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo [2/3] Checking models...
if not exist "saved_models\ann_model.keras" (
    echo No models found. Training now (takes 5-15 minutes)...
    python bootstrap_models.py
) else (
    echo Models found. Skipping training.
)

echo.
echo [3/3] Launching app...
echo Open your browser at: http://localhost:8501
streamlit run app.py
pause
