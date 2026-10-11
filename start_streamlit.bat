@echo off
echo ============================================================
echo   Starting Microplastic Vision - Streamlit Screening App
echo ============================================================
cd /d "%~dp0backend"
if exist "..\venv\Scripts\streamlit.exe" (
    "..\venv\Scripts\streamlit.exe" run streamlit_app.py
) else (
    python -m streamlit run streamlit_app.py
)
pause
