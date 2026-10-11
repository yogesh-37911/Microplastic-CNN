@echo off
echo ============================================================
echo   Starting Microplastic Vision - Backend API Server
echo ============================================================
if exist venv\Scripts\activate.bat (
    echo Activating venv from root...
    call venv\Scripts\activate.bat
) else if exist backend\venv\Scripts\activate.bat (
    echo Activating venv from backend...
    call backend\venv\Scripts\activate.bat
)
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
