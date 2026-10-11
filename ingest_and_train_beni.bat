@echo off
echo ============================================================
echo   Microplastic Vision - Ingest BENI Hugging Face Dataset
echo ============================================================
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist backend\venv\Scripts\activate.bat (
    call backend\venv\Scripts\activate.bat
)
cd backend
python ingest_taiamiti.py
pause
