@echo off
REM Run the Document Q&A API with Python 3.12 venv
cd /d "%~dp0"
if exist ".venv312\Scripts\python.exe" (
    ".venv312\Scripts\python.exe" chatbot_api.py
) else (
    echo Python 3.12 venv not found. Create it with:
    echo   py -3.12 -m venv .venv312
    echo   .venv312\Scripts\pip install -r requirements.txt
    py -3.12 -c "import sys; sys.exit(1)" 2>nul || echo Then run: py -3.12 -m venv .venv312
    pause
)
