@echo off
REM Start Prism on Windows. Double-click this file, or run it from a terminal.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating a virtual environment in .venv ...
    py -3 -m venv .venv || python -m venv .venv
    if errorlevel 1 goto :failed
    echo Installing dependencies. This takes a minute the first time.
    ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
)

echo Starting Prism. Your browser should open at http://localhost:8501
echo Press Ctrl+C in this window to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py
goto :eof

:failed
echo.
echo Setup failed. Check that Python 3.10 or newer is installed and on your PATH.
echo You can download it from https://www.python.org/downloads/windows/
pause
