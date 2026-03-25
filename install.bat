@echo off
:: One-click installer for YOLO Real-Time Face Recognition
:: Works on Windows

echo ==================================================
echo   YOLO Real-Time Face Recognition - Installer
echo ==================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version') do set PY_VERSION=%%v
echo [OK] Python %PY_VERSION% detected

:: Create virtual environment
echo.
echo [*] Creating virtual environment...
python -m venv venv
echo [OK] Virtual environment created

:: Activate and install
echo.
echo [*] Installing dependencies (this may take a few minutes)...
call venv\Scripts\activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo [OK] Dependencies installed

echo.
echo ==================================================
echo   Installation complete!
echo   Run the app with:
echo.
echo     venv\Scripts\activate
echo     python main.py
echo ==================================================
pause
