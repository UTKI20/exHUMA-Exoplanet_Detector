@echo off
setlocal
echo ----------------------------------------------------
echo    ANTIGRAVITY: exHUMA - MISSION LAUNCH SEQUENCE
echo ----------------------------------------------------
echo.

REM Switch to the directory where this script is located
pushd "%~dp0"

REM Set path to the verified virtual environment python
REM Relative to this script (d:\DUhacks\exHUMA), venv is in ..\.venv
set PYTHON_EXE=..\.venv\Scripts\python.exe

echo [1/3] 🔍 Verifying Flight Systems (Python Environment)...
echo       Target: %PYTHON_EXE%
if exist "%PYTHON_EXE%" (
    echo       Status: ONLINE
) else (
    echo       Status: CRITICAL FAILURE
    echo       ERROR: Virtual environment not found at %PYTHON_EXE%
    echo       Current Dir: %CD%
    pause
    popd
    exit /b
)

echo.
echo [2/3] 📦 Loading Payload (Dependencies)...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo       [WARNING]: Dependency installation reported errors. Attempting launch anyway...
) else (
    echo       Status: LOADED
)

echo.
echo [3/3] 🚀 IGNITION sequence start...
echo       Launching Streamlit...
echo.
"%PYTHON_EXE%" -m streamlit run app.py

pause
popd
