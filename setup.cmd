@echo off
setlocal
cd /d "%~dp0"

echo [1/5] Checking Python 3.11...
py -3.11 -V >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python 3.11 is not installed or not available via py launcher.
  echo Install with: winget install Python.Python.3.11
  exit /b 1
)

echo [2/5] Ensuring virtual environment...
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo ERROR: Failed to create .venv
    exit /b 1
  )
)

echo [3/5] Installing backend dependencies...
python -m pip --python ".\.venv\Scripts\python.exe" install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1
python -m pip --python ".\.venv\Scripts\python.exe" install -r ".\requirements.txt"
if errorlevel 1 exit /b 1

echo [4/5] Installing frontend dependencies...
call npm.cmd ci
if errorlevel 1 (
  call npm.cmd install
)
if errorlevel 1 exit /b 1

echo [5/5] Verifying core toolchain...
".\.venv\Scripts\python.exe" -c "import fastapi,uvicorn,numpy,pandas,sklearn,sqlalchemy,pymysql,httpx,shap; print('Python deps OK')"
if errorlevel 1 exit /b 1

echo.
echo Setup complete.
echo Next: run start_backend.cmd and start_frontend.cmd
exit /b 0
