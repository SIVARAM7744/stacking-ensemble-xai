@echo off
setlocal
cd /d "%~dp0"

echo ===== Disaster Project Doctor =====
echo Project path: %cd%
echo.

echo [1] Path sanity
echo %cd% | find "#" >nul
if %errorlevel%==0 (
  echo WARN: Path contains '#'. Frontend Vite may fail in this folder.
  echo      Use start_frontend.cmd with auto runtime mirror, or move project to a clean path.
) else (
  echo OK: Path has no '#'
)
echo.

echo [2] Python venv
if exist ".venv\Scripts\python.exe" (
  echo OK: .venv exists
  ".\.venv\Scripts\python.exe" -V
) else (
  echo FAIL: .venv missing. Run setup.cmd
)
echo.

echo [3] Core Python packages
if exist ".venv\Scripts\python.exe" (
  ".\.venv\Scripts\python.exe" -c "import fastapi,uvicorn,numpy,pandas,sklearn,sqlalchemy,pymysql,httpx; print('OK: python deps import')"
  if errorlevel 1 echo FAIL: Python dependencies incomplete. Run setup.cmd
)
echo.

echo [4] Frontend dependencies
if exist "node_modules" (
  echo OK: node_modules exists
) else (
  echo FAIL: node_modules missing. Run setup.cmd or npm.cmd install
)
echo.

echo [5] Model artefacts
if exist "ml_pipeline\models_tuned_v4_flood\stacking_model.pkl" (
  echo OK: Flood tuned model set found
) else (
  echo WARN: Flood tuned v4 model folder missing or incomplete
)
if exist "ml_pipeline\models_tuned_v3_earthquake\stacking_model.pkl" (
  echo OK: Earthquake tuned model set found
) else (
  echo WARN: Earthquake tuned v3 model folder missing or incomplete
)
if exist "ml_pipeline\models\stacking_model.pkl" (
  echo OK: Generic model fallback found
) else (
  echo WARN: Generic model fallback missing
)
echo.

echo [6] Backend .env
if exist "backend\.env" (
  echo OK: backend\.env present
) else (
  echo WARN: backend\.env missing. Copy backend\.env.example to backend\.env
)
echo.

echo Done.
exit /b 0
