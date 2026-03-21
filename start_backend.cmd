@echo off
setlocal
cd /d "%~dp0backend\app"

if not exist "..\..\.venv\Scripts\python.exe" (
  echo ERROR: .venv not found. Run setup.cmd first.
  exit /b 1
)

echo Starting backend on http://localhost:8000 ...
"..\..\.venv\Scripts\python.exe" -c "from model_loader import load_models,get_loaded_model_types; load_models(); print(get_loaded_model_types())" >nul 2>nul
if errorlevel 1 (
  echo ERROR: Model precheck failed. Backend would start with models not loaded.
  echo Run: repair_models.cmd
  exit /b 1
)
"..\..\.venv\Scripts\python.exe" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
