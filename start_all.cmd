@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv not found. Run setup.cmd first.
  exit /b 1
)

start "Disaster Backend" cmd /k "%~dp0start_backend.cmd"
start "Disaster Frontend" cmd /k "%~dp0start_frontend.cmd"

echo Backend + Frontend launch commands started.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173

