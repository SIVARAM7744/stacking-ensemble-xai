@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv missing. Run setup.cmd first.
  exit /b 1
)

echo Rebuilding model artefacts with current local environment...
cd /d "%~dp0ml_pipeline"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set MPLCONFIGDIR=%CD%\.mplconfig
..\.\.venv\Scripts\python.exe train_pipeline.py
if errorlevel 1 (
  echo ERROR: training pipeline failed.
  exit /b 1
)

echo Syncing rebuilt generic models into disaster folders...
if not exist "models_tuned_v4_flood" mkdir "models_tuned_v4_flood"
if not exist "models_tuned_v3_earthquake" mkdir "models_tuned_v3_earthquake"

copy /y "models\preprocessing.pkl" "models_tuned_v4_flood\preprocessing.pkl" >nul
copy /y "models\rf_model.pkl" "models_tuned_v4_flood\rf_model.pkl" >nul
copy /y "models\gb_model.pkl" "models_tuned_v4_flood\gb_model.pkl" >nul
copy /y "models\svm_model.pkl" "models_tuned_v4_flood\svm_model.pkl" >nul
copy /y "models\stacking_model.pkl" "models_tuned_v4_flood\stacking_model.pkl" >nul

copy /y "models\preprocessing.pkl" "models_tuned_v3_earthquake\preprocessing.pkl" >nul
copy /y "models\rf_model.pkl" "models_tuned_v3_earthquake\rf_model.pkl" >nul
copy /y "models\gb_model.pkl" "models_tuned_v3_earthquake\gb_model.pkl" >nul
copy /y "models\svm_model.pkl" "models_tuned_v3_earthquake\svm_model.pkl" >nul
copy /y "models\stacking_model.pkl" "models_tuned_v3_earthquake\stacking_model.pkl" >nul

cd /d "%~dp0backend\app"
echo Verifying model loading...
"..\..\.venv\Scripts\python.exe" -c "from model_loader import load_models,get_loaded_model_types; load_models(); print(get_loaded_model_types())"
if errorlevel 1 (
  echo ERROR: model verification failed.
  exit /b 1
)

echo.
echo Model repair complete.
echo Restart backend now: start_backend.cmd
exit /b 0

