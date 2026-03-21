@echo off
setlocal EnableDelayedExpansion
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

echo %ROOT% | find "#" >nul
if %errorlevel%==0 (
  set "BASE_HOME=%USERPROFILE%"
  if "%BASE_HOME%"=="" set "BASE_HOME=%HOMEDRIVE%%HOMEPATH%"
  if "%BASE_HOME%"=="" set "BASE_HOME=%TEMP%"
  if "%BASE_HOME%"=="" set "BASE_HOME=C:\Users\Public"
  set "RUNTIME=!BASE_HOME!\disaster_full_stack_runtime"
  set "LOCK_HASH_FILE=!RUNTIME!\.package-lock.hash"
  echo [INFO] Detected '#' in path. Using runtime mirror: !RUNTIME!
  if not exist "!RUNTIME!" mkdir "!RUNTIME!"
  robocopy "%ROOT%" "!RUNTIME!" /E /XD node_modules .venv .git .tmp run_copy /R:1 /W:1 /NFL /NDL /NJH /NJS /NC /NS /NP >nul
  for /f "usebackq delims=" %%H in (`powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 '%ROOT%\package-lock.json').Hash"`) do set "CURRENT_LOCK_HASH=%%H"
  set "STORED_LOCK_HASH="
  if exist "!LOCK_HASH_FILE!" set /p STORED_LOCK_HASH=<"!LOCK_HASH_FILE!"
  if not exist "!RUNTIME!\node_modules" (
    echo [INFO] Installing frontend dependencies in runtime mirror...
    cd /d "!RUNTIME!"
    call npm.cmd ci
    if errorlevel 1 (
      call npm.cmd install --package-lock=false
      if errorlevel 1 exit /b 1
    )
    > "!LOCK_HASH_FILE!" echo(!CURRENT_LOCK_HASH!
  ) else if /I not "!CURRENT_LOCK_HASH!"=="!STORED_LOCK_HASH!" (
    echo [INFO] Detected package-lock.json change. Refreshing runtime dependencies...
    cd /d "!RUNTIME!"
    call npm.cmd ci
    if errorlevel 1 (
      call npm.cmd install --package-lock=false
      if errorlevel 1 exit /b 1
    )
    > "!LOCK_HASH_FILE!" echo(!CURRENT_LOCK_HASH!
  ) else (
    cd /d "!RUNTIME!"
  )
) else (
  cd /d "%ROOT%"
)

echo Starting frontend on http://localhost:5173 ...
call npm.cmd run dev -- --host 127.0.0.1 --port 5173
