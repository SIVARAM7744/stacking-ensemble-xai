# Portable Setup (Zip to Another PC)

This project is prepared for portable use on Windows.

## Preconditions
- Python 3.11 installed (`py -3.11 -V`)
- Node.js + npm installed

## One-time setup after unzip
Open terminal in project root and run:

```bat
.\setup.cmd
```

What this does:
1. Creates `.venv` (local project virtual environment)
2. Installs all Python deps from unified `requirements.txt`
3. Installs frontend deps (`npm ci` fallback `npm install`)
4. Verifies core imports

## Start app (every time)

```bat
.\start_all.cmd
```

Or run separately:

```bat
.\start_backend.cmd
.\start_frontend.cmd
```

## Direct one-command Python install (manual alternative)

```bat
python -m pip --python .\.venv\Scripts\python.exe install -r requirements.txt
```

## Notes
- `.venv` is project-local (not global Python).
- No PowerShell activation is required.
- Do not run `Activate.ps1`; the batch scripts call the venv Python executable directly.
- In PowerShell, run local scripts with `.\`.
- If your folder path includes `#`, Vite may be unstable. Prefer a simple path like `C:\disaster_full_stack`.
