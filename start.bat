@echo off
setlocal
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if exist "..\.venv\Scripts\python.exe" set "PY=..\.venv\Scripts\python.exe"
if not defined PY goto NOVENV

"%PY%" -c "import lightgbm, rdkit" >nul 2>&1
if errorlevel 1 goto NODEPS

for %%I in ("%PY%") do set "PYW=%%~dpInpythonw.exe"
if exist "%PYW%" (
    start "" "%PYW%" cpamp_gui.py
) else (
    "%PY%" cpamp_gui.py
    if errorlevel 1 pause
)
goto :EOF

:NOVENV
echo.
echo  [ERROR] Python environment not found.
echo  Expected one of:
echo      .venv\Scripts\python.exe
echo      ..\.venv\Scripts\python.exe
echo.
echo  Create it:
echo      python -m venv .venv
echo      .venv\Scripts\pip install -r requirements.txt
echo.
pause
goto :EOF

:NODEPS
echo.
echo  [ERROR] "lightgbm" or "rdkit" is missing.
echo      %PY% -m pip install lightgbm rdkit
echo.
pause
goto :EOF
