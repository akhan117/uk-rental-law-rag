@echo off
rem One-time setup. start.bat runs this automatically the first time, so nobody
rem needs to open it by hand - run it again only to switch Fast/Private mode.
setlocal
cd /d "%~dp0"
title UK Rental Law Assistant - setup
rem Chroma's usage telemetry fails noisily; the messages look like errors.
set "ANONYMIZED_TELEMETRY=False"

echo.
echo  ==============================================================
echo   UK Rental Law Assistant - one-time setup
echo   No admin rights needed. Takes about 5-15 minutes.
echo  ==============================================================
echo.

rem --- 1. uv: downloads its own Python and installs the packages, so there is
rem no Python to install by hand, no "Add to PATH" checkbox, and no clash with
rem any Python already on the computer.
set "UV_EXE="
for %%U in (uv.exe) do set "UV_EXE=%%~$PATH:U"
if not defined UV_EXE if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV_EXE (
    echo [1/4] Getting the installer ^(uv^)...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
)
if not defined UV_EXE goto :fail_uv

echo.
echo [2/4] Installing Python and the app's parts ^(about 1GB^)...
if not exist "venv\Scripts\python.exe" (
    if exist venv rmdir /s /q venv
    "%UV_EXE%" venv venv --python 3.11
    if errorlevel 1 goto :fail_packages
)
"%UV_EXE%" pip install --python venv\Scripts\python.exe -r requirements.txt -c constraints.txt
if errorlevel 1 goto :fail_packages

echo.
echo [3/4] One question...
set "MODE=fast"
venv\Scripts\python.exe scripts\setup_wizard.py
if errorlevel 10 (
    set "MODE=private"
) else (
    if errorlevel 1 goto :fail_cancelled
)

echo.
echo [4/4] Downloading the remaining pieces and building the search index...
if "%MODE%"=="private" goto :private_model
:after_private_model

rem Fetch the search model now rather than inside the first question, which
rem would otherwise look like a hang (~130MB).
echo Downloading the search model ^(about 130MB^)...
venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
if errorlevel 1 goto :fail_download

rem Build the index from the bundled legislation. Skipped when one exists, so a
rem local full build (with case law) is never overwritten.
if not exist "data\chroma" (
    echo Building the search index ^(about a minute^)...
    venv\Scripts\python.exe ingestion\build_index.py
    if errorlevel 1 goto :fail_index
)

type nul > "venv\.setup-done"
echo.
echo  ==============================================================
echo   Setup finished.
echo  ==============================================================
if /i "%~1"=="--from-start" exit /b 0
call "%~dp0start.bat"
exit /b 0


:private_model
set "OLLAMA_EXE="
for %%O in (ollama.exe) do set "OLLAMA_EXE=%%~$PATH:O"
if not defined OLLAMA_EXE if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not defined OLLAMA_EXE (
    echo Installing Ollama, which runs the AI model on this computer...
    winget install --id Ollama.Ollama -e --silent --accept-package-agreements --accept-source-agreements
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
)
if not defined OLLAMA_EXE goto :fail_ollama
call :ensure_ollama_running
echo Downloading the AI model ^(about 2.5GB^)...
"%OLLAMA_EXE%" pull qwen3:4b
if errorlevel 1 goto :fail_download
goto :after_private_model


:ensure_ollama_running
curl -s -o nul -m 2 http://127.0.0.1:11434/api/tags && exit /b 0
powershell -NoProfile -Command "Start-Process -FilePath '%OLLAMA_EXE%' -ArgumentList 'serve' -WindowStyle Hidden"
for /l %%i in (1,1,30) do (
    ping -n 2 127.0.0.1 >nul
    curl -s -o nul -m 2 http://127.0.0.1:11434/api/tags && exit /b 0
)
exit /b 0


:fail_uv
call :problem "Couldn't download the installer." "Check the internet connection, then double-click start.bat again."
exit /b 1
:fail_packages
call :problem "Couldn't install the app's parts." "Check the internet connection, then double-click start.bat again."
exit /b 1
:fail_cancelled
call :problem "Setup was stopped before the question was answered." "Double-click start.bat whenever you're ready to carry on."
exit /b 1
:fail_ollama
start "" https://ollama.com/download
call :problem "Couldn't install Ollama automatically." "Install it from the page that just opened, then double-click start.bat again."
exit /b 1
:fail_download
call :problem "A download didn't finish." "Check the internet connection, then double-click start.bat again. It picks up where it stopped."
exit /b 1
:fail_index
call :problem "Couldn't build the search index." "Double-click start.bat to try again."
exit /b 1

:problem
echo.
echo  --------------------------------------------------------------
echo   PROBLEM: %~1
echo   %~2
echo  --------------------------------------------------------------
pause
exit /b 0
