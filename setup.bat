@echo off
setlocal

echo ================================================================
echo  UK Residential Rental Law RAG - First-time setup
echo  (only needs to be run once)
echo ================================================================
echo.

rem Probe for a working Python. "where python" isn't enough: Windows ships a
rem Microsoft Store alias stub at that name which passes the lookup but fails
rem to actually run, so run a version check and require a clean exit code.
set "PY_EXE="
py -3 --version >nul 2>nul
if not errorlevel 1 set "PY_EXE=py -3"

if not defined PY_EXE (
    python --version >nul 2>nul
    if not errorlevel 1 set "PY_EXE=python"
)

if not defined PY_EXE (
    echo ERROR: Python was not found on this computer.
    echo Please install Python 3.10 or newer from https://python.org/downloads
    echo ^(tick "Add python.exe to PATH" during install^), then run this file again.
    pause
    exit /b 1
)

if not exist venv (
    echo Creating a private Python environment for this app...
    %PY_EXE% -m venv venv
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create the Python environment.
        pause
        exit /b 1
    )
)

echo.
echo Installing required packages ^(this can take a few minutes^)...
venv\Scripts\python.exe -m pip install --upgrade pip --quiet
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Package install failed. Check your internet connection and try again.
    pause
    exit /b 1
)

set "OLLAMA_EXE=ollama"
where ollama >nul 2>nul
if errorlevel 1 (
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
    ) else (
        echo.
        echo Ollama ^(the local AI model runner^) was not found. Installing it now...
        winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements
        if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
            set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
        ) else (
            echo.
            echo Automatic install didn't work. Please install Ollama manually from
            echo https://ollama.com then run this setup file again.
            pause
            exit /b 1
        )
    )
)

echo.
echo Downloading the AI model ^(about 2.5GB, one-time only^)...
"%OLLAMA_EXE%" pull qwen3:4b
if errorlevel 1 (
    echo.
    echo ERROR: Could not download the model. Check your internet connection and try again.
    pause
    exit /b 1
)

rem Fetch the embedding model now rather than during the first question. It is a
rem ~130MB download that sentence-transformers otherwise performs lazily, inside
rem the first request - which makes the first question appear to hang, and fails
rem outright on a machine that is offline by then.
echo.
echo Downloading the search model ^(about 130MB, one-time only^)...
venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
if errorlevel 1 (
    echo WARNING: could not pre-download the search model; it will download on first use.
)

rem A fresh clone has the parsed legislation (data\processed) but no search
rem index; build it once. Skipped when an index already exists, so a local full
rem build (with case law) is never overwritten.
if not exist "%~dp0data\chroma" (
    echo.
    echo Building the search index ^(one-time, about a minute^)...
    venv\Scripts\python.exe ingestion\build_index.py
    if errorlevel 1 (
        echo.
        echo ERROR: Could not build the search index.
        pause
        exit /b 1
    )
)

echo.
echo ================================================================
echo  Setup complete! Double-click start.bat to run the app.
echo ================================================================
pause
