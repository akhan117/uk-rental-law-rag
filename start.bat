@echo off
setlocal

if not exist "%~dp0venv" (
    echo It looks like setup hasn't been run yet.
    echo Please double-click setup.bat first ^(only needed once^).
    pause
    exit /b 1
)

set "OLLAMA_EXE=ollama"
where ollama >nul 2>nul
if errorlevel 1 (
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
)

rem Start the local AI model service in the background, unless it's already running
netstat -ano | findstr ":11434" | findstr "LISTENING" >nul
if errorlevel 1 (
    powershell -NoProfile -Command "Start-Process -FilePath '%OLLAMA_EXE%' -ArgumentList 'serve' -WindowStyle Hidden"
    timeout /t 3 /nobreak >nul
)

rem Start the app server in the background, unless it's already running
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if errorlevel 1 (
    echo Starting the app, this takes about 10 seconds the first time...
    powershell -NoProfile -Command "Start-Process -FilePath '%~dp0venv\Scripts\python.exe' -ArgumentList '-m','uvicorn','main:app','--host','127.0.0.1','--port','8000' -WorkingDirectory '%~dp0backend' -WindowStyle Hidden"
    timeout /t 10 /nobreak >nul
)

rem Open a clean app-style window (no address bar/tabs) if Edge is available, else default browser
set "EDGE_EXE=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not exist "%EDGE_EXE%" set "EDGE_EXE=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"

rem Open with a one-off query value so the page is always fetched fresh. The
rem browser caches index.html, and a cached copy references the old script and
rem stylesheet - so after an update the app can silently run an old interface
rem against a new backend, which looks like "the changes didn't happen" and is
rem near-impossible to diagnose remotely. A URL it has never seen can't be cached.
set "APP_URL=http://127.0.0.1:8000/?launch=%RANDOM%%RANDOM%"

if exist "%EDGE_EXE%" (
    start "" "%EDGE_EXE%" --app=%APP_URL%
) else (
    start "" "%APP_URL%"
)
