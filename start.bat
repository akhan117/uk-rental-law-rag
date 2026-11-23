@echo off
rem The only file anyone needs to double-click. The first time, it runs setup;
rem after that it starts the app and opens it in its own window.
setlocal
cd /d "%~dp0"
title UK Rental Law Assistant

if not exist "venv\.setup-done" (
    call "%~dp0setup.bat" --from-start
    if errorlevel 1 exit /b 1
)

rem Private mode needs Ollama running; Fast mode doesn't use it at all.
set "USE_OLLAMA=1"
if exist ".env" (
    findstr /r /b /i /c:"LLM_PROVIDER=[a-z]" ".env" | findstr /v /i /c:"=ollama" >nul && set "USE_OLLAMA="
)
if defined USE_OLLAMA call :start_ollama

echo.
echo Starting the app ^(about 15 seconds^)...

rem Always start a fresh server: one left running from before would keep serving
rem the old code or the old Fast/Private setting.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*uvicorn main:app --host 127.0.0.1 --port 8000*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
ping -n 2 127.0.0.1 >nul

if not exist logs mkdir logs
set "ENV_ARGS="
if exist ".env" set "ENV_ARGS=,'--env-file','..\.env'"
powershell -NoProfile -Command "Start-Process -FilePath '%~dp0venv\Scripts\python.exe' -ArgumentList '-m','uvicorn','main:app','--host','127.0.0.1','--port','8000'%ENV_ARGS% -WorkingDirectory '%~dp0backend' -WindowStyle Hidden -RedirectStandardOutput '%~dp0logs\server.log' -RedirectStandardError '%~dp0logs\server-errors.log'"

set /a WAITED=0
:wait_for_server
curl -s -f -o nul -m 2 http://127.0.0.1:8000/health && goto :open_app
set /a WAITED+=1
if %WAITED% geq 90 goto :fail_start
ping -n 2 127.0.0.1 >nul
goto :wait_for_server

:open_app
rem A one-off query value so the page is always fetched fresh: a cached
rem index.html references the old script and stylesheet, which silently pairs
rem an old interface with a new backend after an update.
set "APP_URL=http://127.0.0.1:8000/?launch=%RANDOM%%RANDOM%"

rem A clean app-style window (no address bar or tabs) if Edge is available.
set "EDGE_EXE=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not exist "%EDGE_EXE%" set "EDGE_EXE=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if exist "%EDGE_EXE%" (
    start "" "%EDGE_EXE%" --app=%APP_URL%
) else (
    start "" "%APP_URL%"
)
echo The app is open. This window can be closed.
exit /b 0


:start_ollama
set "OLLAMA_EXE=ollama"
where ollama >nul 2>nul
if errorlevel 1 if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
curl -s -o nul -m 2 http://127.0.0.1:11434/api/tags && exit /b 0
powershell -NoProfile -Command "Start-Process -FilePath '%OLLAMA_EXE%' -ArgumentList 'serve' -WindowStyle Hidden"
for /l %%i in (1,1,30) do (
    ping -n 2 127.0.0.1 >nul
    curl -s -o nul -m 2 http://127.0.0.1:11434/api/tags && exit /b 0
)
exit /b 0


:fail_start
echo.
echo  --------------------------------------------------------------
echo   PROBLEM: The app didn't start. The last lines of its log:
echo  --------------------------------------------------------------
powershell -NoProfile -Command "Get-Content '%~dp0logs\server-errors.log' -Tail 15"
echo.
echo  If it mentions port 8000, another program is using it: restart
echo  the computer and double-click start.bat again.
pause
exit /b 1
