#!/bin/bash
# The only file anyone needs to double-click. The first time, it runs setup;
# after that it starts the app and opens it in its own window.
cd "$(dirname "$0")" || exit 1

if [ ! -f venv/.setup-done ]; then
    bash ./setup.command --from-start || exit 1
fi

# Private mode needs Ollama running; Fast mode doesn't use it at all.
USE_OLLAMA=1
if [ -f .env ] && grep -Eiq '^LLM_PROVIDER=[a-z]' .env && ! grep -Eiq '^LLM_PROVIDER=ollama' .env; then
    USE_OLLAMA=0
fi
if [ "$USE_OLLAMA" = 1 ] && ! curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    OLLAMA_EXE=""
    for c in "$(command -v ollama)" /usr/local/bin/ollama /opt/homebrew/bin/ollama \
             /Applications/Ollama.app/Contents/Resources/ollama; do
        if [ -n "$c" ] && [ -x "$c" ]; then OLLAMA_EXE="$c"; break; fi
    done
    if [ -n "$OLLAMA_EXE" ]; then
        nohup "$OLLAMA_EXE" serve >/dev/null 2>&1 &
        for _ in $(seq 1 30); do
            sleep 1
            curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null && break
        done
    fi
fi

echo
echo "Starting the app (about 15 seconds)..."

# Always start a fresh server: one left running from before would keep serving
# the old code or the old Fast/Private setting.
pkill -f "uvicorn main:app --host 127.0.0.1 --port 8000" 2>/dev/null && sleep 1

mkdir -p logs
ENV_ARGS=()
[ -f .env ] && ENV_ARGS=(--env-file ../.env)
(cd backend && nohup ../venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000 \
    "${ENV_ARGS[@]}" >../logs/server.log 2>&1 &)

started=0
for _ in $(seq 1 90); do
    if curl -s -f -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then started=1; break; fi
    sleep 1
done
if [ "$started" != 1 ]; then
    echo
    echo " --------------------------------------------------------------"
    echo "  PROBLEM: The app didn't start. The last lines of its log:"
    echo " --------------------------------------------------------------"
    tail -n 15 logs/server.log
    echo
    echo "  If it mentions port 8000, another program is using it: restart"
    echo "  the Mac and double-click start.command again."
    read -r -p "Press return to close..."
    exit 1
fi

# A one-off query value so the page is always fetched fresh: a cached
# index.html references the old script and stylesheet, which silently pairs an
# old interface with a new backend after an update.
APP_URL="http://127.0.0.1:8000/?launch=$RANDOM$RANDOM"

if [ -d "/Applications/Google Chrome.app" ]; then
    open -na "Google Chrome" --args --app="$APP_URL"
elif [ -d "/Applications/Microsoft Edge.app" ]; then
    open -na "Microsoft Edge" --args --app="$APP_URL"
else
    open "$APP_URL"
fi
echo "The app is open. This window can be closed."
