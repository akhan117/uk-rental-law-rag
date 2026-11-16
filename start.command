#!/bin/bash
# macOS equivalent of start.bat. Double-clickable in Finder.
#   chmod +x start.command   (if macOS refuses to run it)
cd "$(dirname "$0")" || exit 1

if [ ! -d venv ]; then
    echo "It looks like setup hasn't been run yet."
    echo "Please double-click setup.command first (only needed once)."
    read -r -p "Press return to close..."
    exit 1
fi

OLLAMA_EXE="$(command -v ollama || true)"
[ -z "$OLLAMA_EXE" ] && [ -x /usr/local/bin/ollama ] && OLLAMA_EXE=/usr/local/bin/ollama
[ -z "$OLLAMA_EXE" ] && [ -x /opt/homebrew/bin/ollama ] && OLLAMA_EXE=/opt/homebrew/bin/ollama

# Start the local AI model service unless it's already listening
if ! curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    [ -n "$OLLAMA_EXE" ] && nohup "$OLLAMA_EXE" serve >/dev/null 2>&1 &
    sleep 3
fi

# Start the app server unless it's already listening
if ! curl -s -m 2 http://127.0.0.1:8000/ >/dev/null 2>&1; then
    echo "Starting the app, this takes about 10 seconds the first time..."
    (cd backend && nohup ../venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000 >/dev/null 2>&1 &)
    sleep 10
fi

# A one-off query value so the page is always fetched fresh - a cached
# index.html references the old script and stylesheet, which silently pairs an
# old interface with a new backend. See the comment in start.bat.
APP_URL="http://127.0.0.1:8000/?launch=$RANDOM$RANDOM"

if [ -d "/Applications/Google Chrome.app" ]; then
    open -na "Google Chrome" --args --app="$APP_URL"
elif [ -d "/Applications/Microsoft Edge.app" ]; then
    open -na "Microsoft Edge" --args --app="$APP_URL"
else
    open "$APP_URL"
fi
