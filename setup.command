#!/bin/bash
# One-time setup for macOS. start.command runs this automatically the first
# time, so nobody needs to open it by hand - run it again only to switch
# Fast/Private mode.
cd "$(dirname "$0")" || exit 1
# Chroma's usage telemetry fails noisily; the messages look like errors.
export ANONYMIZED_TELEMETRY=False

problem() {
    echo
    echo " --------------------------------------------------------------"
    echo "  PROBLEM: $1"
    echo "  $2"
    echo " --------------------------------------------------------------"
    read -r -p "Press return to close..."
    exit 1
}

find_ollama() {
    for c in "$(command -v ollama)" /usr/local/bin/ollama /opt/homebrew/bin/ollama \
             /Applications/Ollama.app/Contents/Resources/ollama; do
        if [ -n "$c" ] && [ -x "$c" ]; then echo "$c"; return; fi
    done
}

ensure_ollama_running() {
    curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null && return
    nohup "$OLLAMA_EXE" serve >/dev/null 2>&1 &
    for _ in $(seq 1 30); do
        sleep 1
        curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null && return
    done
}

echo
echo " =============================================================="
echo "  UK Rental Law Assistant - one-time setup"
echo "  No admin password needed. Takes about 5-15 minutes."
echo " =============================================================="
echo

# 1. uv: downloads its own Python and installs the packages, so there is no
# Python to install by hand and no clash with any Python already installed.
UV_EXE="$(command -v uv || true)"
[ -z "$UV_EXE" ] && [ -x "$HOME/.local/bin/uv" ] && UV_EXE="$HOME/.local/bin/uv"
if [ -z "$UV_EXE" ]; then
    echo "[1/4] Getting the installer (uv)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -x "$HOME/.local/bin/uv" ] && UV_EXE="$HOME/.local/bin/uv"
fi
[ -n "$UV_EXE" ] || problem "Couldn't download the installer." \
    "Check the internet connection, then double-click start.command again."

echo
echo "[2/4] Installing Python and the app's parts (about 1GB)..."
if [ ! -x venv/bin/python ]; then
    rm -rf venv
    "$UV_EXE" venv venv --python 3.11 || problem "Couldn't install Python." \
        "Check the internet connection, then double-click start.command again."
fi
"$UV_EXE" pip install --python venv/bin/python -r requirements.txt -c constraints.txt \
    || problem "Couldn't install the app's parts." \
        "Check the internet connection, then double-click start.command again."

echo
echo "[3/4] One question..."
venv/bin/python scripts/setup_wizard.py
case $? in
    0) MODE=fast ;;
    10) MODE=private ;;
    *) problem "Setup was stopped before the question was answered." \
        "Double-click start.command whenever you're ready to carry on." ;;
esac

echo
echo "[4/4] Downloading the remaining pieces and building the search index..."
if [ "$MODE" = private ]; then
    OLLAMA_EXE="$(find_ollama)"
    if [ -z "$OLLAMA_EXE" ]; then
        echo "Installing Ollama, which runs the AI model on this computer..."
        if command -v brew >/dev/null 2>&1; then
            brew install --cask ollama
        else
            tmp="$(mktemp -d)"
            curl -fL -o "$tmp/Ollama.zip" https://ollama.com/download/Ollama-darwin.zip \
                && unzip -q -o "$tmp/Ollama.zip" -d /Applications
            rm -rf "$tmp"
        fi
        OLLAMA_EXE="$(find_ollama)"
    fi
    if [ -z "$OLLAMA_EXE" ]; then
        open https://ollama.com/download
        problem "Couldn't install Ollama automatically." \
            "Install it from the page that just opened, then double-click start.command again."
    fi
    ensure_ollama_running
    echo "Downloading the AI model (about 2.5GB)..."
    "$OLLAMA_EXE" pull qwen3:4b || problem "A download didn't finish." \
        "Check the internet connection, then double-click start.command again."
fi

# Fetch the search model now rather than inside the first question, which
# would otherwise look like a hang (~130MB).
echo "Downloading the search model (about 130MB)..."
venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" \
    || problem "A download didn't finish." \
        "Check the internet connection, then double-click start.command again."

# Build the index from the bundled legislation. Skipped when one exists, so a
# local full build (with case law) is never overwritten.
if [ ! -d data/chroma ]; then
    echo "Building the search index (about a minute)..."
    venv/bin/python ingestion/build_index.py \
        || problem "Couldn't build the search index." "Double-click start.command to try again."
fi

touch venv/.setup-done
# A downloaded copy carries macOS's quarantine flag, which makes every script in
# it ask for permission again. The person has already approved this folder.
xattr -dr com.apple.quarantine . 2>/dev/null || true

echo
echo " =============================================================="
echo "  Setup finished."
echo " =============================================================="
[ "$1" = "--from-start" ] && exit 0
exec bash ./start.command
