#!/bin/bash
# macOS equivalent of setup.bat. Double-clickable in Finder (.command files open
# in Terminal). If macOS refuses to run it, the file needs the executable bit:
#   chmod +x setup.command
cd "$(dirname "$0")" || exit 1

echo "================================================================"
echo " UK Residential Rental Law RAG - First-time setup"
echo " (only needs to be run once)"
echo "================================================================"
echo

PY_EXE=""
for candidate in python3 python; do
    if "$candidate" --version >/dev/null 2>&1; then PY_EXE="$candidate"; break; fi
done

if [ -z "$PY_EXE" ]; then
    echo "ERROR: Python was not found on this computer."
    echo "Please install Python 3.10 or newer from https://python.org/downloads"
    echo "then run this file again."
    read -r -p "Press return to close..."
    exit 1
fi

if [ ! -d venv ]; then
    echo "Creating a private Python environment for this app..."
    "$PY_EXE" -m venv venv || { echo "ERROR: Could not create the Python environment."; read -r -p "Press return to close..."; exit 1; }
fi

echo
echo "Installing required packages (this can take a few minutes)..."
venv/bin/python -m pip install --upgrade pip --quiet
if ! venv/bin/python -m pip install -r requirements.txt; then
    echo
    echo "ERROR: Package install failed. Check your internet connection and try again."
    read -r -p "Press return to close..."
    exit 1
fi

if ! command -v ollama >/dev/null 2>&1 && [ ! -x /usr/local/bin/ollama ] && [ ! -x /opt/homebrew/bin/ollama ]; then
    echo
    echo "Ollama (the local AI model runner) was not found."
    if command -v brew >/dev/null 2>&1; then
        echo "Installing it with Homebrew..."
        brew install --cask ollama
    fi
fi

OLLAMA_EXE="$(command -v ollama || true)"
[ -z "$OLLAMA_EXE" ] && [ -x /usr/local/bin/ollama ] && OLLAMA_EXE=/usr/local/bin/ollama
[ -z "$OLLAMA_EXE" ] && [ -x /opt/homebrew/bin/ollama ] && OLLAMA_EXE=/opt/homebrew/bin/ollama

if [ -z "$OLLAMA_EXE" ]; then
    echo
    echo "Automatic install didn't work. Please install Ollama manually from"
    echo "https://ollama.com then run this setup file again."
    read -r -p "Press return to close..."
    exit 1
fi

echo
echo "Downloading the AI model (about 2.5GB, one-time only)..."
if ! "$OLLAMA_EXE" pull qwen3:4b; then
    echo
    echo "ERROR: Could not download the model. Check your internet connection and try again."
    read -r -p "Press return to close..."
    exit 1
fi

# Fetch the embedding model now rather than during the first question. It is a
# ~130MB download that sentence-transformers otherwise performs lazily, inside
# the first request - which makes the first question appear to hang, and fails
# outright on a machine that is offline by then.
echo
echo "Downloading the search model (about 130MB, one-time only)..."
venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" \
    || echo "WARNING: could not pre-download the search model; it will download on first use."

echo
echo "================================================================"
echo " Setup complete! Double-click start.command to run the app."
echo "================================================================"
read -r -p "Press return to close..."
