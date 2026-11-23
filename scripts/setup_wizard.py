"""The one question setup asks: how should the app answer questions?

  Fast    - a hosted model on Groq's free tier. Needs a free API key; nothing
            else to download, and answers take seconds on any computer.
  Private - Qwen3-4B through Ollama on this computer. A 2.5GB download, and
            slow without a graphics card, but nothing leaves the machine.

The choice is written to .env (created from .env.example if missing; any other
lines are kept). The exit code tells setup.bat / setup.command what is left to
install: 0 = fast (nothing), 10 = private (Ollama and the model), 1 = cancelled.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
ENV_TEMPLATE = ROOT / ".env.example"
GROQ_KEYS_URL = "https://console.groq.com/keys"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

FAST, PRIVATE, CANCELLED = 0, 10, 1
PASTE_HINT = "Cmd+V" if sys.platform == "darwin" else "Ctrl+V, or right-click"


def has_local_accelerator() -> bool:
    """True when a local model will be fast enough to recommend Private mode."""
    if sys.platform == "darwin":
        return platform.machine() == "arm64"  # Apple Silicon
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "-L"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def set_env(values: dict[str, str], env_file: Path | None = None,
            template: Path | None = None) -> None:
    """Set KEY=value lines in .env, keeping every other line as it is."""
    env_file = env_file or ENV_FILE
    template = template or ENV_TEMPLATE
    source = env_file if env_file.exists() else template
    lines = source.read_text(encoding="utf-8").splitlines() if source.exists() else []
    pending = dict(values)
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key in pending:
            lines[i] = f"{key}={pending.pop(key)}"
    lines += [f"{k}={v}" for k, v in pending.items()]
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_groq_key(key: str) -> bool | None:
    """True if Groq accepts the key, False if it rejects it, None if unreachable."""
    try:
        r = httpx.get(GROQ_MODELS_URL, headers={"Authorization": f"Bearer {key}"}, timeout=15)
    except httpx.HTTPError:
        return None
    if r.status_code in (401, 403):
        return False
    return True if r.is_success else None


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(CANCELLED) from None


def choose_mode() -> str:
    gpu = has_local_accelerator()
    print()
    print("How should the app answer questions?")
    print()
    print("  1) Fast" + ("" if gpu else "  (recommended)"))
    print("     Answers in a few seconds on any computer. Uses a free online AI")
    print("     service (Groq), so you need a free key. Getting one takes a minute.")
    print()
    print("  2) Private" + ("  (recommended: this computer can run AI models well)" if gpu else ""))
    print("     Everything stays on this computer. Downloads a 2.5GB AI model.")
    if not gpu:
        print("     Without a graphics card, each answer can take up to a minute.")
    print()
    while True:
        choice = ask("Type 1 or 2 and press Enter: ")
        if choice in ("1", "2"):
            return choice
        print("Please type just 1 or 2.")


def get_groq_key() -> str | None:
    print()
    print("Getting your free key:")
    print("  1. A web page is opening (console.groq.com). Sign in - a Google account works.")
    print("  2. Click 'Create API Key', type any name, and click Submit.")
    print("  3. Click Copy, come back to this window, paste the key and press Enter.")
    print(f"     (To paste: {PASTE_HINT}.)")
    print()
    webbrowser.open(GROQ_KEYS_URL)
    while True:
        key = ask("Your key (or just press Enter to go back): ").strip("\"' ")
        if not key:
            return None
        print("Checking the key...")
        ok = check_groq_key(key)
        if ok:
            print("The key works.")
            return key
        if ok is None:
            print("Couldn't reach Groq to check it (is the internet on?). Saving it anyway.")
            return key
        print("Groq didn't accept that key. Copy it again - all of it, it starts with gsk_ -")
        print("and paste it here.")


def main() -> int:
    while True:
        if choose_mode() == "2":
            set_env({"LLM_PROVIDER": "ollama"})
            print()
            print("Private mode it is.")
            return PRIVATE
        key = get_groq_key()
        if key:
            set_env({"LLM_PROVIDER": "groq", "GROQ_API_KEY": key})
            print()
            print("Fast mode is set up. The key is saved in the .env file in this folder.")
            return FAST


if __name__ == "__main__":
    sys.exit(main())
