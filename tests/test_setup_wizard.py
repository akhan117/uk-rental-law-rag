import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import setup_wizard as wiz  # noqa: E402


def test_set_env_creates_from_template_and_keeps_other_lines(tmp_path):
    template = tmp_path / ".env.example"
    template.write_text("# comment\nLLM_PROVIDER=ollama\nGROQ_API_KEY=\nOTHER=1\n")
    env = tmp_path / ".env"

    wiz.set_env({"LLM_PROVIDER": "groq", "GROQ_API_KEY": "gsk_x"}, env, template)

    assert env.read_text().splitlines() == [
        "# comment", "LLM_PROVIDER=groq", "GROQ_API_KEY=gsk_x", "OTHER=1"]


def test_set_env_updates_existing_file_and_appends_missing_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LLM_PROVIDER=groq\n# LLM_PROVIDER=commented\nRAG_TIMEOUT=30\n")

    wiz.set_env({"LLM_PROVIDER": "ollama", "NEW_KEY": "v"}, env, tmp_path / "missing")

    assert env.read_text().splitlines() == [
        "LLM_PROVIDER=ollama", "# LLM_PROVIDER=commented", "RAG_TIMEOUT=30", "NEW_KEY=v"]


@pytest.mark.parametrize("status,expected", [(200, True), (401, False), (503, None)])
def test_check_groq_key(monkeypatch, status, expected):
    monkeypatch.setattr(wiz.httpx, "get", lambda *a, **k: httpx.Response(status))
    assert wiz.check_groq_key("gsk_x") is expected


def test_check_groq_key_offline(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(wiz.httpx, "get", boom)
    assert wiz.check_groq_key("gsk_x") is None


def _run(monkeypatch, tmp_path, answers, key_ok=True):
    env = tmp_path / ".env"
    monkeypatch.setattr(wiz, "ENV_FILE", env)
    monkeypatch.setattr(wiz, "ENV_TEMPLATE", tmp_path / "missing")
    monkeypatch.setattr(wiz, "has_local_accelerator", lambda: False)
    monkeypatch.setattr(wiz.webbrowser, "open", lambda url: True)
    monkeypatch.setattr(wiz, "check_groq_key", lambda k: key_ok if k == "gsk_good" else False)
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))
    return wiz.main(), env.read_text() if env.exists() else ""


def test_private_mode(monkeypatch, tmp_path):
    code, env = _run(monkeypatch, tmp_path, ["3", "2"])
    assert code == wiz.PRIVATE and "LLM_PROVIDER=ollama" in env


def test_fast_mode_retries_a_bad_key(monkeypatch, tmp_path):
    code, env = _run(monkeypatch, tmp_path, ["1", "gsk_bad", "'gsk_good'"])
    assert code == wiz.FAST
    assert "LLM_PROVIDER=groq" in env and "GROQ_API_KEY=gsk_good" in env


def test_fast_mode_can_go_back_to_private(monkeypatch, tmp_path):
    code, env = _run(monkeypatch, tmp_path, ["1", "", "2"])
    assert code == wiz.PRIVATE and "GROQ_API_KEY" not in env


def test_end_of_input_cancels(monkeypatch, tmp_path):
    monkeypatch.setattr(wiz, "has_local_accelerator", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError))
    with pytest.raises(SystemExit) as exc:
        wiz.main()
    assert exc.value.code == wiz.CANCELLED
