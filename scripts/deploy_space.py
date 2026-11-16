"""Deploy the app to a Hugging Face Docker Space and smoke-test the live URL.

Run by .github/workflows/deploy.yml after CI passes; also runnable by hand:

    HF_TOKEN=... HF_SPACE=user/uk-rental-law-rag GROQ_API_KEY=... \
        python scripts/deploy_space.py

What it does:
1. creates the Space (Docker SDK, public) if it doesn't exist;
2. stages an allowlisted copy of the repo - the same set .dockerignore admits,
   so nothing local-only (case law, documents, personal files) can be uploaded
   - plus a Space README carrying the front matter Spaces requires;
3. uploads it as one commit, removing files a previous deploy left behind;
4. syncs provider API keys from the environment into Space *secrets* (never
   into files), and LLM_PROVIDER / LLM_MODEL / APP_VERSION into Space variables;
5. waits for the Space to build and serve, then checks /health and the scope
   gate on the live URL.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

# Mirrors .dockerignore: the build context is exactly this set.
INCLUDE_DIRS = ["backend", "frontend", "eval"]
INCLUDE_FILES = ["Dockerfile", ".dockerignore", "requirements.txt", "constraints.txt",
                 "ingestion/build_index.py", "data/processed/legislation_chunks.jsonl",
                 "data/processed/repealed.json"]
SKIP_NAMES = {"__pycache__"}
SKIP_SUFFIXES = {".pyc"}
SECRET_KEYS = ["GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "LLM_API_KEY"]
VARIABLE_KEYS = ["LLM_PROVIDER", "LLM_MODEL"]

SPACE_README = """---
title: UK Rental Law RAG
emoji: 🏠
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Cited answers on England residential tenancy law
---

# UK Residential Rental Law RAG - live demo

Source-grounded legal *information* (not advice) for England residential landlord-tenant
law: hybrid retrieval over legislation, citation-constrained generation, post-generation
citation verification, and an optional tool-calling agent mode.

This Space is deployed automatically from GitHub by CI. Source, architecture and eval
results: {repo_url}

Metrics: [`/metrics`](./metrics) (Prometheus) - [`/metrics/summary`](./metrics/summary) (JSON)
"""


def stage(dest: Path, repo_url: str) -> list[str]:
    copied = []
    for d in INCLUDE_DIRS:
        for path in sorted((ROOT / d).rglob("*")):
            rel = path.relative_to(ROOT)
            if (path.is_dir() or SKIP_NAMES & set(rel.parts) or path.suffix in SKIP_SUFFIXES
                    or (rel.parts[0] == "eval" and path.name.startswith("eval_results_"))):
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append(rel.as_posix())
    for f in INCLUDE_FILES:
        src = ROOT / f
        if not src.exists():
            raise SystemExit(f"missing required file for deploy: {f}")
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / f)
        copied.append(f)
    (dest / "README.md").write_text(SPACE_README.format(repo_url=repo_url or "(see GitHub)"),
                                    encoding="utf-8")
    return copied


def space_url(api: HfApi, repo_id: str) -> str:
    try:
        host = api.space_info(repo_id).host
        if host:
            return host.rstrip("/")
    except Exception:  # noqa: BLE001 - fall back to the documented subdomain scheme
        pass
    owner, name = repo_id.split("/", 1)
    sub = f"{owner}-{name}".lower().replace("_", "-").replace(".", "-")
    return f"https://{sub}.hf.space"


def wait_until_running(api: HfApi, repo_id: str, sha: str, timeout_s: int = 1200) -> str:
    deadline = time.time() + timeout_s
    stage_ = ""
    while time.time() < deadline:
        runtime = api.get_space_runtime(repo_id)
        stage_ = str(getattr(runtime.stage, "value", runtime.stage))
        print(f"  space stage: {stage_}", flush=True)
        if stage_ in ("BUILD_ERROR", "CONFIG_ERROR", "RUNTIME_ERROR"):
            raise SystemExit(f"Space failed to start: {stage_}. See the Space's build logs.")
        if stage_ == "RUNNING":
            return stage_
        time.sleep(20)
    raise SystemExit(f"Timed out waiting for the Space (last stage: {stage_}).")


def _get(url: str, timeout: int = 30) -> dict:
    return json.load(urllib.request.urlopen(url, timeout=timeout))


def smoke_test(url: str, sha: str, timeout_s: int = 600) -> None:
    deadline = time.time() + timeout_s
    health: dict = {}
    while time.time() < deadline:
        try:
            health = _get(f"{url}/health")
            if health.get("corpus") and health.get("version", "").startswith(sha[:7]):
                break
        except Exception as exc:  # noqa: BLE001 - the Space may still be starting
            print(f"  waiting for {url}: {exc}", flush=True)
        time.sleep(10)
    print("  /health:", json.dumps(health))
    if not health.get("corpus"):
        raise SystemExit("Live /health never reported a loaded corpus.")
    if not health.get("version", "").startswith(sha[:7]):
        print(f"::warning::Live version is {health.get('version')!r}, expected {sha[:7]}")
    req = urllib.request.Request(
        f"{url}/query", data=json.dumps({"question": "Eviction notice rules in Scotland?"}).encode(),
        headers={"Content-Type": "application/json"})
    answer = json.load(urllib.request.urlopen(req, timeout=60))
    if not answer.get("abstained"):
        raise SystemExit(f"Scope gate did not abstain on the live Space: {answer}")
    print("  live smoke test passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", default=os.environ.get("GITHUB_SHA", "local"))
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    token, repo_id = os.environ.get("HF_TOKEN"), os.environ.get("HF_SPACE")
    if not token or not repo_id:
        raise SystemExit("Set HF_TOKEN and HF_SPACE (e.g. user/uk-rental-law-rag).")
    api = HfApi(token=token)
    short_sha = args.sha[:7]

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)

    # Secrets and variables first: changing them restarts the Space, so setting
    # them after the upload would restart the build the upload just started.
    for key in SECRET_KEYS:
        if os.environ.get(key):
            api.add_space_secret(repo_id, key, os.environ[key])
            print(f"  synced secret {key}")
    for key in VARIABLE_KEYS:
        if os.environ.get(key):
            api.add_space_variable(repo_id, key, os.environ[key])
            print(f"  set variable {key}={os.environ[key]}")
    # The version the live app reports on /health, so the smoke test can tell
    # the new build from the old one still serving during a rebuild.
    api.add_space_variable(repo_id, "APP_VERSION", short_sha)

    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp), os.environ.get("GITHUB_REPO_URL", ""))
        print(f"Uploading {len(files)} files to {repo_id}", flush=True)
        api.upload_folder(
            repo_id=repo_id, repo_type="space", folder_path=tmp,
            commit_message=f"Deploy {short_sha}",
            # Remove anything a previous deploy left that this one doesn't ship.
            delete_patterns=[f"{d}/**" for d in INCLUDE_DIRS] + ["data/**", "ingestion/**"],
        )

    url = space_url(api, repo_id)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"url={url}\n")
    print(f"Space URL: {url}")
    if args.no_wait:
        return
    wait_until_running(api, repo_id, short_sha)
    smoke_test(url, short_sha)


if __name__ == "__main__":
    sys.exit(main())
