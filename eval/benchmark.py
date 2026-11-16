"""Load-test a running deployment: latency percentiles, time-to-first-sources,
tokens and estimated cost per query.

The eval measures answer quality offline; this measures serving, end to end,
against a real URL - the local server, a container, or the live Space - over
the same HTTP path a browser uses (POST /query/stream).

    python eval/benchmark.py --url http://127.0.0.1:8000 --n 30
    python eval/benchmark.py --url https://<space>.hf.space --n 20 --concurrency 2 --mode agent

Questions are drawn from the gold set (in-scope + must-answer: the ones that
reach generation). Run the server with ANSWER_CACHE_SIZE=0 for honest numbers;
cached responses are counted and reported separately either way. Mind a public
deployment's rate limit (RATE_LIMIT_PER_MINUTE) - 429s are reported, not retried.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
from telemetry import percentile  # noqa: E402


def questions(n: int) -> list[str]:
    items = [json.loads(line) for line in (HERE / "gold_questions.jsonl").open(encoding="utf-8")
             if line.strip()]
    qs = [i["question"] for i in items if i["type"] in ("in_scope", "must_answer")]
    return [qs[i % len(qs)] for i in range(n)]


def one(client: httpx.Client, url: str, question: str, mode: str) -> dict:
    t0 = time.perf_counter()
    first_sources = None
    answer = None
    status = None
    try:
        with client.stream("POST", f"{url}/query/stream",
                           json={"question": question, "mode": mode}) as resp:
            status = resp.status_code
            if status != 200:
                return {"question": question, "status": status,
                        "latency_s": time.perf_counter() - t0}
            event = None
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[5:])
                    if event == "sources" and first_sources is None:
                        first_sources = time.perf_counter() - t0
                    elif event == "answer":
                        answer = data
                    elif event == "error":
                        return {"question": question, "status": "error",
                                "detail": data.get("detail"), "latency_s": time.perf_counter() - t0}
    except httpx.HTTPError as exc:
        return {"question": question, "status": "http_error", "detail": str(exc),
                "latency_s": time.perf_counter() - t0}
    meta = (answer or {}).get("meta") or {}
    return {
        "question": question, "status": status,
        "latency_s": time.perf_counter() - t0,
        "time_to_sources_s": first_sources,
        "server_latency_s": meta.get("latency_ms", 0) / 1000,
        "abstained": bool((answer or {}).get("abstained")),
        "cached": bool(meta.get("cached")),
        "tokens_in": meta.get("tokens_in", 0), "tokens_out": meta.get("tokens_out", 0),
        "llm_calls": meta.get("llm_calls", 0), "tool_calls": meta.get("tool_calls", 0),
        "unsupported": len((answer or {}).get("unsupported_citations") or []),
        "model": meta.get("model"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--mode", choices=["rag", "agent"], default="rag")
    ap.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=0.0, help="USD per 1M output tokens")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    url = args.url.rstrip("/")

    with httpx.Client(timeout=300) as client:
        health = client.get(f"{url}/health").json()
        print(f"Target: {url}  provider={health.get('provider')} model={health.get('model')} "
              f"mode={args.mode} n={args.n} concurrency={args.concurrency}\n")
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rows = list(pool.map(lambda q: one(client, url, q, args.mode), questions(args.n)))
        wall = time.perf_counter() - t0

    ok = [r for r in rows if r["status"] == 200]
    fresh = [r for r in ok if not r["cached"]]
    answered = [r for r in fresh if not r["abstained"]]
    lat = [r["latency_s"] for r in fresh]
    ttfs = [r["time_to_sources_s"] for r in fresh if r.get("time_to_sources_s")]
    tin = [r["tokens_in"] for r in answered]
    tout = [r["tokens_out"] for r in answered]
    avg = lambda xs: sum(xs) / len(xs) if xs else 0  # noqa: E731
    summary = {
        "url": url, "model": health.get("model"), "provider": health.get("provider"),
        "mode": args.mode, "requests": len(rows), "ok": len(ok),
        "rate_limited": sum(1 for r in rows if r["status"] == 429),
        "errors": sum(1 for r in rows if r["status"] not in (200, 429)),
        "cached": len(ok) - len(fresh), "abstained": sum(1 for r in fresh if r["abstained"]),
        "latency_p50_s": round(percentile(lat, 50), 2),
        "latency_p95_s": round(percentile(lat, 95), 2),
        "time_to_sources_p50_s": round(percentile(ttfs, 50), 2),
        "time_to_sources_p95_s": round(percentile(ttfs, 95), 2),
        "tokens_in_avg": round(avg(tin)), "tokens_out_avg": round(avg(tout)),
        "llm_calls_avg": round(avg([r["llm_calls"] for r in answered]), 2),
        "tool_calls_avg": round(avg([r["tool_calls"] for r in answered]), 2),
        "citation_flag_rate": round(avg([1 if r["unsupported"] else 0 for r in answered]), 3),
        "throughput_qpm": round(len(ok) / wall * 60, 1) if wall else 0,
    }
    if args.price_in or args.price_out:
        summary["est_cost_per_answer_usd"] = round(
            (avg(tin) * args.price_in + avg(tout) * args.price_out) / 1e6, 6)

    width = max(len(k) for k in summary)
    for k, v in summary.items():
        print(f"{k:<{width}}  {v}")
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                            encoding="utf-8")
        print(f"\nWritten to {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
