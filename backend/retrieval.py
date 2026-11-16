"""Hybrid retrieval: dense vector search fused with BM25 lexical search.

Dense search alone misses provisions whose wording is amendment-instruction
drafting ("omit Chapter 2 of Part 1") rather than natural language, even when
the section heading states the point plainly ("Abolition of assured shorthold
tenancies"). BM25 catches those by exact term overlap. Measured on the gold
set, fusing the two lifts recall@6 from 74% to 87%; a cross-encoder rerank on
top of this was also tried and added nothing, so it isn't used.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from chunk_quality import has_operative_text

DATA_DIR = Path(os.environ.get("RAG_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "uk_rental_law"
CANDIDATES_PER_METHOD = 20


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# People ask in everyday words ("lease", "move out", "kick me out"); the statute
# says "tenancy", "notice to quit", "possession". Neither the embedding nor BM25
# bridges that reliably, so append the statutory wording to the query. Measured:
# colloquial phrasings 1/5 -> 4/5 retrieved correctly, gold set 80% -> 83%.
QUERY_SYNONYMS = {
    r"\bleaser\b|\blessee\b|\brenter\b": "tenant",
    r"\blease\b|\bleases\b": "tenancy",
    r"\bterminate\b|\bterminating\b|\bend(ing)? (my|the|a) (lease|tenancy|contract)\b":
        "notice to quit end tenancy",
    r"\bmove out\b|\bleave the (property|flat|house)\b": "notice to quit end tenancy",
    r"\bkick (me|us) out\b|\bthrow (me|us) out\b": "possession eviction",
    r"\bevict(ed|ion)?\b": "possession notice",
    r"\bcontract\b": "tenancy agreement",
    r"\bdeposit back\b": "tenancy deposit repayment",
}
# Deliberately NOT here: mandatory -> "Court must order possession" and
# discretionary -> "Court may order possession". It looks like the obvious
# counterpart to putting Part titles into ground chunks, and it measurably hurt.
# "Is Ground 10 mandatory or discretionary?" contains both words, so both
# phrasings get appended, and they match every Part heading in Schedule 2 - which
# buried the ground itself from rank 1 ("Ground 10 possession") to rank 23. The
# Part title in the chunk text is enough: retrieval finds the ground on its
# number, and the model then reads "Court may Order Possession" from the text.


def expand_query(question: str) -> str:
    lowered = question.lower()
    extra = [add for pattern, add in QUERY_SYNONYMS.items() if re.search(pattern, lowered)]
    return f"{question} {' '.join(extra)}" if extra else question


class Retriever:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"),
                                           settings=Settings(anonymized_telemetry=False))
        self.collection = client.get_collection(COLLECTION_NAME)

        # Same exclusion the Chroma index applies, so the lexical and dense
        # halves search exactly the same set of chunks.
        self.chunks: list[dict] = []
        for name in ["legislation_chunks.jsonl", "case_chunks.jsonl"]:
            path = DATA_DIR / "processed" / name
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    self.chunks.extend(
                        c for c in (json.loads(line) for line in f)
                        if has_operative_text(c["text"])
                    )
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])

    def _dense_ids(self, query: str, n: int,
                   where: dict | None = None) -> tuple[list[str], dict[str, float]]:
        """Dense candidates and their distances. Returned, not stored on the
        instance: one Retriever serves concurrent requests (and agent tool calls on
        worker threads), so per-query state on self would leak between them."""
        emb = self.model.encode([query], normalize_embeddings=True,
                                show_progress_bar=False).tolist()
        res = self.collection.query(query_embeddings=emb, n_results=n,
                                    **({"where": where} if where else {}))
        return res["ids"][0], dict(zip(res["ids"][0], res["distances"][0]))

    def _lexical_ids(self, query: str, n: int, allowed: set[str] | None = None) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ids = (cid for i in order if (cid := self.chunk_ids[i]) is not None
               and (allowed is None or cid in allowed))
        return [cid for cid, _ in zip(ids, range(n))]

    def _allowed_ids(self, where: dict) -> set[str]:
        """Chunk ids matching a metadata filter, for the lexical side.

        BM25 scores plain text and knows nothing about metadata, so the filter
        has to be applied to its candidate list separately or a filtered search
        would silently return unfiltered lexical hits.
        """
        return {
            cid for cid, c in zip(self.chunk_ids, self.chunks)
            if all(c.get(k) == v for k, v in where.items())
        }

    def multi_search(self, queries: list[str], k: int = 6, where: dict | None = None) -> list[dict]:
        """Retrieve for several queries and interleave the results.

        A multi-part question ("my deposit wasn't protected and there's black
        mould") has no single chunk that answers it and no single embedding that
        finds both halves. Running one search per sub-question and interleaving
        guarantees each half gets slots, instead of whichever half dominates the
        combined embedding taking all six.

        A chunk found by more than one query keeps its best (smallest) distance.
        """
        if len(queries) == 1:
            return self.search(queries[0], k=k, where=where)

        per_query = [self.search(q, k=k, where=where) for q in queries]
        merged: list[dict] = []
        seen: dict[str, dict] = {}
        for i in range(k):
            for hits in per_query:
                if i >= len(hits):
                    continue
                h = hits[i]
                if h["chunk_id"] in seen:
                    prev = seen[h["chunk_id"]]
                    prev["distance"] = min(prev["distance"], h["distance"])
                    continue
                seen[h["chunk_id"]] = h
                merged.append(h)
                if len(merged) >= k:
                    return merged
        return merged

    def search(self, query: str, k: int = 6, where: dict | None = None) -> list[dict]:
        """`where` restricts retrieval by chunk metadata (Docs.pdf §3): e.g.
        {"source_type": "legislation"}, {"court": "UKSC"}, {"act_title": "..."}.
        Applied to both halves of the hybrid search, not just the dense one."""
        query = expand_query(query)
        dense_ids, distances = self._dense_ids(query, CANDIDATES_PER_METHOD, where)
        lexical_ids = self._lexical_ids(query, CANDIDATES_PER_METHOD,
                                        self._allowed_ids(where) if where else None)

        # Round-robin interleave rather than reciprocal-rank fusion: RRF rewards
        # appearing in both lists, which buries a provision that only the lexical
        # side finds even when it ranks #1 there - exactly the case this exists for.
        merged: list[str] = []
        seen: set[str] = set()
        for i in range(CANDIDATES_PER_METHOD):
            for source in (dense_ids, lexical_ids):
                if i < len(source) and source[i] not in seen:
                    seen.add(source[i])
                    merged.append(source[i])
                    if len(merged) >= k:
                        break
            if len(merged) >= k:
                break

        results = self.collection.get(ids=merged, include=["documents", "metadatas"])
        by_id = {
            cid: {"text": doc, **meta}
            for cid, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])
        }
        hits = []
        for cid in merged:
            entry = by_id.get(cid)
            if entry is None:
                continue
            hits.append({
                "chunk_id": cid,
                # Distance is only defined for the dense side; lexical-only hits get
                # the dense cutoff distance so the scope gate stays conservative.
                "distance": distances.get(cid, 1.0),
                **entry,
            })
        return hits
