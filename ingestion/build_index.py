"""Embed every chunk (legislation + case law) with a local sentence-transformers
model and persist to a Chroma collection. CPU is fine at this corpus size
(~300-400 chunks).
"""
import json
import re
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from chunk_quality import has_operative_text  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "uk_rental_law"


def load_chunks() -> list[dict]:
    chunks = []
    for name in ["legislation_chunks.jsonl", "case_chunks.jsonl"]:
        path = DATA_DIR / "processed" / name
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        with path.open(encoding="utf-8") as f:
            chunks.extend(json.loads(line) for line in f)
    return chunks


def write_repeal_manifest(dropped: list[dict], kept_chunks: list[dict]) -> None:
    """Record legislation provisions whose body text is gone.

    In legislation.gov.uk's current revised text a repealed section keeps its
    heading and loses its body, so these are provisions no longer in force. They
    are excluded from the index because there is nothing to retrieve - which is
    precisely why the system needs to be told about them separately, or it will
    answer about them from the model's memory instead (see backend/repeals.py).
    """
    # A document that parsed to *nothing but* headings has a parsing gap, not a
    # wholesale repeal. Without this guard the Assured Tenancies (Forms)
    # Regulations - whose substance sits in Schedules this pipeline doesn't read -
    # would have been reported to users as five repealed provisions.
    kept_docs = {c["chunk_id"].split(":", 1)[0] for c in kept_chunks}

    entries = []
    for c in dropped:
        if c.get("source_type") != "legislation":
            continue  # one-line judicial concurrences are also dropped; not repeals
        if c["chunk_id"].split(":", 1)[0] not in kept_docs:
            print(f"  [skip] {c['chunk_id']} - whole document parsed empty, treating as a "
                  f"parsing gap rather than a repeal")
            continue
        tail = c["chunk_id"].rsplit(":", 1)[-1]
        m = re.search(r"(?:section-|crossheading-ground-)(\d+[A-Za-z]*)", tail)
        if not m:
            continue
        ref = ("ground" if "ground-" in tail else "s") + m.group(1).upper()
        entries.append({
            "ref": ref,
            "chunk_id": c["chunk_id"],
            "citation": c.get("citation", ""),
            "heading": (c.get("heading") or "").strip(),
            "url": c.get("url", ""),
        })

    path = DATA_DIR / "processed" / "repealed.json"
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} repealed/emptied provisions to {path}")
    for e in entries:
        print(f"  {e['ref']:10} {e['citation']}")


def main() -> None:
    chunks = load_chunks()
    kept = [c for c in chunks if has_operative_text(c["text"])]
    dropped = [c for c in chunks if not has_operative_text(c["text"])]
    print(f"Loaded {len(chunks)} chunks total; indexing {len(kept)} "
          f"({len(dropped)} heading-only/empty excluded - see backend/chunk_quality.py)")
    write_repeal_manifest(dropped, kept)
    chunks = kept

    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    # bge models expect no special prefix for the passage/document side
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids = [c["chunk_id"] for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {k: v for k, v in c.items() if k not in ("text", "chunk_id")}
        # Chroma metadata values must be str/int/float/bool, not None/lists
        meta = {k: (v if isinstance(v, (str, int, float, bool)) else json.dumps(v))
                 for k, v in meta.items() if v is not None}
        metadatas.append(meta)

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Indexed {collection.count()} chunks into Chroma collection '{COLLECTION_NAME}' at {DATA_DIR / 'chroma'}")


if __name__ == "__main__":
    main()
