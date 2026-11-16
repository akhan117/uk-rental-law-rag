"""Chunk judgments (Akoma Ntoso XML from Find Case Law) into coherent groups
of numbered paragraphs, each carrying case name, neutral citation, court,
date and paragraph range metadata (per Docs.pdf's case-law chunking rule).
"""
from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from acts import CASES

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
MAX_PARAS_PER_CHUNK = 6


def q(tag: str) -> str:
    return f"{{{AKN_NS}}}{tag}"


def local(tag: str) -> str:
    return etree.QName(tag).localname


def para_text(para_el: etree._Element) -> str:
    content = para_el.find(q("content"))
    if content is None:
        return ""
    return " ".join("".join(content.itertext()).split())


def para_number(para_el: etree._Element) -> str:
    num_el = para_el.find(q("num"))
    if num_el is None:
        return ""
    return "".join(num_el.itertext()).strip().rstrip(".")


@dataclass
class CaseMeta:
    citation: str
    case_name: str
    court: str
    date: str
    url: str


def extract_meta(root: etree._Element, fallback_citation: str) -> CaseMeta:
    nc = root.find(f".//{q('neutralCitation')}")
    citation = "".join(nc.itertext()).strip() if nc is not None else fallback_citation
    name_el = root.find(f".//{q('FRBRname')}")
    case_name = name_el.get("value") if name_el is not None else ""
    court_el = root.find(f".//{{https://caselaw.nationalarchives.gov.uk/akn}}court")
    court = "".join(court_el.itertext()).strip() if court_el is not None else ""
    date_el = root.find(f".//{q('FRBRWork')}/{q('FRBRdate')}")
    date = date_el.get("date") if date_el is not None else ""
    uri_el = root.find(f".//{q('FRBRWork')}/{q('FRBRuri')}")
    url = uri_el.get("value") if uri_el is not None else ""
    return CaseMeta(citation=citation, case_name=case_name, court=court, date=date, url=url)


def collect_paragraphs(body: etree._Element) -> list[tuple[str, str, str]]:
    """Walk the judgment body in document order, returning
    (current_heading, paragraph_number, paragraph_text) for every paragraph.
    """
    results: list[tuple[str, str, str]] = []
    heading_stack: list[str] = []

    def heading_text(level_el: etree._Element) -> str | None:
        h = level_el.find(q("heading"))
        if h is None:
            return None
        text = " ".join("".join(h.itertext()).split())
        return text or None

    def walk(el: etree._Element) -> None:
        tag = local(el.tag)
        pushed = False
        if tag == "level":
            h = heading_text(el)
            if h:
                heading_stack.append(h)
                pushed = True
        if tag == "paragraph" and re.fullmatch(r"\d+\.?", para_number(el) or ""):
            # Only treat main numbered paragraphs (1., 2., 14. ...) as chunk
            # units - nested sub-items ((a), (i), (4) quoting a statute etc.)
            # are para_text()'s itertext() already pulls their text in, so
            # don't recurse into this paragraph's children or they'd be
            # double-counted as spurious extra "paragraphs".
            text = para_text(el)
            if text:
                current_heading = heading_stack[-1] if heading_stack else ""
                results.append((current_heading, para_number(el), text))
            if pushed:
                heading_stack.pop()
            return
        for child in el:
            walk(child)
        if pushed:
            heading_stack.pop()

    walk(body)
    return results


def group_into_chunks(paras: list[tuple[str, str, str]]) -> list[tuple[str, str, str, list[str]]]:
    """Group consecutive same-heading paragraphs, capped at MAX_PARAS_PER_CHUNK.
    Returns (heading, first_num, last_num, texts).
    """
    groups: list[tuple[str, str, str, list[str]]] = []
    current_heading = None
    current_texts: list[str] = []
    current_nums: list[str] = []

    def flush():
        if current_texts:
            groups.append((current_heading or "", current_nums[0], current_nums[-1], current_texts[:]))

    for heading, num, text in paras:
        if heading != current_heading or len(current_texts) >= MAX_PARAS_PER_CHUNK:
            flush()
            current_heading, current_texts, current_nums = heading, [], []
        current_heading = heading
        current_texts.append(f"[{num}] {text}")
        current_nums.append(num)
    flush()
    return groups


def parse_case(case: dict, xml_path: Path, retrieval_date: str) -> list[dict]:
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    meta = extract_meta(root, case["citation"])
    body = root.find(f".//{q('judgmentBody')}")
    if body is None:
        print(f"  [warn] no judgmentBody found in {xml_path.name}")
        return []

    paras = collect_paragraphs(body)
    groups = group_into_chunks(paras)

    chunks = []
    for i, (heading, first_num, last_num, texts) in enumerate(groups):
        para_range = f"[{first_num}]" if first_num == last_num else f"[{first_num}]-[{last_num}]"
        header = f"{meta.case_name} {meta.citation}"
        if heading:
            header += f" — {heading}"
        header += f", paragraphs {para_range}"
        chunks.append({
            "chunk_id": f"{case['slug']}:paras-{first_num}-{last_num}",
            "source_type": "case",
            "case_name": meta.case_name,
            "citation": meta.citation,
            "court": meta.court,
            "date": meta.date,
            "heading": heading,
            "paragraph_range": para_range,
            "url": meta.url,
            "retrieval_date": retrieval_date,
            "text": header + "\n\n" + "\n".join(texts),
        })
    return chunks


def main() -> None:
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "cases"
    out_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "case_chunks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_date = datetime.date.today().isoformat()

    all_chunks: list[dict] = []
    for case in CASES:
        xml_path = raw_dir / f"{case['slug']}.xml"
        if not xml_path.exists():
            print(f"[skip] missing {xml_path}")
            continue
        chunks = parse_case(case, xml_path, retrieval_date)
        print(f"{case['slug']}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    with out_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            # Docs.pdf §2 lists jurisdiction among the metadata every document
            # must carry. This corpus is England-only by construction, so it is a
            # constant - but recording it explicitly is what makes the field
            # usable as a retrieval filter when a second jurisdiction is added.
            f.write(json.dumps({**c, "jurisdiction": "England"}, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_chunks)} case chunks to {out_path}")


if __name__ == "__main__":
    main()
