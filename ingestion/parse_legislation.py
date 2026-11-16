"""Structure-aware chunking of legislation.gov.uk XML: Act -> Part -> Section
-> Subsection (and the Schedule equivalent: Schedule -> Part -> Ground/Paragraph).

Each chunk keeps the whole section/ground together (not exploded per-subsection)
unless it's long enough that a single embedding would blur too many distinct
provisions together, in which case it falls back to per-subsection chunks that
still carry the parent section's heading/citation as metadata.

Known simplification: legislation.gov.uk marks in-progress amendments with
<Repeal>/<Addition> spans (old text vs. replacement text shown side by side).
We drop <Repeal> text and keep <Addition> text as a heuristic for "current
text", rather than resolving exact commencement dates. Flagged in the README.
"""
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from acts import ACTS

NS = {"leg": "http://www.legislation.gov.uk/namespaces/legislation"}
LONG_CHUNK_CHARS = 1800  # above this, split section into per-subsection chunks


def q(tag: str) -> str:
    return f"{{{NS['leg']}}}{tag}"


@dataclass
class Chunk:
    source_type: str
    act_title: str
    citation: str
    heading: str
    url: str
    retrieval_date: str
    text: str
    chunk_id: str
    # Schedule 2 Part I / Part II - whether the court must or may order possession
    part_title: str | None = None
    extra: dict = field(default_factory=dict)


def _repeal_applies_in_england(node: etree._Element) -> bool:
    """Does this <Repeal> actually remove the text for England?

    legislation.gov.uk marks repeals with an Extent. A repeal of Extent="W" is a
    Wales-only repeal: in an England-only corpus the text is still in force and
    dropping it loses live law. That is not a corner case - it is why the
    statutory instruments parsed to nothing but headings, since the prescribed
    deposit information is repealed for Wales and retained for England.

    Everything else is treated as applying. In particular RetainText="true" with
    no Extent is NOT read as "still in force": that exact combination appears both
    in Housing Act 1988 s.21, which the Renters' Rights Act repealed outright, and
    in s.13, which is in force. The attribute cannot tell them apart - only
    commencement data could, which this pipeline does not resolve (see the
    amendment-heuristic limitation in the README). Between wrongly dropping some
    text and wrongly presenting repealed law as current, this errs towards the
    first.
    """
    return (node.get("Extent") or "").strip().upper() != "W"


def local_text(el: etree._Element) -> str:
    """Join text under el, dropping <Repeal> subtrees (see module docstring)."""
    parts: list[str] = []

    def walk(node: etree._Element) -> None:
        if etree.QName(node).localname == "Repeal" and _repeal_applies_in_england(node):
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(el)
    text = "".join(parts)
    return " ".join(text.split())


def find_by_id(root: etree._Element, element_id: str) -> etree._Element | None:
    matches = root.xpath(f'.//*[@id="{element_id}"]')
    return matches[0] if matches else None


def title_of(el: etree._Element) -> str:
    t = el.find(q("Title"))
    if t is None:
        return ""
    return " ".join("".join(t.itertext()).split())


def number_of(el: etree._Element) -> str:
    n = el.find(q("Pnumber"))
    if n is None:
        n = el.find(q("Number"))
    if n is None:
        return ""
    return " ".join("".join(n.itertext()).split())


# Acts number provisions as sections; statutory instruments use articles or
# regulations, with otherwise identical P1group/P1/P2 structure. Keying off the
# id prefix lets one parser handle both (Docs.pdf §2 asks for "Acts and statutory
# instruments") and keeps each citation in the form lawyers actually write.
PROVISION_LABELS = {"section": "s.", "article": "art.", "regulation": "reg."}


def section_chunks(act: dict, root: etree._Element, retrieval_date: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    wanted = act["wanted_ids"]
    for p1 in root.iter(q("P1")):
        sec_id = p1.get("id")
        kind = sec_id.split("-")[0] if sec_id else ""
        if kind not in PROVISION_LABELS:
            continue
        label = PROVISION_LABELS[kind]
        if wanted != "all" and sec_id not in wanted:
            continue

        group = p1.getparent()
        if etree.QName(group).localname != "P1group":
            group = p1
        heading = title_of(group)
        sec_num = number_of(p1)
        url = p1.get("DocumentURI", "")
        full_text = local_text(p1)

        if len(full_text) <= LONG_CHUNK_CHARS:
            chunks.append(Chunk(
                source_type="legislation",
                act_title=act["expected_title"],
                citation=f"{act['expected_title']}, {label}{sec_num}",
                heading=heading,
                url=url,
                retrieval_date=retrieval_date,
                text=f"{heading}\n\n{full_text}" if heading else full_text,
                chunk_id=sec_id,
            ))
            continue

        # Long section: split into per-subsection chunks, keep section context.
        subs = p1.findall(f"{q('P1para')}/{q('P2')}")
        if not subs:
            chunks.append(Chunk(
                source_type="legislation",
                act_title=act["expected_title"],
                citation=f"{act['expected_title']}, {label}{sec_num}",
                heading=heading,
                url=url,
                retrieval_date=retrieval_date,
                text=f"{heading}\n\n{full_text}" if heading else full_text,
                chunk_id=sec_id,
            ))
            continue

        for p2 in subs:
            sub_id = p2.get("id", sec_id)
            sub_num = number_of(p2)
            sub_text = local_text(p2)
            if not sub_text:
                continue
            chunks.append(Chunk(
                source_type="legislation",
                act_title=act["expected_title"],
                citation=f"{act['expected_title']}, {label}{sec_num}({sub_num})",
                heading=heading,
                url=p2.get("DocumentURI", url),
                retrieval_date=retrieval_date,
                text=f"{heading} (s.{sec_num}({sub_num}))\n\n{sub_text}",
                chunk_id=sub_id,
                extra={"parent_section": sec_id},
            ))
    return chunks


def ground_chunks(act: dict, root: etree._Element, retrieval_date: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    wanted = act.get("wanted_schedule_grounds", set())
    if wanted == "all":
        # Only the cross-heading block itself ("...crossheading-ground-8"), never its
        # nested paragraph wrappers ("...crossheading-ground-8_paragraph-wrapper8n1"),
        # which repeat the same text and would put duplicate copies of one ground in
        # the corpus - wasting retrieval slots and defeating the tamper test.
        wanted = sorted(
            el.get("id") for el in root.iter(q("Pblock"))
            if "crossheading-ground-" in (el.get("id") or "")
        )
    for ground_id in wanted:
        pblock = find_by_id(root, ground_id)
        if pblock is None:
            print(f"  [warn] schedule ground id not found: {ground_id}")
            continue
        heading = title_of(pblock)
        text = local_text(pblock)
        url = pblock.get("DocumentURI", "")
        # schedule number is embedded in the id, e.g. schedule-2-part-I-...
        schedule_num = ground_id.split("-")[1]

        # The Part a ground sits in carries whether possession is mandatory or
        # discretionary - Schedule 2 Part I is "Grounds on which Court must order
        # possession", Part II "...may order possession". That was previously kept
        # only in the chunk id, so it was invisible to both retrieval and the
        # model: asked whether Ground 10 is mandatory or discretionary the system
        # could not retrieve the distinction and invented a citation instead.
        # Including the Part's own title keeps the chunk faithful to the source
        # while making the distinction searchable.
        part_title = ""
        node = pblock.getparent()
        while node is not None:
            if etree.QName(node).localname == "Part":
                part_title = title_of(node)
                break
            node = node.getparent()

        # The Part title is deliberately NOT prefixed onto the text. Doing so gave
        # all 39 ground chunks the same long opening line, which homogenised their
        # embeddings: queries stopped resolving to a specific ground and returned
        # an arbitrary cluster of them (asking for Ground 10 returned 5d, 6a, 5e,
        # 5b). It is carried as metadata and attached to the excerpt label at
        # prompt time instead - the distinction is needed to answer, not to find.
        prefix = f"{act['expected_title']}, Schedule {schedule_num}"
        chunks.append(Chunk(
            source_type="legislation",
            act_title=act["expected_title"],
            citation=f"{act['expected_title']}, Sch.{schedule_num} {heading}",
            heading=heading,
            url=url,
            retrieval_date=retrieval_date,
            text=f"{prefix}, {heading}\n\n{text}",
            chunk_id=ground_id,
            part_title=part_title or None,
        ))
    return chunks


def whole_schedule_chunks(act: dict, root: etree._Element, retrieval_date: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sched_id in act.get("wanted_schedules", set()):
        sched = find_by_id(root, sched_id)
        if sched is None:
            print(f"  [warn] schedule id not found: {sched_id}")
            continue
        sched_title = title_of(sched.find(q("TitleBlock"))) or title_of(sched)
        url = sched.get("DocumentURI", "")
        full_text = local_text(sched)

        blocks = sched.findall(f"{q('ScheduleBody')}/{q('Pblock')}")
        if len(full_text) <= LONG_CHUNK_CHARS or not blocks:
            chunks.append(Chunk(
                source_type="legislation",
                act_title=act["expected_title"],
                citation=f"{act['expected_title']}, {sched_id.replace('-', ' ').title()}",
                heading=sched_title,
                url=url,
                retrieval_date=retrieval_date,
                text=f"{act['expected_title']}, {sched_title}\n\n{full_text}",
                chunk_id=sched_id,
            ))
            continue

        for block in blocks:
            block_text = local_text(block)
            if not block_text:
                continue
            block_heading = title_of(block)
            chunks.append(Chunk(
                source_type="legislation",
                act_title=act["expected_title"],
                citation=f"{act['expected_title']}, {sched_title} — {block_heading}",
                heading=block_heading,
                url=block.get("DocumentURI", url),
                retrieval_date=retrieval_date,
                text=f"{act['expected_title']}, {sched_title}, {block_heading}\n\n{block_text}",
                chunk_id=block.get("id", sched_id),
                extra={"parent_schedule": sched_id},
            ))
    return chunks


def parse_act(act: dict, xml_path: Path, retrieval_date: str) -> list[Chunk]:
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    chunks = section_chunks(act, root, retrieval_date)
    chunks += ground_chunks(act, root, retrieval_date)
    chunks += whole_schedule_chunks(act, root, retrieval_date)
    # chunk_id (e.g. "section-13") is only unique within one Act's XML -
    # namespace it so ids stay unique across the whole corpus (used as the
    # vector store's document id downstream).
    for c in chunks:
        c.chunk_id = f"{act['slug']}:{c.chunk_id}"
    return chunks


def main() -> None:
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "legislation"
    out_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "legislation_chunks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_date = datetime.date.today().isoformat()

    all_chunks: list[Chunk] = []
    for act in ACTS:
        xml_path = raw_dir / f"{act['slug']}.xml"
        if not xml_path.exists():
            print(f"[skip] missing {xml_path}")
            continue
        chunks = parse_act(act, xml_path, retrieval_date)
        print(f"{act['slug']}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    with out_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps({
                "chunk_id": c.chunk_id,
                "source_type": c.source_type,
                "act_title": c.act_title,
                "citation": c.citation,
                "heading": c.heading,
                "url": c.url,
                "retrieval_date": c.retrieval_date,
                "jurisdiction": "England",
                "text": c.text,
                **({"part_title": c.part_title} if c.part_title else {}),
                **c.extra,
            }, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_chunks)} legislation chunks to {out_path}")


if __name__ == "__main__":
    main()
