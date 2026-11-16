"""Fetch the curated cases from Find Case Law (caselaw.nationalarchives.gov.uk)
and save raw XML (judgment in LegalDocML/Akoma Ntoso format) locally.
"""
from pathlib import Path

import time

from fetch import WafBrowser
from acts import CASES

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "cases"


def main() -> None:
    with WafBrowser() as browser:
        for case in CASES:
            dest = RAW_DIR / f"{case['slug']}.xml"
            if dest.exists():
                print(f"{case['slug']}: [skip, cached]")
                continue
            html_url = f"https://caselaw.nationalarchives.gov.uk/{case['path']}"
            xml_url = f"{html_url}/data.xml"
            print(f"{case['slug']}: fetching {xml_url}")
            content = browser.get_via_inpage_fetch(html_url, xml_url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            time.sleep(2.0)


if __name__ == "__main__":
    main()
