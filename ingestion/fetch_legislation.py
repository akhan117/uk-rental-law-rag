"""Fetch the curated Acts from legislation.gov.uk and save raw XML locally.
Re-run is safe/cheap: already-downloaded files are skipped (see fetch.fetch_and_save).
"""
from pathlib import Path

from fetch import WafBrowser, fetch_and_save
from acts import ACTS

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "legislation"


def main() -> None:
    with WafBrowser() as browser:
        for act in ACTS:
            dest = RAW_DIR / f"{act['slug']}.xml"
            url = f"https://www.legislation.gov.uk/{act['path']}/data.xml"
            print(f"{act['slug']}:")
            fetch_and_save(browser, url, dest)


if __name__ == "__main__":
    main()
