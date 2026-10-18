"""Check the 30 recently verified IEEE records against BibTeX and manuscript citations.

This offline check preserves verified publisher-deposited metadata. It does not
pretend that a successful parse is a new network or full-text verification.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit():
    manifest = json.loads((ROOT / "paper/recent_ieee_references.json").read_text())
    bibliography = (ROOT / "paper/references.bib").read_text()
    manuscript = (ROOT / "paper/manuscript.tex").read_text()
    records = manifest["records"]
    if len(records) != 30 or len({r["doi"].lower() for r in records}) != 30:
        raise ValueError("Require exactly 30 unique recent IEEE records")
    entries = {
        key: body
        for key, body in re.findall(r"@\w+\{([^,]+),(.*?)(?=\n@|\Z)", bibliography, flags=re.S)
    }
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", manuscript)
        for key in group.split(",")
    }
    for row in records:
        if not 2020 <= row["year"] <= 2026 or "IEEE" not in row["publisher"]:
            raise ValueError(f"Invalid recent IEEE publication: {row['key']}")
        if not row["doi"].lower().startswith("10.1109/"):
            raise ValueError(f"Unexpected DOI publisher prefix: {row['key']}")
        body = entries.get(row["key"], "")
        if f"doi={{{row['doi']}}}" not in body or f"year={{{row['year']}}}" not in body:
            raise ValueError(f"BibTeX disagrees with verified DOI/year: {row['key']}")
        if row["key"] not in cited:
            raise ValueError(f"Recent reference is not discussed in the manuscript: {row['key']}")
    if cited - set(entries):
        raise ValueError(f"Missing BibTeX entries: {sorted(cited - set(entries))}")
    return {"recent_ieee_references": len(records), "all_cited": True, "passed": True}


if __name__ == "__main__":
    print(json.dumps(audit()))
