"""Citation verification tests. Run: python tests/test_verify.py

Pure logic, no model or index needed, so this runs in under a second and is the
cheapest guard in the repo. Both "wrong Act" cases below are real outputs
observed during stress testing, not invented examples.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from verify import unsupported_citations  # noqa: E402

HITS = [
    {"citation": "Housing Act 2004, s.215(1)",
     "text": "Housing Act 2004, s.215(1)\n\nNo section 21 notice may be given while the "
             "deposit is not held in an authorised scheme."},
    {"citation": "Housing Act 1988, s.8(4)",
     "text": "Housing Act 1988, s.8(4)\n\nThe court may not entertain proceedings unless "
             "subsection (3) is met."},
    {"citation": "Landlord and Tenant Act 1985, s.11(1)",
     "text": "Landlord and Tenant Act 1985, s.11(1)\n\nThere is implied a covenant to keep "
             "in repair."},
    {"citation": "Housing Act 1988, Sch.2 Ground 8",
     "text": "Housing Act 1988, Schedule 2, Ground 8\n\nBoth at the date of service and at "
             "the hearing at least two months rent is unpaid."},
]

# (label, answer text, should_be_flagged)
CASES = [
    # Observed: asked about s.21, the model cited a real provision to the wrong Act.
    ("wrong Act", "Section 21 notices can be challenged as per section 215 of the Housing Act 1988.", True),
    # Observed: HMO is defined in the Housing Act 2004, not the 1988 Act.
    ("wrong Act (HMO)", "An HMO is defined in the Housing Act 1988, section 254.", True),
    ("invented section", "This is governed by the Housing Act 2004, s.999.", True),
    ("wrong Act for a Ground", "Housing Act 2004, Ground 8 requires two months arrears.", True),
    # Must NOT flag: a capitalised word running into the Act title used to break
    # the match and turn a correct citation into a warning.
    ("leading word", "Under Landlord and Tenant Act 1985, s.11(1) the landlord must repair.", False),
    ("Act named after the number", "See section 215 of the Housing Act 2004.", False),
    ("mid-sentence", "The answer is that the Housing Act 1988, s.8(4) applies here.", False),
    # No Act named anywhere: unattributable is not the same as wrong.
    ("unattributable", "The landlord must comply with section 11.", False),
    ("correct Ground", "Housing Act 1988, Sch.2 Ground 8 requires two months arrears.", False),
]

# Two false positives found by running the real pipeline, not by reading the code.
APOSTROPHE_HITS = [
    {"citation": "Renters' Rights Act 2025, s.2",
     "text": "Renters' Rights Act 2025, s.2\n\nAssured shorthold tenancies are abolished."},
    {"citation": "Housing Act 2004, s.215(10)",
     "text": "Housing Act 2004, s.215(10)\n\nNo section 21 notice may be given."},
]

EXTRA_CASES = [
    # The model writes a typographic apostrophe; the corpus citation uses ASCII.
    ("curly apostrophe", ["Under the Renters’ Rights Act 2025, s.2, shortholds are abolished."], False),
    # Fields used to be concatenated, putting "section 21" at the end of one field
    # within lookback range of an Act named at the start of the next.
    ("cross-field attribution",
     ["Section 21 notices can still be served.", "Housing Act 2004, s.215(10)"], False),
    ("still catches a real error", ["Renters' Rights Act 2025, s.254 applies."], True),
]

# A chunk headed by one Act whose body cross-refers to a section of another. The
# "of the <Act>" immediately after the number owns it; attributing it to the
# heading's Act instead made the verifier vouch for a provision that doesn't
# exist, which is the exact error it is there to catch.
CROSS_REF_HITS = [
    {"citation": "Housing Act 1988, s.8(5B)",
     "text": "Housing Act 1988, s.8(5B)\n\nsection 215 of the Housing Act 2004 applies"},
]

CROSS_REF_CASES = [
    ("cross-Act reference, wrong Act claimed",
     ["Ensure the deposit is held as required by section 215 of the Housing Act 1988."], True),
    ("cross-Act reference, right Act claimed",
     ["Ensure the deposit is held as required by section 215 of the Housing Act 2004."], False),
]


def test_citation_verification_cases():
    """pytest entry point; the script form below is kept for the README command."""
    assert main() == 0


def main() -> int:
    checks = ([(label, [text], should, HITS) for label, text, should in CASES]
              + [(label, fields, should, APOSTROPHE_HITS)
                 for label, fields, should in EXTRA_CASES]
              + [(label, fields, should, CROSS_REF_HITS)
                 for label, fields, should in CROSS_REF_CASES])
    failures = 0
    for label, fields, should_flag, hits in checks:
        flagged = unsupported_citations(fields, hits)
        if bool(flagged) != should_flag:
            expected = "flagged" if should_flag else "clean"
            print(f"FAIL  {label}: expected {expected}, got {flagged}")
            failures += 1
        else:
            print(f"ok    {label}")
    print(f"\n{len(checks) - failures}/{len(checks)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
