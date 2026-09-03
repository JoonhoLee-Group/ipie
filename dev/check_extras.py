"""Check that every extra advertised in the docs exists in setup.py and is non-empty.

Usage (nothing is installed):
    python -m pip install --dry-run --no-deps --quiet --report - -e . | python dev/check_extras.py
"""

import json
import os
import re
import sys

DOCS = ["README.rst", "docs/source/installation.rst"]
# Matches `pip install -e .[gpu]`, `pip install -e ".[mpi,torch]"` and `pip install "ipie[mpi]"`.
ADVERTISED = re.compile(r"""pip install (?:-e )?["']?(?:\.|ipie)\[([\w,\s]+)\]""")


def main() -> int:
    meta = json.load(sys.stdin)["install"][0]["metadata"]
    provided = set(meta.get("provides_extra", []))
    reqs = {}
    for req in meta.get("requires_dist", []):
        match = re.search(r"""extra\s*==\s*["'](\w+)["']""", req)
        if match:
            reqs.setdefault(match.group(1), []).append(req.split(";")[0].strip())
    advertised = set()
    for doc in DOCS:
        if os.path.exists(doc):
            with open(doc, encoding="utf-8") as fh:
                for group in ADVERTISED.findall(fh.read()):
                    advertised.update(name.strip() for name in group.split(","))

    for name in sorted(provided):
        print(f"[{name}] {', '.join(reqs.get(name, [])) or '(empty)'}")

    problems = []
    for name in sorted(advertised):
        if name not in provided:
            problems.append(f"extra '{name}' is advertised in the docs but not defined in setup.py")
        elif not reqs.get(name):
            problems.append(f"extra '{name}' is defined but empty (check dev/{name}.txt)")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("extras ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
