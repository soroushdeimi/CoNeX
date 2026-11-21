"""Run every generated CoNeX code snippet and report PASS/FAIL.

Each plate's "CoNeX code" box is written to site/tests/snippets/<id>.py by
site/tests/plates.test.mjs. This script runs each one with the CoNeX venv
python in a subprocess (60 s limit) and prints PASS/FAIL per file with the last
error line. Exits 1 if any snippet fails.

Run:
    /Users/soroushdeimi/Desktop/myprojects/.conex-site-venv/bin/python scripts/site/check_snippets.py
"""

import os
import subprocess
import sys

VENV_PY = "/Users/soroushdeimi/Desktop/myprojects/.conex-site-venv/bin/python"
TIMEOUT = 60

HERE = os.path.dirname(os.path.abspath(__file__))
SNIPPETS = os.path.normpath(os.path.join(HERE, "..", "..", "site", "tests", "snippets"))


def last_error_line(output):
    lines = [ln for ln in output.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else "(no output)"


def main():
    if not os.path.isdir(SNIPPETS):
        print(f"snippets dir not found: {SNIPPETS}")
        return 1

    files = sorted(f for f in os.listdir(SNIPPETS) if f.endswith(".py"))
    if not files:
        print(f"no snippets found in {SNIPPETS}")
        return 1

    failures = 0
    for name in files:
        path = os.path.join(SNIPPETS, name)
        try:
            proc = subprocess.run(
                [VENV_PY, path],
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
            )
            if proc.returncode == 0:
                print(f"PASS  {name}")
            else:
                failures += 1
                err = last_error_line(proc.stderr or proc.stdout)
                print(f"FAIL  {name}: {err}")
        except subprocess.TimeoutExpired:
            failures += 1
            print(f"FAIL  {name}: timed out after {TIMEOUT}s")

    print(f"\n{len(files) - failures}/{len(files)} snippets passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
