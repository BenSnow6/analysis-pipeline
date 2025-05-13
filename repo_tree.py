#!/usr/bin/env python3
"""
repo_tree.py  ──  Generate a plain-text directory tree of the current repo.

Usage
-----
    python repo_tree.py               # writes _tree.md in cwd
    python repo_tree.py /path/to/repo --out my_tree.md
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

def make_tree(root: Path) -> str:
    root = root.resolve()
    lines: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip common trash (edit to taste)
        dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules"}]

        rel = Path(dirpath).relative_to(root)
        indent = "    " * len(rel.parts)
        folder_name = rel.name if rel.parts else root.name
        lines.append(f"{indent}{folder_name}/")

        for fname in sorted(filenames):
            lines.append(f"{indent}    {fname}")

    return "\n".join(lines)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default=".", help="Repo root (default: current dir)")
    p.add_argument("--out", "-o", default="_tree.md", help="Output file name")
    args = p.parse_args()

    tree_md = f"```tree\n{make_tree(Path(args.path))}\n```"

    try:
        Path(args.out).write_text(tree_md, encoding="utf-8")
        print(f"✓ Wrote tree to {args.out}")
    except OSError as e:
        print(f"✗ Could not write file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
