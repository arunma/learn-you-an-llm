#!/usr/bin/env python3
"""Build the nanochat reference book PDF from journal markdown files.

Usage:
    python scripts/build_book.py

Requires: pandoc, weasyprint (pip install weasyprint)
Outputs:  build/nanochat_reference_book.pdf
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOURNAL_DIR = ROOT / "journal"
BUILD_DIR = ROOT / "build"
CSS_FILE = Path(__file__).resolve().parent / "book.css"
OUTPUT_HTML = BUILD_DIR / "nanochat_reference_book.html"
OUTPUT_PDF = BUILD_DIR / "nanochat_reference_book.pdf"

TITLE = "nanochat Reference Book"
SUBTITLE = "Building an LLM from scratch — concepts, code, and intuition"


def find_chapters() -> list[Path]:
    """Find all journal/*.md files, sorted by numeric prefix."""
    md_files = sorted(JOURNAL_DIR.glob("*.md"))
    if not md_files:
        print(f"Error: no .md files found in {JOURNAL_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(md_files)} chapters:")
    for f in md_files:
        print(f"  {f.name}")
    return md_files


def fix_image_paths(content: str) -> str:
    """Convert image paths to absolute so both pandoc and weasyprint can find them.

    Handles both markdown images ![alt](path) and HTML <img src="path">.
    """
    journal_abs = str(JOURNAL_DIR)
    # Markdown images: make paths absolute
    content = re.sub(
        r"!\[([^\]]*)\]\((?!https?://)(?!/)([^)]+)\)",
        lambda m: f"![{m.group(1)}]({journal_abs}/{m.group(2)})",
        content,
    )
    # HTML img tags
    content = re.sub(
        r'<img\s+([^>]*?)src="(?!https?://)(?!/)([^"]+)"',
        lambda m: f'<img {m.group(1)}src="{journal_abs}/{m.group(2)}"',
        content,
    )
    return content


def add_shape_badges(content: str) -> str:
    """Wrap tensor shape annotations like (B, T, C) in <span class="shape">."""
    # Match patterns like (B, T, C) or (B, T) that look like shape annotations
    # Only inside inline code or plain text, not inside code blocks
    pattern = r"`\(([BTCNHED],?\s*)+\)`"
    content = re.sub(
        pattern,
        lambda m: f'<span class="shape">{m.group(0)[1:-1]}</span>',  # strip backticks
        content,
    )
    return content


def combine_chapters(chapters: list[Path]) -> str:
    """Concatenate chapters with page-break markers."""
    parts = []

    # Title block — pandoc will render this as the h1
    parts.append(f"# {TITLE}\n\n*{SUBTITLE}*\n")

    for i, chapter in enumerate(chapters):
        text = chapter.read_text(encoding="utf-8")
        text = fix_image_paths(text)
        text = add_shape_badges(text)
        parts.append(text)

    return "\n\n".join(parts)


def run(cmd: list[str], description: str, extra_env: dict | None = None) -> None:
    """Run a subprocess, exit on failure."""
    import os
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    env = {**os.environ, **(extra_env or {})}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"Error running {cmd[0]}:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    if result.stderr:
        # Some tools emit warnings on stderr even on success
        print(result.stderr, file=sys.stderr)


def check_tool(name: str) -> None:
    """Verify an external tool is available."""
    result = subprocess.run(["which", name], capture_output=True)
    if result.returncode != 0:
        print(f"Error: '{name}' not found. Install it first.", file=sys.stderr)
        if name == "weasyprint":
            print("  pip install weasyprint", file=sys.stderr)
        elif name == "pandoc":
            print("  brew install pandoc  (or see https://pandoc.org/installing.html)", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    check_tool("pandoc")
    check_tool("weasyprint")

    BUILD_DIR.mkdir(exist_ok=True)

    chapters = find_chapters()
    combined = combine_chapters(chapters)

    combined_md = BUILD_DIR / "combined.md"
    combined_md.write_text(combined, encoding="utf-8")
    print(f"\nCombined markdown: {combined_md} ({len(combined)} chars)")

    # Pandoc: markdown -> standalone HTML with TOC
    run(
        [
            "pandoc",
            str(combined_md),
            "-o", str(OUTPUT_HTML),
            "--standalone",
            "--css", str(CSS_FILE),
            "--highlight-style", "pygments",
            "--toc",
            "--toc-depth=3",
            "--metadata", f"title={TITLE}",
            "--from", "markdown+smart+autolink_bare_uris",
            "--resource-path", str(ROOT),
        ],
        "Pandoc: Markdown -> HTML",
    )
    print(f"HTML output: {OUTPUT_HTML}")

    # WeasyPrint: HTML -> PDF (needs Homebrew libs on macOS with conda)
    run(
        [
            "weasyprint",
            str(OUTPUT_HTML),
            str(OUTPUT_PDF),
        ],
        "WeasyPrint: HTML -> PDF",
        extra_env={"DYLD_FALLBACK_LIBRARY_PATH": "/opt/homebrew/lib"},
    )
    print(f"\nPDF output: {OUTPUT_PDF}")
    print("Done.")


if __name__ == "__main__":
    main()
