"""Export assembled book to Markdown and PDF."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import markdown as md_lib
from weasyprint import HTML

from src.book.structurer import Chapter
from src.config import Config

logger = logging.getLogger(__name__)

# ── Professional CSS for the PDF ─────────────────────────────────────

_BOOK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,400&family=Source+Code+Pro:wght@400;600&display=swap');

@page {
    size: A4;
    margin: 2.5cm 2cm 2.5cm 2cm;

    @bottom-center {
        content: counter(page);
        font-family: 'Merriweather', Georgia, serif;
        font-size: 10pt;
        color: #666;
    }
}

@page :first {
    @bottom-center { content: none; }
}

body {
    font-family: 'Merriweather', Georgia, serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #1a1a1a;
    text-align: justify;
    hyphens: auto;
}

h1 {
    font-size: 26pt;
    font-weight: 700;
    color: #111;
    margin-top: 3cm;
    margin-bottom: 1cm;
    page-break-before: always;
    border-bottom: 2px solid #333;
    padding-bottom: 0.3cm;
}

h1:first-of-type {
    page-break-before: auto;
}

h2 {
    font-size: 18pt;
    font-weight: 700;
    color: #222;
    margin-top: 1.5cm;
    margin-bottom: 0.5cm;
}

h3 {
    font-size: 14pt;
    font-weight: 700;
    color: #333;
    margin-top: 1cm;
    margin-bottom: 0.3cm;
}

h4 {
    font-size: 12pt;
    font-weight: 700;
    color: #444;
    margin-top: 0.8cm;
    margin-bottom: 0.2cm;
}

p {
    margin-bottom: 0.4cm;
    text-indent: 0;
}

code {
    font-family: 'Source Code Pro', 'Courier New', monospace;
    font-size: 9.5pt;
    background-color: #f5f5f5;
    padding: 1px 4px;
    border-radius: 3px;
    color: #c7254e;
}

pre {
    background-color: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 0.5cm 0;
    overflow-x: auto;
    page-break-inside: avoid;
}

pre code {
    background: none;
    padding: 0;
    color: #333;
    font-size: 9pt;
    line-height: 1.5;
}

blockquote {
    border-left: 4px solid #666;
    margin: 0.5cm 0;
    padding: 0.3cm 1cm;
    background-color: #fafafa;
    font-style: italic;
    color: #555;
}

ul, ol {
    margin: 0.3cm 0;
    padding-left: 1.2cm;
}

li {
    margin-bottom: 0.15cm;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.5cm 0;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #ccc;
    padding: 6px 10px;
    text-align: left;
    font-size: 10pt;
}

th {
    background-color: #f0f0f0;
    font-weight: 700;
}

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 1cm 0;
}

.title-page {
    text-align: center;
    padding-top: 8cm;
}

.title-page h1 {
    font-size: 36pt;
    border: none;
    page-break-before: auto;
    margin-top: 0;
}

.title-page p {
    font-size: 14pt;
    color: #555;
    text-indent: 0;
}

.toc {
    page-break-after: always;
}

.toc h1 {
    page-break-before: auto;
}

.toc ul {
    list-style: none;
    padding-left: 0;
}

.toc li {
    margin-bottom: 0.3cm;
    font-size: 12pt;
}
"""


class MarkdownExporter:
    """Assemble chapters into a single Markdown manuscript."""

    def __init__(self, config: Config) -> None:
        self._output_path = Path(config.output.manuscript_md)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        chapters: List[Chapter],
        book_title: str = "Building LLMs from Scratch",
    ) -> str:
        """Write combined Markdown to disk and return the content."""
        parts: List[str] = []

        # Title page
        parts.append(f"# {book_title}\n")
        parts.append("*Generated from YouTube Playlist*\n")
        parts.append("---\n")

        # Table of Contents
        parts.append("## Table of Contents\n")
        for ch in chapters:
            parts.append(f"- **Chapter {ch.number}**: {ch.title}")
        parts.append("\n---\n")

        # Chapters
        for ch in chapters:
            parts.append(f"\n{ch.content}\n")
            parts.append("\n---\n")

        manuscript = "\n".join(parts)

        with open(self._output_path, "w") as fh:
            fh.write(manuscript)

        logger.info("Markdown manuscript written to %s", self._output_path)
        return manuscript


class PDFExporter:
    """Convert Markdown manuscript to a styled PDF."""

    def __init__(self, config: Config) -> None:
        self._output_path = Path(config.output.manuscript_pdf)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        markdown_content: str,
        book_title: str = "Building LLMs from Scratch",
    ) -> None:
        """Render Markdown → HTML → PDF with professional styling."""
        # Convert Markdown to HTML
        extensions = [
            "extra",
            "codehilite",
            "toc",
            "tables",
            "fenced_code",
            "sane_lists",
        ]
        html_body = md_lib.markdown(
            markdown_content, extensions=extensions, output_format="html5"
        )

        # Wrap in full HTML document
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{book_title}</title>
    <style>{_BOOK_CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

        # Generate PDF
        HTML(string=html_doc).write_pdf(str(self._output_path))
        logger.info("PDF manuscript written to %s", self._output_path)
