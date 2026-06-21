#!/usr/bin/env python3
"""Convert a Markdown file to PDF using python-markdown + WeasyPrint."""
import argparse, pathlib, markdown

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Source+Code+Pro&display=swap');

@page { margin: 2.5cm 2.8cm; size: A4; }

body {
    font-family: "Georgia", "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #111;
    max-width: 100%;
}

h1 { font-size: 16pt; margin-top: 0; margin-bottom: 6pt; font-weight: bold; }
h2 { font-size: 13pt; margin-top: 22pt; margin-bottom: 4pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin-top: 14pt; margin-bottom: 3pt; }

p { margin: 0 0 8pt 0; text-align: justify; }
blockquote { margin: 8pt 0 8pt 18pt; color: #444; border-left: 3px solid #ccc; padding-left: 10pt; }

code { font-family: "Courier New", monospace; font-size: 9pt; background: #f4f4f4; padding: 1px 3px; }
pre  { font-family: "Courier New", monospace; font-size: 9pt; background: #f4f4f4;
       padding: 8pt; border: 1px solid #ddd; line-height: 1.4; white-space: pre-wrap;
       page-break-inside: avoid; }

table { border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 10pt; page-break-inside: avoid; }
th { background: #eee; font-weight: bold; text-align: left; padding: 4pt 8pt; border: 1px solid #bbb; }
td { padding: 3pt 8pt; border: 1px solid #ccc; }
tr:nth-child(even) td { background: #f9f9f9; }

em  { font-style: italic; }
strong { font-weight: bold; }

hr { border: none; border-top: 1px solid #ccc; margin: 16pt 0; }

a { color: #1a5276; text-decoration: none; }

/* numeric columns right-align */
td:nth-child(2), td:nth-child(3), td:nth-child(4), td:nth-child(5) { text-align: right; }
th:nth-child(2), th:nth-child(3), th:nth-child(4), th:nth-child(5) { text-align: right; }
"""

def convert(src: str, dst: str) -> None:
    md_text = pathlib.Path(src).read_text(encoding="utf-8")
    # strip leading "# Title\n# Subtitle" lines that render as two separate h1
    # join them into one h1 if they're both raw lines starting with "# "
    lines = md_text.split("\n")
    if len(lines) >= 2 and lines[0].startswith("# ") and lines[1].startswith("# "):
        lines[0] = lines[0] + " " + lines[1].lstrip("# ")
        lines.pop(1)
        md_text = "\n".join(lines)

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )
    full_html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    from weasyprint import HTML, CSS as WCSS
    HTML(string=full_html, base_url=".").write_pdf(dst)
    size_kb = pathlib.Path(dst).stat().st_size // 1024
    print(f"Saved → {dst}  ({size_kb} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()
    convert(args.src, args.dst)
