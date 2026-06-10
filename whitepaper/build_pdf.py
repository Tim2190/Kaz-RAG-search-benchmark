"""
Сборка PDF из paper.md для публикации (Zenodo / HuggingFace).

ЗАПУСК:
    pip install markdown weasyprint
    python whitepaper/build_pdf.py

Результат: whitepaper/paper.pdf
"""

import re
from pathlib import Path

import markdown
from weasyprint import HTML

HERE = Path(__file__).parent
SRC = HERE / "paper.md"
OUT = HERE / "paper.pdf"

# Первые две строки — это заголовок (две # строки склеиваем в один <h1>)
raw = SRC.read_text(encoding="utf-8")

# Склеиваем title из двух строк "# ..." в начале
lines = raw.split("\n")
if lines[0].startswith("# ") and lines[1].startswith("# "):
    title = lines[0][2:].strip() + " " + lines[1][2:].strip()
    body = "\n".join(lines[2:])
    raw = f"# {title}\n{body}"

html_body = markdown.markdown(
    raw,
    extensions=["tables", "fenced_code", "sane_lists"],
)

CSS = """
@page {
    size: A4;
    margin: 2.2cm 2cm;
    @bottom-center { content: counter(page); font-size: 9pt; color: #666; }
}
body {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 10.5pt;
    line-height: 1.5;
    color: #1a1a1a;
}
h1 {
    font-size: 17pt;
    line-height: 1.25;
    text-align: center;
    margin-bottom: 0.3em;
    font-family: 'Helvetica', Arial, sans-serif;
}
h2 {
    font-size: 13pt;
    margin-top: 1.4em;
    border-bottom: 1px solid #ccc;
    padding-bottom: 2px;
    font-family: 'Helvetica', Arial, sans-serif;
}
h3 {
    font-size: 11pt;
    margin-top: 1em;
    font-family: 'Helvetica', Arial, sans-serif;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8em 0;
    font-size: 9pt;
}
th, td {
    border: 1px solid #bbb;
    padding: 4px 7px;
    text-align: left;
}
th { background: #f0f0f0; font-family: 'Helvetica', Arial, sans-serif; }
code {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
    background: #f5f5f5;
    padding: 1px 3px;
    border-radius: 3px;
}
pre {
    background: #f5f5f5;
    padding: 8px 10px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 8.5pt;
}
pre code { background: none; padding: 0; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.2em 0; }
a { color: #0050a0; text-decoration: none; }
strong { color: #000; }
"""

full_html = f"<html><head><meta charset='utf-8'></head><body>{html_body}</body></html>"

HTML(string=full_html).write_pdf(OUT, stylesheets=[__import__("weasyprint").CSS(string=CSS)])
print(f"PDF собран → {OUT}")
