"""
create_single_file_html.py
--------------------------
Converts docs/master_research_report.md into a 100% self-contained standalone HTML document
with all 17 high-resolution charts embedded directly as Base64 strings.
Zero external image dependencies.
Can be opened in any browser, emailed, printed to PDF (Ctrl+P), or copy-pasted into Word/Google Docs.
"""

import base64
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MD_FILE = ROOT / "docs" / "master_research_report.md"
OUT_HTML = ROOT / "docs" / "master_research_report.html"
ARTIFACT_HTML = Path("/home/vishrut/.gemini/antigravity/brain/269ec839-7b13-4c7b-a244-7e54d8464da8/master_research_report.html")

def md_table_to_html(table_text):
    lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
    if len(lines) < 2:
        return table_text
    
    header_cols = [c.strip() for c in lines[0].strip("|").split("|")]
    html = ['<div class="table-container"><table><thead><tr>']
    for c in header_cols:
        html.append(f'<th>{c}</th>')
    html.append('</tr></thead><tbody>')
    
    for row in lines[2:]:
        cols = [c.strip() for c in row.strip("|").split("|")]
        html.append('<tr>')
        for c in cols:
            html.append(f'<td>{c}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    return "\n".join(html)

def convert():
    with open(MD_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Base64 encode all images
    def replace_img(match):
        alt = match.group(1)
        path = match.group(2).strip()
        if os.path.exists(path):
            with open(path, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("utf-8")
            return f'<div class="figure-card"><img src="data:image/png;base64,{b64}" alt="{alt}" /><div class="figure-caption">{alt}</div></div>'
        return f'<p><em>[Missing Image: {alt}]</em></p>'

    content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_img, content)

    # 2. Convert markdown tables
    table_pattern = re.compile(r'((?:\|[^\n]+\|\n)+)', re.MULTILINE)
    content = table_pattern.sub(lambda m: md_table_to_html(m.group(1)), content)

    # 3. Headers
    content = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
    content = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
    content = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)

    # 4. Math blocks ($$...$$) and inline math ($...$)
    content = re.sub(r'\$\$(.*?)\$\$', r'<div class="math-block">\[\1\]</div>', content, flags=re.DOTALL)
    content = re.sub(r'(?<!\$)\$([^\$\n]+)\$(?!\$)', r'<span class="math-inline">\(\1\)</span>', content)

    # 5. Bold & Italics
    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)

    # 6. Code blocks
    content = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', content, flags=re.DOTALL)
    content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)

    # 7. Blockquotes
    content = re.sub(r'^> (.*?)$', r'<blockquote>\1</blockquote>', content, flags=re.MULTILINE)

    # 8. Unordered lists
    content = re.sub(r'^\* (.*?)$', r'<li>\1</li>', content, flags=re.MULTILINE)
    content = re.sub(r'((?:<li>.*?</li>\n?)+)', r'<ul>\1</ul>', content)

    # 9. Horizontal rules
    content = re.sub(r'^---$', r'<hr/>', content, flags=re.MULTILINE)

    # 10. Paragraphs
    blocks = content.split("\n\n")
    processed_blocks = []
    for b in blocks:
        b_strip = b.strip()
        if not b_strip:
            continue
        if b_strip.startswith(("<h1", "<h2", "<h3", "<div", "<table", "<pre", "<blockquote>", "<ul>", "<hr")):
            processed_blocks.append(b_strip)
        else:
            processed_blocks.append(f"<p>{b_strip}</p>")
    body_html = "\n".join(processed_blocks)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Null-Conditioned Verbalizer Calibration for Label-Free Task Distillation</title>
<!-- MathJax for KaTeX/LaTeX math rendering -->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  :root {{
    --primary: #2a9d8f;
    --primary-dark: #21867a;
    --text: #2d3748;
    --bg: #ffffff;
    --code-bg: #f7fafc;
    --border: #e2e8f0;
    --card-bg: #f8fafc;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.7;
    color: var(--text);
    background-color: #f1f5f9;
    margin: 0;
    padding: 30px 15px;
  }}
  .article-container {{
    max-width: 960px;
    margin: 0 auto;
    background: #ffffff;
    padding: 50px 60px;
    border-radius: 12px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
  }}
  h1 {{
    font-size: 2.2rem;
    color: #1a202c;
    border-bottom: 2px solid var(--primary);
    padding-bottom: 12px;
    margin-top: 0;
    line-height: 1.3;
  }}
  h2 {{
    font-size: 1.5rem;
    color: #2d3748;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-top: 36px;
  }}
  h3 {{
    font-size: 1.2rem;
    color: #4a5568;
    margin-top: 24px;
  }}
  p, li {{
    font-size: 1.02rem;
  }}
  blockquote {{
    border-left: 4px solid var(--primary);
    background-color: var(--card-bg);
    margin: 20px 0;
    padding: 14px 20px;
    border-radius: 0 8px 8px 0;
    color: #4a5568;
  }}
  code {{
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.9em;
    color: #d9534f;
  }}
  pre {{
    background: #1e293b;
    color: #f8fafc;
    padding: 16px 20px;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 0.9em;
  }}
  pre code {{
    background: transparent;
    border: none;
    color: inherit;
    padding: 0;
  }}
  .table-container {{
    overflow-x: auto;
    margin: 24px 0;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
    text-align: left;
  }}
  th, td {{
    padding: 10px 14px;
    border: 1px solid var(--border);
  }}
  th {{
    background-color: #f8fafc;
    font-weight: 600;
    color: #334155;
  }}
  tr:nth-child(even) td {{
    background-color: #fbfcfe;
  }}
  .figure-card {{
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
    margin: 26px 0;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
  }}
  .figure-card img {{
    max-width: 100%;
    height: auto;
    border-radius: 6px;
  }}
  .figure-caption {{
    font-size: 0.88rem;
    color: #64748b;
    font-weight: 600;
    margin-top: 8px;
  }}
  .math-block {{
    margin: 16px 0;
    overflow-x: auto;
    text-align: center;
  }}
  hr {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 40px 0;
  }}
  @media print {{
    body {{
      background: #ffffff;
      padding: 0;
    }}
    .article-container {{
      box-shadow: none;
      padding: 0;
      max-width: 100%;
    }}
    .figure-card {{
      page-break-inside: avoid;
      box-shadow: none;
      border: 1px solid #ccc;
    }}
    table {{
      page-break-inside: avoid;
    }}
  }}
</style>
</head>
<body>
<div class="article-container">
{body_html}
</div>
</body>
</html>
"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"[✓] Saved standalone HTML to {OUT_HTML} ({len(full_html) / 1024 / 1024:.2f} MB)")

    with open(ARTIFACT_HTML, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"[✓] Copied standalone HTML to {ARTIFACT_HTML}")

if __name__ == "__main__":
    convert()

