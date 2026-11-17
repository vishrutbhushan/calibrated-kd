"""
Converts docs/master_research_report.md into:
1. docs/master_research_report.docx (Formatted Word Document)
2. docs/master_research_report.html (Stand-alone HTML document with KaTeX and CSS styling)
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
MD_PATH = ROOT / "docs" / "master_research_report.md"
DOCX_PATH = ROOT / "docs" / "master_research_report.docx"
HTML_PATH = ROOT / "docs" / "master_research_report.html"

def make_docx():
    doc = docx.Document()
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    with open(MD_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    def set_cell_background(cell, fill_hex):
        tcPr = cell._tc.get_or_add_tcPr()
        xml_str = f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'
        tcPr.append(parse_xml(xml_str))

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue

        if line.startswith("# "):
            p = doc.add_paragraph()
            run = p.add_run(line[2:].strip())
            run.bold = True
            run.font.size = Pt(18)
            run.font.color.rgb = RGBColor(17, 24, 39)
            p.paragraph_format.space_after = Pt(6)
        elif line.startswith("## "):
            p = doc.add_paragraph()
            run = p.add_run(line[3:].strip())
            run.bold = True
            run.font.size = Pt(14)
            run.font.color.rgb = RGBColor(30, 58, 138)
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
        elif line.startswith("### "):
            p = doc.add_paragraph()
            run = p.add_run(line[4:].strip())
            run.bold = True
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(55, 65, 81)
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
        elif line.startswith("#### "):
            p = doc.add_paragraph()
            run = p.add_run(line[5:].strip())
            run.bold = True
            run.font.size = Pt(10.5)
            run.font.color.rgb = RGBColor(55, 65, 81)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
        elif line.startswith("|") and "---" not in line:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if "---" not in lines[i]:
                    table_lines.append([c.strip() for c in lines[i].strip().split("|")[1:-1]])
                i += 1
            i -= 1
            if table_lines:
                num_cols = len(table_lines[0])
                table = doc.add_table(rows=len(table_lines), cols=num_cols)
                table.style = "Table Grid"
                for r_idx, row in enumerate(table_lines):
                    for c_idx, val in enumerate(row):
                        if c_idx < num_cols:
                            cell = table.cell(r_idx, c_idx)
                            clean_val = re.sub(r"\*\*([^*]+)\*\*", r"\1", val)
                            clean_val = re.sub(r"\*([^*]+)\*", r"\1", clean_val)
                            clean_val = clean_val.replace("`", "")
                            cell.text = clean_val
                            p = cell.paragraphs[0]
                            p.paragraph_format.space_before = Pt(2)
                            p.paragraph_format.space_after = Pt(2)
                            if r_idx == 0:
                                if p.runs:
                                    p.runs[0].bold = True
                                set_cell_background(cell, "E5E7EB")
                            elif "**" in val:
                                if p.runs:
                                    p.runs[0].bold = True
                doc.add_paragraph().paragraph_format.space_after = Pt(6)
        elif line.startswith("!["):
            m = re.search(r"\((.*?)\)", line)
            if m:
                img_rel = m.group(1)
                img_path = (ROOT / "docs" / img_rel).resolve()
                if not img_path.exists():
                    img_path = (ROOT / "figures" / "publication" / Path(img_rel).name).resolve()
                if img_path.exists():
                    doc.add_picture(str(img_path), width=Inches(6.2))
                    last_p = doc.paragraphs[-1]
                    last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    last_p.paragraph_format.space_after = Pt(4)
        elif line.startswith("* ") or line.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            clean_text = line[2:].strip()
            p.add_run(clean_text)
            p.paragraph_format.space_after = Pt(2)
        elif re.match(r"^\d+\.\s", line):
            p = doc.add_paragraph(style="List Number")
            clean_text = re.sub(r"^\d+\.\s*", "", line).strip()
            p.add_run(clean_text)
            p.paragraph_format.space_after = Pt(2)
        elif line.startswith("$$"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(line.replace("$$", "").strip())
            run.italic = True
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
        elif line == "---":
            p = doc.add_paragraph()
            run = p.add_run("_________________________________________________________________________________")
            run.font.color.rgb = RGBColor(209, 213, 219)
            p.paragraph_format.space_after = Pt(6)
        else:
            p = doc.add_paragraph()
            p.add_run(line)
            p.paragraph_format.space_after = Pt(4)
        i += 1

    doc.save(str(DOCX_PATH))
    print("[✓] Saved DOCX:", DOCX_PATH)


def make_html():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Simple clean HTML generator with styling
    html_content = []
    html_content.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>An Empirical Study of Verbalizer Prior Transfer in Label-Free Knowledge Distillation</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #1f2937; max-width: 960px; margin: 40px auto; padding: 0 20px; background: #fafafa; }
    .container { background: #ffffff; padding: 40px 50px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }
    h1 { font-size: 26px; color: #111827; border-bottom: 2px solid #e5e7eb; padding-bottom: 12px; margin-top: 0; }
    h2 { font-size: 20px; color: #1e3a8a; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; margin-top: 32px; }
    h3 { font-size: 16px; color: #374151; margin-top: 24px; }
    h4 { font-size: 14px; color: #4b5563; margin-top: 16px; }
    p, li { font-size: 14.5px; color: #374151; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 13.5px; }
    th, td { border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }
    th { background: #f3f4f6; font-weight: 600; color: #111827; }
    tr:nth-child(even) { background: #f9fafb; }
    img { max-width: 100%; height: auto; display: block; margin: 20px auto; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    blockquote { border-left: 4px solid #3b82f6; margin: 16px 0; padding: 8px 16px; background: #eff6ff; color: #1e40af; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; background: #f3f4f6; padding: 2px 6px; border-radius: 4px; }
    hr { border: 0; border-top: 1px solid #e5e7eb; margin: 30px 0; }
    .best { background-color: #dcf5e1 !important; color: #147828; font-weight: bold; }
    .worst { background-color: #ffe1e1 !important; color: #be1923; }
</style>
</head>
<body>
<div class="container">
""")

    lines = md_text.splitlines()
    i = 0
    in_table = False
    table_rows = []

    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            if in_table:
                # render table
                html_content.append("<table>")
                for r_idx, row in enumerate(table_rows):
                    tag = "th" if r_idx == 0 else "td"
                    html_content.append("  <tr>")
                    for cell in row:
                        cell_clean = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", cell)
                        cell_clean = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", cell_clean)
                        cell_clean = re.sub(r"`([^`]+)`", r"<code>\1</code>", cell_clean)
                        html_content.append(f"    <{tag}>{cell_clean}</{tag}>")
                    html_content.append("  </tr>")
                html_content.append("</table>")
                in_table = False
                table_rows = []
            i += 1
            continue

        if line.startswith("|") and "---" not in line:
            in_table = True
            cells = [c.strip() for c in line.split("|")[1:-1]]
            table_rows.append(cells)
            i += 1
            continue
        elif line.startswith("|") and "---" in line:
            i += 1
            continue

        if in_table:
            html_content.append("<table>")
            for r_idx, row in enumerate(table_rows):
                tag = "th" if r_idx == 0 else "td"
                html_content.append("  <tr>")
                for cell in row:
                    cell_clean = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", cell)
                    cell_clean = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", cell_clean)
                    cell_clean = re.sub(r"`([^`]+)`", r"<code>\1</code>", cell_clean)
                    html_content.append(f"    <{tag}>{cell_clean}</{tag}>")
                html_content.append("  </tr>")
            html_content.append("</table>")
            in_table = False
            table_rows = []

        if line.startswith("# "):
            html_content.append(f"<h1>{line[2:].strip()}</h1>")
        elif line.startswith("## "):
            html_content.append(f"<h2>{line[3:].strip()}</h2>")
        elif line.startswith("### "):
            html_content.append(f"<h3>{line[4:].strip()}</h3>")
        elif line.startswith("#### "):
            html_content.append(f"<h4>{line[5:].strip()}</h4>")
        elif line.startswith("!["):
            m = re.search(r"!\[(.*?)\]\((.*?)\)", line)
            if m:
                alt = m.group(1)
                src = m.group(2)
                html_content.append(f'<img src="{src}" alt="{alt}">')
        elif line.startswith("* ") or line.startswith("- "):
            item = line[2:].strip()
            item = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", item)
            item = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", item)
            item = re.sub(r"`([^`]+)`", r"<code>\1</code>", item)
            html_content.append(f"<ul><li>{item}</li></ul>")
        elif line.startswith("$$"):
            eq = line.replace("$$", "").strip()
            html_content.append(f'<p style="text-align:center; font-style:italic;"><code>{eq}</code></p>')
        elif line == "---":
            html_content.append("<hr>")
        else:
            p_line = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", line)
            p_line = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", p_line)
            p_line = re.sub(r"`([^`]+)`", r"<code>\1</code>", p_line)
            html_content.append(f"<p>{p_line}</p>")
        i += 1

    html_content.append("""</div>
</body>
</html>""")

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))
    print("[✓] Saved HTML:", HTML_PATH)


if __name__ == "__main__":
    make_docx()
    make_html()
