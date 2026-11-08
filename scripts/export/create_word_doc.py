"""
create_word_doc.py
------------------
Converts docs/master_research_report.md into a native, standalone Microsoft Word (.docx) document
with all 17 high-resolution figures embedded directly inside the document.
"""

import os
import re
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

ROOT = Path(__file__).resolve().parents[2]
MD_FILE = ROOT / "docs" / "master_research_report.md"
OUT_DOCX = ROOT / "docs" / "master_research_report.docx"
ARTIFACT_DOCX = Path("/home/vishrut/.gemini/antigravity/brain/269ec839-7b13-4c7b-a244-7e54d8464da8/master_research_report.docx")

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_styled_table(doc, lines):
    header_cols = [c.strip() for c in lines[0].strip("|").split("|")]
    data_rows = []
    for line in lines[2:]:
        cols = [c.strip() for c in line.strip("|").split("|")]
        data_rows.append(cols)

    table = doc.add_table(rows=len(data_rows) + 1, cols=len(header_cols))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # Style Header
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(header_cols):
        clean_title = re.sub(r'[\*\$`]', '', title)
        hdr_cells[i].text = clean_title
        set_cell_background(hdr_cells[i], "F1F5F9")
        set_cell_margins(hdr_cells[i])
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.bold = True
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

    # Fill Rows
    for r_idx, row_data in enumerate(data_rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            if c_idx < len(row_cells):
                clean_val = re.sub(r'[\*\$`]', '', val)
                row_cells[c_idx].text = clean_val
                set_cell_background(row_cells[c_idx], bg_color)
                set_cell_margins(row_cells[c_idx])
                p = row_cells[c_idx].paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(9.0)
                    if "**" in val:
                        run.font.bold = True

    doc.add_paragraph() # Spacing

def build_docx():
    doc = Document()

    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    with open(MD_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Header 1
        if line.startswith("# "):
            h = doc.add_heading(line[2:].strip(), level=1)
            h.runs[0].font.color.rgb = RGBColor(0x1A, 0x20, 0x2C)
            i += 1
            continue

        # Header 2
        if line.startswith("## "):
            h = doc.add_heading(line[3:].strip(), level=2)
            h.runs[0].font.color.rgb = RGBColor(0x2A, 0x9D, 0x8F)
            i += 1
            continue

        # Header 3
        if line.startswith("### "):
            h = doc.add_heading(line[4:].strip(), level=3)
            h.runs[0].font.color.rgb = RGBColor(0x33, 0x41, 0x55)
            i += 1
            continue

        # Image ![alt](path)
        img_match = re.match(r'!\[(.*?)\]\((.*?)\)', line)
        if img_match:
            alt = img_match.group(1)
            img_path = img_match.group(2).strip()
            if os.path.exists(img_path):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run()
                run.add_picture(img_path, width=Inches(6.0))
                cap = doc.add_paragraph(alt)
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in cap.runs:
                    r.font.italic = True
                    r.font.size = Pt(8.5)
                    r.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
            i += 1
            continue

        # Markdown Table
        if line.startswith("|") and i + 1 < len(lines) and lines[i+1].strip().startswith("|"):
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            add_styled_table(doc, table_lines)
            continue

        # Bullet List
        if line.startswith("* ") or line.startswith("- "):
            p = doc.add_paragraph(style='List Bullet')
            clean_item = re.sub(r'[\*\$`]', '', line[2:].strip())
            p.add_run(clean_item)
            i += 1
            continue

        # Blockquote
        if line.startswith("> "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.4)
            clean_q = re.sub(r'[\*\$`]', '', line[2:].strip())
            run = p.add_run(f"“{clean_q}”")
            run.font.italic = True
            run.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)
            i += 1
            continue

        # Code Block
        if line.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            run = p.add_run("\n".join(code_lines))
            run.font.name = "Consolas"
            run.font.size = Pt(8.5)
            continue

        # Normal Paragraph
        clean_para = re.sub(r'[\*\$`]', '', line)
        doc.add_paragraph(clean_para)
        i += 1

    doc.save(OUT_DOCX)
    print(f"[✓] Saved Word Document to {OUT_DOCX}")

    doc.save(ARTIFACT_DOCX)
    print(f"[✓] Copied Word Document to {ARTIFACT_DOCX}")

if __name__ == "__main__":
    build_docx()

