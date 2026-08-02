"""
convert_paper_to_pdf.py
========================
Converts docs/ECABSD_RESEARCH_PAPER.md into a high-quality PDF manuscript
(docs/ECABSD_RESEARCH_PAPER.pdf) using ReportLab.
"""

import os
import sys
import re
from pathlib import Path

try:
    import markdown
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


def parse_markdown_to_reportlab_elements(md_path: str):
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12,
        alignment=1, # Center
    )

    authors_style = ParagraphStyle(
        "DocAuthors",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=15,
        alignment=1, # Center
    )

    h1_style = ParagraphStyle(
        "DocH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=15,
        spaceAfter=4,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#2D3748"),
    )

    elements = []
    lines = text.split("\n")
    in_table = False
    table_lines = []

    for line in lines:
        stripped = line.strip()

        # Handle Markdown Tables
        if "|" in stripped and (stripped.startswith("|") or stripped.endswith("|")):
            in_table = True
            table_lines.append(stripped)
            continue
        elif in_table:
            # Process accumulated table
            if table_lines:
                table_flowable = process_table_lines(table_lines, table_header_style, table_cell_style)
                if table_flowable:
                    elements.append(Spacer(1, 6))
                    elements.append(table_flowable)
                    elements.append(Spacer(1, 6))
            in_table = False
            table_lines = []

        if not stripped:
            continue

        # Handle Markdown Images ![caption](path)
        if stripped.startswith("![") and "](" in stripped and stripped.endswith(")"):
            caption = stripped[2:stripped.index("](")]
            img_rel_path = stripped[stripped.index("](") + 2:-1]
            img_full_path = os.path.join("docs", img_rel_path) if not os.path.isabs(img_rel_path) else img_rel_path
            if not os.path.exists(img_full_path):
                img_full_path = os.path.join(os.path.dirname(md_path), img_rel_path)
            
            if os.path.exists(img_full_path):
                try:
                    from reportlab.platypus import Image as RLImage
                    # Calculate aspect ratio
                    img_flowable = RLImage(img_full_path, width=450, height=220)
                    elements.append(Spacer(1, 6))
                    elements.append(img_flowable)
                    caption_style = ParagraphStyle("ImgCaption", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=8.5, textColor=colors.HexColor("#4A5568"), alignment=1, spaceAfter=8)
                    elements.append(Paragraph(caption, caption_style))
                    elements.append(Spacer(1, 6))
                    continue
                except Exception as img_err:
                    print(f"[PDF Warning] Could not render image {img_full_path}: {img_err}")

        # Horizontal Rule
        if stripped in ["---", "***", "___"]:
            elements.append(Spacer(1, 4))
            elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#E2E8F0"), spaceAfter=8))
            continue

        # Title (# )
        if stripped.startswith("# "):
            title_text = clean_formatting(stripped[2:])
            elements.append(Paragraph(title_text, title_style))
            elements.append(Spacer(1, 4))
            continue

        # H2 / H3 (## , ### )
        if stripped.startswith("## "):
            h_text = clean_formatting(stripped[3:])
            elements.append(Paragraph(h_text, h1_style))
            continue
        if stripped.startswith("### "):
            h_text = clean_formatting(stripped[4:])
            elements.append(Paragraph(h_text, h2_style))
            continue

        # Authors / Affiliations
        if stripped.startswith("**Authors:**") or stripped.startswith("¹ *") or stripped.startswith("**Mentor:**"):
            formatted = clean_formatting(stripped)
            elements.append(Paragraph(formatted, authors_style))
            continue

        # Bullet points (* or - )
        if stripped.startswith("* ") or stripped.startswith("- "):
            b_text = clean_formatting(stripped[2:])
            elements.append(Paragraph(f"• {b_text}", bullet_style))
            continue

        # Regular Body Paragraph
        p_text = clean_formatting(stripped)
        elements.append(Paragraph(p_text, body_style))

    # Process trailing table if any
    if in_table and table_lines:
        table_flowable = process_table_lines(table_lines, table_header_style, table_cell_style)
        if table_flowable:
            elements.append(Spacer(1, 6))
            elements.append(table_flowable)
            elements.append(Spacer(1, 6))

    return elements


def clean_formatting(text: str) -> str:
    # Convert markdown bold/italics/math to ReportLab XML tags
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.*?)`", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"\$(.*?)\$", r"<i>\1</i>", text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Restore converted tags
    text = text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    text = text.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
    text = text.replace("&lt;font face='Courier'&gt;", "<font face='Courier'>").replace("&lt;/font&gt;", "</font>")
    return text


def process_table_lines(lines, header_style, cell_style):
    data = []
    for line in lines:
        if "---" in line:
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if cells:
            data.append(cells)

    if not data:
        return None

    table_data = []
    for row_idx, row in enumerate(data):
        row_cells = []
        for cell in row:
            style = header_style if row_idx == 0 else cell_style
            cell_text = clean_formatting(cell)
            row_cells.append(Paragraph(cell_text, style))
        table_data.append(row_cells)

    t = Table(table_data, hAlign="CENTER")
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2B6CB0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
    ]))
    return t


def main():
    md_path = "docs/ECABSD_RESEARCH_PAPER.md"
    pdf_path = "docs/ECABSD_RESEARCH_PAPER.pdf"

    print(f"[PDF Build] Parsing '{md_path}'...")
    elements = parse_markdown_to_reportlab_elements(md_path)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    doc.build(elements)
    print(f"[PDF Build] [SUCCESS] Successfully compiled PDF: '{pdf_path}'")


if __name__ == "__main__":
    main()
