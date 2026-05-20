import markdown
import os

with open('RESEARCH_PAPER_DRAFT.md', 'r', encoding='utf-8') as f:
    md_text = f.read()

html_content = markdown.markdown(md_text, extensions=['tables'])

css = """
<style>
    body {
        font-family: 'Times New Roman', Times, serif;
        line-height: 1.6;
        color: #000;
        max-width: 800px;
        margin: 40px auto;
        padding: 40px;
        background-color: #fff;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        text-align: justify;
    }
    h1 {
        text-align: center;
        font-size: 24pt;
        margin-bottom: 5px;
    }
    h2 {
        font-size: 16pt;
        border-bottom: 1px solid #000;
        padding-bottom: 5px;
        margin-top: 30px;
    }
    h3 {
        font-size: 14pt;
        margin-top: 20px;
        font-style: italic;
    }
    p {
        font-size: 12pt;
        margin-bottom: 15px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 11pt;
    }
    th, td {
        border: 1px solid #000;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
    }
    code {
        font-family: 'Courier New', Courier, monospace;
        font-size: 11pt;
    }
    strong {
        font-weight: bold;
    }
</style>
"""

full_html = f'<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n<title>ECABSD Research Paper</title>\n{css}\n</head>\n<body>\n{html_content}\n</body>\n</html>'

with open('RESEARCH_PAPER_DRAFT.html', 'w', encoding='utf-8') as f:
    f.write(full_html)
