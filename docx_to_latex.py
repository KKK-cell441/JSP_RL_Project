#!/usr/bin/env python3
"""Convert the current L2D distill DOCX manuscript to a LaTeX draft."""
import os

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "JSP_RL_Paper_L2D_Distill.docx")
OUT = os.path.join(HERE, "JSP_RL_Paper_L2D_Distill.tex")
FIG_MAP = {
    "Figure 1.": "fig1_static_results.png",
    "Figure 2.": "fig2_generalization.png",
    "Figure 3.": "fig3_dynamic.png",
    "Figure 4.": "fig4_gantt.png",
    "Figure 5.": "fig5_gantt_hgnn.png",
}


def esc(text):
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace("[", r"{[}")
        .replace("]", r"{]}")
        .replace("^", r"\textasciicircum{}")
        .replace("~", r"\textasciitilde{}")
    )


def heading_level(para):
    name = (para.style.name or "") if para.style else ""
    if name.startswith("Heading 1"):
        return 1
    if name.startswith("Heading 2"):
        return 2
    return 0


def emit_table(table):
    rows = [[c.text.strip() for c in row.cells] for row in table.rows]
    ncols = max((len(r) for r in rows), default=1)
    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"\centering")
    out.append("\\begin{tabular}{" + "l" * ncols + "}")
    out.append(r"\toprule")
    for row in rows:
        row = row + [""] * (ncols - len(row))
        out.append(" & ".join(esc(c) for c in row) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    return out


def main():
    doc = Document(SRC)
    lines = []
    lines.append(r"""\documentclass{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[hidelinks]{hyperref}
\title{Teacher-Distilled Lightweight Graph Learning for Real-Time Dynamic Job Shop Scheduling}
\author{ZHONGKUAN MA\\ Northeast Forestry University\\ \texttt{2024212760@nefu.edu.cn}}
\date{}
\begin{document}
\maketitle
""")

    table_pending = False
    seen_heading = False
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            para = Paragraph(child, doc)
            text = para.text.strip()
            level = heading_level(para)
            if level == 1:
                seen_heading = True
                lines.append(f"\n\\section*{{{esc(text)}}}\n")
                continue
            if level == 2:
                lines.append(f"\n\\subsection*{{{esc(text)}}}\n")
                continue
            if not seen_heading:
                continue
            if not text:
                continue
            xml = para._p.xml
            if "<w:drawing>" in xml or "<pic:pic>" in xml:
                continue
            if text.startswith("Table ") and table_pending:
                lines.append(f"\\caption{{{esc(text)}}}")
                lines.append(r"\end{table}")
                lines.append("")
                table_pending = False
                continue
            if text.startswith("Figure "):
                key = text.split(".")[0] + "."
                img = FIG_MAP.get(key)
                if img:
                    lines.append(r"\begin{figure}[h]")
                    lines.append(r"\centering")
                    lines.append(f"\\includegraphics[width=0.8\\textwidth]{{{img}}}")
                    lines.append(f"\\caption{{{esc(text)}}}")
                    lines.append(r"\end{figure}")
                    lines.append("")
                continue
            lines.append(esc(text))
            lines.append("")
        elif child.tag.endswith("}tbl"):
            lines.extend(emit_table(Table(child, doc)))
            table_pending = True

    lines.append(r"\end{document}")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
