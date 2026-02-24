import os
import subprocess
from pathlib import Path

LATEX_HEADER = r"""
\usepackage{titling}
\pretitle{\begin{center}\vspace*{\fill}\Huge\bfseries}
\posttitle{\end{center}}
\preauthor{\begin{center}\large}
\postauthor{\end{center}\vspace*{\fill}\clearpage}
"""

def convert_md_to_pdf():
    # 1. Create the header file
    with open("header.tex", "w") as f:
        f.write(LATEX_HEADER)
    
    # 2. Create a file to force a page break after the TOC
    with open("after_toc.tex", "w") as f:
        f.write(r"\clearpage")

    for path in Path(".").rglob("Compiled Notes.md"):
        course_name = path.parent.name
        output_path = path.with_suffix(".pdf")
        
        print(f"Converting: {path} for Course: {course_name}")

        cmd = [
            "pandoc", str(path),
            "-o", str(output_path),
            "--pdf-engine=xelatex",
            "--include-in-header=header.tex",
            # This is the secret sauce: it puts the page break right after the TOC
            "--include-before-body=after_toc.tex", 
            "--variable", f"title={course_name}",
            "--variable", "author=Compiled Revision Notes",
            "--variable", "geometry:margin=1in",
            "--variable", "fontsize=11pt",
            "--toc",
            "--toc-depth=2",
            "--variable", "toc-title=Table of Contents",
            "-V", "colorlinks=true",
            "-V", "linkcolor=blue",
            "-V", "urlcolor=blue"
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully created {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {path}: {e}")

if __name__ == "__main__":
    convert_md_to_pdf()