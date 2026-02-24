import os
import subprocess
from pathlib import Path

# 1. Setup the LaTeX header as a clean multi-line string
LATEX_HEADER = r"""
\usepackage{titling}
\pretitle{\begin{center}\vspace*{\fill}\Huge\bfseries}
\posttitle{\end{center}}
\preauthor{\begin{center}\large}
\postauthor{\end{center}\vspace*{\fill}\clearpage}
"""

def convert_md_to_pdf():
    # Create the temporary header file
    with open("header.tex", "w") as f:
        f.write(LATEX_HEADER)

    # Find all "Compiled Notes.md" files
    for path in Path(".").rglob("Compiled Notes.md"):
        course_name = path.parent.name
        output_path = path.with_suffix(".pdf")
        
        print(f"Converting: {path} for Course: {course_name}")

        # Construct the Pandoc command
        cmd = [
            "pandoc", str(path),
            "-o", str(output_path),
            "--pdf-engine=xelatex",
            "--include-in-header=header.tex",
            "--variable", f"title={course_name}",
            "--variable", "author=Compiled Revision Notes",
            "--variable", "geometry:margin=1in",
            "--variable", "fontsize=11pt",
            "--toc",
            "--toc-depth=2",
            "--variable", "toc-title=Table of Contents",
            "-V", "colorlinks=true",
            "-V", "linkcolor=blue",
            "-V", "urlcolor=blue",
            "-V", 'include-after=\\clearpage'
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully created {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {path}: {e}")

if __name__ == "__main__":
    convert_md_to_pdf()