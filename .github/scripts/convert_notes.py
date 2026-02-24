import os
import subprocess
from pathlib import Path

# The header uses etoolbox to patch the \tableofcontents command
# to ensure \clearpage runs immediately after it finishes.
LATEX_HEADER = r"""
\usepackage{titling}
\usepackage{etoolbox}

% 1. Format the Title Page: Center title and subtitle vertically
\pretitle{\begin{center}\vspace*{\fill}\Huge\bfseries}
\posttitle{\end{center}}
\preauthor{\begin{center}\large}
\postauthor{\end{center}\vspace*{\fill}\clearpage}

% 2. Force a page break after the Table of Contents
\patchcmd{\tableofcontents}{\endgroup}{\endgroup\clearpage}{}{}
"""

def convert_md_to_pdf():
    # 1. Create the header file
    # Using 'w' with encoding ensure clean file creation
    with open("header.tex", "w", encoding="utf-8") as f:
        f.write(LATEX_HEADER)
    
    # 2. Iterate through folders to find the notes
    for path in Path(".").rglob("Compiled Notes.md"):
        # The course name is the name of the folder containing the .md file
        course_name = path.parent.name
        output_path = path.with_suffix(".pdf")
        
        print(f"Converting: {path} for Course: {course_name}")

        # Construct the Pandoc command
        # Note: We removed --include-before-body as the LaTeX patch handles the break now
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
            "-V", "urlcolor=blue"
        ]

        try:
            # Run the conversion
            subprocess.run(cmd, check=True)
            print(f"Successfully created {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {path}: {e}")
            
    # Cleanup temporary header file
    if os.path.exists("header.tex"):
        os.remove("header.tex")

if __name__ == "__main__":
    convert_md_to_pdf()