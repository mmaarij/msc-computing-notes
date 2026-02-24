import os
import subprocess
from pathlib import Path

LATEX_HEADER = r"""
\usepackage{titling}
% Center Title and Subtitle vertically on Page 1
\pretitle{\begin{center}\vspace*{\fill}\Huge\bfseries}
\posttitle{\end{center}}
\preauthor{\begin{center}\large}
\postauthor{\end{center}\vspace*{\fill}\clearpage}
"""

def convert_md_to_pdf():
    # Files to convert: filename -> subtitle shown as "author" on title page
    FILES = {
        "Compiled Notes.md": "Compiled Revision Notes",
        "Lab Notes.md":      "Lab Notes",
    }

    # 1. Create the header file
    with open("header.tex", "w", encoding="utf-8") as f:
        f.write(LATEX_HEADER)

    for filename, subtitle in FILES.items():
        for path in Path(".").rglob(filename):
            course_name = path.parent.name
            output_path = path.with_suffix(".pdf")
            temp_md_path = path.with_name("temp_compiled.md")

            print(f"Converting: {path} for Course: {course_name} ({subtitle})")

            # 2. Create a temporary MD file with a page break at the very top
            # This forces a break after the TOC renders but before content starts
            with open(path, "r", encoding="utf-8") as original:
                content = original.read()

            with open(temp_md_path, "w", encoding="utf-8") as temp_file:
                temp_file.write("\\newpage\n\n" + content)

            # 3. Construct Pandoc command using the temp file
            cmd = [
                "pandoc", str(temp_md_path),
                "-o", str(output_path),
                "--pdf-engine=xelatex",
                "--include-in-header=header.tex",
                "--variable", f"title={course_name}",
                "--variable", f"author={subtitle}",
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
            finally:
                # 4. Cleanup the temporary MD file
                if os.path.exists(temp_md_path):
                    os.remove(temp_md_path)

    # Cleanup header
    if os.path.exists("header.tex"):
        os.remove("header.tex")

if __name__ == "__main__":
    convert_md_to_pdf()