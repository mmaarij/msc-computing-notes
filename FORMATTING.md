**MSC Computing Notes Formatter**

Fix the markdown formatting in the notes file(s) I provide. Apply ALL of the following rules without changing any content wording:

**Headings**
- Top-level headings must be `# Week N: Topic` only — one per week
- All other headings use `##`, `###`, `####` as appropriate
- Remove all numeric prefixes from headings (e.g. `## 1.`, `## 2\.`, `### 1.1`)
- Remove all "Part N:" prefixes from headings
- Remove backslash escapes from heading text (e.g. `1\.` → `1`)

**Math Formulas**
- All multiline display math blocks must be collapsed to a single line: `$$...$$` on one line (no line break after opening `$$` or before closing `$$`)
- **Simple/short formulas** that accompany a label can stay inline: e.g. `$P(\Omega) = 1$`
- **Complex formulas** (fractions with sums/integrals, square roots with content, long expressions) that are the main point of a bullet must be on their own display line, formatted as:

```markdown
- **Label:**

    $$formula$$

```

- Never leave a complex formula like `$\bar{x} = \frac{\sum x_i}{n}$` crammed onto the same line as its description — that looks clunky

**Lists**
- Add a blank line before every bullet list and numbered list (required for correct PDF rendering)

**Unicode Math Symbols**
- Never use raw Unicode math characters anywhere in the notes — they cause missing-glyph warnings in PDF export
- In regular markdown text: replace with LaTeX math notation inside `$...$`, e.g. `$\cap$`, `$\cup$`, `$\ge$`, `$\le$`, `$\ne$`, `$\to$`
- In code block comments (R, Python, etc.): replace with ASCII equivalents, e.g. `&` for ∩, `|` for ∪, `>=` for ≥, `<=` for ≤, `!=` for ≠, `->` for →
- Common offenders: ∩ (U+2229), ∪ (U+222A), ≥ (U+2265), ≤ (U+2264), ≠ (U+2260), → (U+2192), — (U+2014 em dash in code blocks)

**Scope**
- Do not rename, reword, reorder, or remove any content
- Apply changes to every week/section in the file
- After editing, confirm which sections were changed