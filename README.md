# DVCL Paper Repository

This repository stores the paper-writing materials for **DVCL: Dual-View Contrastive Learning for Robust Heterogeneous Graph Neural Networks**.

> Note: the related implementation code is currently maintained separately at `D:\dev\HSeCo`. This repository is only used for paper assets and manuscript management for now.

## Repository Layout

```text
.
+-- paper/
|   +-- dvcl/
|   |   +-- main.tex
|   |   +-- sections/
|   |   +-- references.bib
|   +-- dvcl_reading/
|   |   +-- main_zh.tex
|   |   +-- main_bilingual.tex
|   |   +-- sections_zh/
|   +-- _exports/              # generated, ignored
|   +-- dvcl.zip               # generated, ignored
|   +-- dvcl_reading.zip       # generated, ignored
+-- scripts/
|   +-- package_overleaf.ps1
|   +-- package_reading.ps1
+-- README.md
```

## Source Of Truth

- English submission source: `paper/dvcl/`
- English section files: `paper/dvcl/sections/`
- Shared references: `paper/dvcl/references.bib`
- Chinese reading source: `paper/dvcl_reading/sections_zh/`

The English manuscript exists in only one place. The bilingual draft reads English sections from `paper/dvcl/sections/`, so edits to the English paper automatically appear in the bilingual reading draft.

## Entry Points

- English submission draft: `paper/dvcl/main.tex`
- Chinese reading draft: `paper/dvcl_reading/main_zh.tex`
- Bilingual reading draft: `paper/dvcl_reading/main_bilingual.tex`

For the Chinese and bilingual drafts, use a LaTeX engine with CJK support, such as XeLaTeX.

## VSCode

Install the recommended VSCode extension:

```text
James-Yu.latex-workshop
```

The repository includes workspace settings under `.vscode/`.

Recommended recipes:

- English draft: `latexmk (pdfLaTeX)`
- Chinese and bilingual drafts: `latexmk (XeLaTeX)`

In VSCode, run `LaTeX Workshop: Build with recipe` and choose the matching recipe for the current entry file.

## Packaging

Generate the English Overleaf upload package:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package_overleaf.ps1
```

This creates:

```text
paper/dvcl.zip
```

Generate the Chinese/bilingual reading package:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package_reading.ps1
```

This creates:

```text
paper/dvcl_reading.zip
```

The reading package is built from a temporary export directory. It copies the shared English sections and references only at packaging time, so the repository does not maintain duplicate English manuscript content.

## Code Location

The code related to this work is not migrated into this repository yet.

Current code directory:

```text
D:\dev\HSeCo
```

If code is migrated later, update this README with the new project layout, setup instructions, and experiment reproduction steps.

## Suggested Workflow

1. Edit the English submission source under `paper/dvcl/`.
2. Edit Chinese reading notes under `paper/dvcl_reading/sections_zh/`.
3. Use `paper/dvcl_reading/main_bilingual.tex` for local bilingual review.
4. Run the packaging scripts only when you need an upload zip.
5. Avoid committing generated zip files or LaTeX build outputs.

## Git Notes

This repository is initialized as a Git repository. Use focused commits for paper updates, for example:

```text
docs(paper): revise DVCL introduction
docs: update repository README
```
