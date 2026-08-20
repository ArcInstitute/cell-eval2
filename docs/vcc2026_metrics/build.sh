#!/usr/bin/env bash
# Build vcc2026-metrics.pdf from vcc2026-metrics.md.
#
# vcc2026-metrics.md is the SOURCE. This script drops its front matter (everything above the
# first `## ` heading, i.e. the title, the build note and the contents list, all of which the
# LaTeX wrapper supplies itself), promotes `##` to `#` so pandoc's --top-level-division=chapter
# maps each one to a \chapter, converts to build/body.tex, and typesets main.tex around it.
#
# Three pdflatex passes: the first writes .aux, the second resolves cross-references and fills
# the table of contents, the third settles the page numbers the ToC itself shifted.
set -euo pipefail
cd "$(dirname "$0")"
AUX=build
mkdir -p "$AUX"

# The `N. ` prefixes exist so the Markdown's own contents list has anchors to link to; LaTeX
# numbers its chapters itself, so they are stripped on the way through.
# GitHub's Markdown escaper reaches the math BEFORE the math renderer does: it strips a backslash
# before ASCII punctuation (`\;` -> `;`, `\{` -> `{`, `\#` -> `#`), and subscript underscores in
# two different inline spans pair into <em> across the paragraph. Both are avoided by writing the
# math where the escaper cannot reach -- a ```math fence for display, GitHub's documented $`x`$
# form for inline. pandoc knows neither, so both are converted back to plain $ math here.
sed -n '/^## /,$p' vcc2026-metrics.md | sed -E 's/^## ([0-9]+\. )?/# /' \
  | awk '/^```math$/{print "$$"; m=1; next} m && /^```$/{print "$$"; m=0; next} {print}' \
  | sed -E 's/\$`([^`]*)`\$/$\1$/g' \
  | pandoc --from=markdown --to=latex --top-level-division=chapter \
           --output="$AUX/body.tex"

# pandoc renders every pipe table as a longtable, which page-breaks a three-row table across a
# sheet whenever it lands near the bottom. These tables are all small, so turn each one back
# into a plain tabular in a float. Safe because the tables carry no caption, so pandoc emits
# only the \endhead form -- no \endfirsthead / \endfoot to reconcile.
sed -i -e 's/\\begin{longtable}\[\]{\(.*\)}/\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{\1}/' \
       -e '/^\\endhead$/d' \
       -e 's/\\end{longtable}/\\end{tabular}\n\\end{table}/' "$AUX/body.tex"

for _ in 1 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$AUX" main.tex >/dev/null
done
cp -f "$AUX/main.pdf" vcc2026-metrics.pdf
grep -n -E 'Overfull|Underfull' "$AUX/main.log" || true
echo "wrote $(pwd)/vcc2026-metrics.pdf"
