#!/usr/bin/env bash
# Build vcc2026-metrics-brief.pdf from vcc2026-metrics-brief.md.
#
# Same pipeline as build.sh, pointed at the abridged source and its own LaTeX wrapper:
# drop the front matter (everything above the first `## `), promote `##` to `#` so
# --top-level-division=chapter maps each to a \chapter, convert the GitHub math forms back
# to plain $ math (pandoc knows neither), then typeset main_brief.tex around the result.
set -euo pipefail
cd "$(dirname "$0")"
AUX=build
mkdir -p "$AUX"

sed -n '/^## /,$p' vcc2026-metrics-brief.md | sed -E 's/^## ([0-9]+\. )?/# /' \
  | awk '/^```math$/{print "$$"; m=1; next} m && /^```$/{print "$$"; m=0; next} {print}' \
  | sed -E 's/\$`([^`]*)`\$/$\1$/g' \
  | pandoc --from=markdown --to=latex --top-level-division=chapter \
           --output="$AUX/body_brief.tex"

# As in build.sh: pandoc emits every pipe table as a longtable, which page-breaks a small
# table across a sheet. These carry no caption, so only \endhead is emitted and turning them
# back into a plain tabular in a float is safe.
#
# ⚠️ GNU sed REQUIRED, and `-i` is not the reason -- `\n` on the REPLACEMENT side is a GNU
# extension that BSD/macOS sed emits as a literal `n`, so this substitution needs GNU sed no
# matter how the in-place edit is spelled. Measured: the three-expression pipeline turns a
# 4-line longtable into 6 lines here. `build.sh:33` carries the byte-identical line, so both
# documents share the dependency; do not "portably" patch one of them (Gemini, #368).
sed -i -e 's/\\begin{longtable}\[\]{\(.*\)}/\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{\1}/' \
       -e '/^\\endhead$/d' \
       -e 's/\\end{longtable}/\\end{tabular}\n\\end{table}/' "$AUX/body_brief.tex"

for _ in 1 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$AUX" \
             -jobname=main_brief main_brief.tex >/dev/null
done
cp -f "$AUX/main_brief.pdf" vcc2026-metrics-brief.pdf
grep -n -E 'Overfull|Underfull' "$AUX/main_brief.log" || true
echo "wrote $(pwd)/vcc2026-metrics-brief.pdf"
