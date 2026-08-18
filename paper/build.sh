#!/usr/bin/env bash
# Build the paper with pdflatex + bibtex (the IEEE conference toolchain).
#
# Use pdflatex, not tectonic/xelatex: ieeeconf + the `times` package resolve
# Times correctly only under pdflatex. Under XeTeX the TU/ptm font shapes are
# undefined, the engine silently substitutes, and the page count drifts --
# which matters for a page-limited venue.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
export TEXINPUTS=.:

DOC=root

# ./build.sh            -> working build, co-author notes visible
# ./build.sh submission -> notes stripped, for the PDF you actually upload
if [ "${1:-}" = "submission" ]; then
  SRC="\\def\\SUBMISSION{1}\\input{$DOC}"
  echo "mode:     SUBMISSION (co-author notes hidden)"
else
  SRC="$DOC"
  echo "mode:     working draft (co-author notes VISIBLE -- do not submit this PDF)"
fi

run() { pdflatex -interaction=nonstopmode -halt-on-error -jobname="$DOC" "$SRC" >/dev/null; }

run
bibtex "$DOC" >/dev/null || true
run
run

echo "pages:    $(pdfinfo "$DOC.pdf" | awk '/^Pages/{print $2}')"
echo "size:     $(du -h "$DOC.pdf" | cut -f1)"

echo "--- undefined citations/references ---"
grep -E "Citation .* undefined|Reference .* undefined|There were undefined" "$DOC.log" \
  | sed 's/^/  /' | sort -u || echo "  none"

echo "--- overfull boxes (>1pt) ---"
grep -c "Overfull" "$DOC.log" | sed 's/^/  count: /'
