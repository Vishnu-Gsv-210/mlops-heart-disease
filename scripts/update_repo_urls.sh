#!/usr/bin/env bash
# Swap the repository owner in all report/status surfaces and rebuild deliverables.
# Usage: scripts/update_repo_urls.sh <old-owner> <new-owner>   (repo name fixed: mlops-heart-disease)
set -euo pipefail
OLD="${1:?old owner required}"
NEW="${2:?new owner required}"
cd "$(dirname "$0")/.."

for f in reports/final_report.md STATUS.md reports/results_summary.json README.md; do
  [ -f "$f" ] || continue
  LC_ALL=C sed -i '' "s|github\.com/${OLD}/mlops-heart-disease|github.com/${NEW}/mlops-heart-disease|g; s|kteja-av/mlops-heart-disease|${NEW}/mlops-heart-disease|g" "$f"
done

.venv/bin/python scripts/export_report_pdf.py
.venv/bin/python scripts/generate_report.py >/dev/null
echo "URLs swapped ${OLD} -> ${NEW}; PDFs rebuilt."
