"""Export reports/final_report.md to reports/final_report.pdf.

Pipeline: python-markdown -> styled standalone HTML (written next to the
report so relative image paths resolve) -> headless Chrome print-to-pdf.

Run from repository root: python scripts/export_report_pdf.py
"""
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "reports" / "final_report.md"
HTML_OUT = ROOT / "reports" / "final_report.html"
PDF_OUT = ROOT / "reports" / "final_report.pdf"
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]

CSS = """
@page { size: A4; margin: 14mm 13mm 15mm 13mm; }
* { box-sizing: border-box; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
       font-size: 9.6pt; line-height: 1.36; color: #1f2937; margin: 0; }
h1 { font-size: 16.5pt; color: #17365d; border-bottom: 2.5px solid #17365d;
     padding-bottom: 5px; margin: 12px 0 8px 0; page-break-after: avoid; }
h2 { font-size: 12.5pt; color: #17365d; margin: 12px 0 6px 0;
     border-bottom: 1px solid #cbd5e1; padding-bottom: 3px; }
h3 { font-size: 10.8pt; color: #334155; margin: 10px 0 5px 0; }
p, li { margin: 4px 0; }
img { max-width: 86%; height: auto; display: block;
      margin: 6px auto; border: 1px solid #e2e8f0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0;
        font-size: 8.2pt; page-break-inside: avoid; }
th { background: #17365d; color: #ffffff; text-align: left;
     padding: 4px 6px; }
td { border: 0.5px solid #cbd5e1; padding: 3px 6px;
     vertical-align: top; }
tr:nth-child(even) td { background: #f8fafc; }
pre { background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px;
      padding: 6px 9px; font-size: 7.8pt; line-height: 1.3;
      overflow-x: hidden; white-space: pre-wrap; word-wrap: break-word; }
code { font-family: Menlo, monospace; font-size: 8.2pt; }
p code, li code, td code { background: #f1f5f9; padding: 0 3px;
                           border-radius: 3px; }
blockquote { border-left: 4px solid #94a3b8; margin: 6px 0;
             padding: 2px 10px; color: #475569; background: #f8fafc; }
hr { border: 0; border-top: 1px solid #cbd5e1; margin: 10px 0; }
a { color: #1d4ed8; text-decoration: none; }
.cover { text-align: center; padding: 46px 0 20px 0; }
.cover .t1 { font-size: 22pt; font-weight: 700; color: #17365d; }
.cover .t2 { font-size: 12pt; color: #334155; margin-top: 7px; }
.cover .t3 { font-size: 10pt; color: #64748b; margin-top: 12px; line-height: 1.55; }
.cover .box { margin: 20px auto 0 auto; width: 78%; border: 1px solid #cbd5e1;
              border-radius: 6px; padding: 10px 14px; text-align: left;
              font-size: 9pt; background: #f8fafc; }
.disclaimer { background: #fff7e6; border: 1px solid #f59e0b;
              border-radius: 4px; padding: 6px 11px; font-size: 8.6pt;
              margin: 10px 0; }
"""


def build_html() -> str:
    text = REPORT_MD.read_text(encoding="utf-8")
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "smarty"]
    )
    cover = """
<div class="cover">
  <div class="t1">End-to-End MLOps:<br/>Heart Disease Classification Pipeline</div>
  <div class="t2">Data &ndash; EDA &ndash; Models &ndash; MLflow &ndash; CI/CD &ndash; Docker &ndash; Kubernetes &ndash; Monitoring</div>
  <div class="t3">
    Vishnu Gopisetti &nbsp;|&nbsp; Student ID 2025AE05666<br/>
    Machine Learning Operations (MLOps) &mdash; AIMLCZG523 &nbsp;|&nbsp; Assignment 01 &nbsp;|&nbsp; Total marks: 50<br/>
    Dataset: UCI Heart Disease (processed Cleveland), DOI 10.24432/C52P4X &nbsp;|&nbsp; 2026-10-20
  </div>
  <div class="box">
    <b>Contents.</b> Executive summary &middot; Dataset &amp; EDA &middot; Preprocessing &amp; feature
    engineering &middot; Model development &amp; results &middot; Experiment tracking &middot;
    Reproducibility &middot; Testing &amp; CI/CD &middot; API &amp; containerization &middot; Deployment
    &middot; Monitoring &amp; logging &middot; Architecture &middot; Limitations &middot; How to run &middot; References.
    <br/><br/>
    <b>Evidence.</b> Every metric in this report is generated from real local execution
    (<code>reports/artifacts/*.json</code>, <code>run_logs/</code>, <code>screenshots/</code>);
    items not yet executed are marked as pending rather than claimed.
  </div>
</div>
<div class="disclaimer"><b>Health disclaimer:</b> educational project only &mdash; not for
diagnosis, treatment, triage, or any clinical decision.</div>
"""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>MLOps Assignment 01 - Vishnu Gopisetti 2025AE05666</title>
<style>{CSS}</style></head>
<body>{cover}{body}</body></html>"""


def chrome_bin() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    raise SystemExit("No Chrome/Chromium binary found for print-to-pdf")


def main() -> None:
    HTML_OUT.write_text(build_html(), encoding="utf-8")
    cmd = [
        chrome_bin(), "--headless=new", "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUT}",
        HTML_OUT.as_uri(),
    ]
    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        raise SystemExit(f"Chrome print-to-pdf failed ({proc.returncode})")
    size = PDF_OUT.stat().st_size
    if size < 20_000:
        raise SystemExit(f"Suspiciously small PDF ({size} bytes); inspect {HTML_OUT}")
    print(f"Exported {PDF_OUT} ({size:,} bytes) from {REPORT_MD.name}")


if __name__ == "__main__":
    main()
