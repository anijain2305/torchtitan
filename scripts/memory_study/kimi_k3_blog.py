# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build kimi_k3_blog.html from the Markdown drafts in blog/.

Sections are the blog/*.md files in name order (README.md excluded). Draft
conventions: `::: figure <name>` inserts an SVG from blog_figures.py;
`::: chart <name>` / `::: table <name>` lines become placeholder
boxes for the interactive charts and tables to come; `> TODO:` blockquotes are
shown as callouts; each file's `## Notes` section is shown muted as author
notes (not part of the published text).

Sections are numbered at build time (the first file is the title, appendix_*
files get letters), so files can be reordered freely. Refer to a section by its
file name without the order prefix: [[memory]] -> "Section 3", [[memory.2]] ->
"Section 3.2", linked.

Usage:
  python scripts/memory_study/kimi_k3_blog.py           # build once
  python scripts/memory_study/kimi_k3_blog.py --watch   # rebuild on save
"""

import argparse
import html
import importlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import blog_figures

import markdown
from page_common import PAGE_CSS

HERE = Path(__file__).resolve().parent
BLOG = HERE / "blog"
OUT = HERE / "kimi_k3_blog.html"

EXTRA_CSS = """
.blog { max-width: 960px; margin: 0 auto; font-size: 16px; line-height: 1.65; }
.figure svg { max-width: 860px; display: block; }
.blog h1 { font-size: 30px; margin: 40px 0 12px; }
.blog h2 { font-size: 22px; margin: 32px 0 8px; }
.blog h3 { font-size: 18px; margin: 24px 0 6px; }
.blog p, .blog li { color: var(--text-primary); }
.blog a { color: #2a78d6; }
.blog table { border-collapse: collapse; margin: 12px 0; font-size: 14px; }
.blog th, .blog td { border-bottom: 1px solid var(--grid); padding: 6px 10px; text-align: left; }
.blog blockquote { background: var(--note-bg); border-left: 3px solid #eda100; margin: 12px 0;
                   padding: 8px 14px; border-radius: 6px; color: var(--text-secondary); }
.blog code { background: var(--wash); padding: 1px 4px; border-radius: 4px; }
.figure { margin: 18px 0 8px; }
.ctx-toggle { position: fixed; top: 14px; right: 18px; z-index: 5; background: var(--surface-1);
              border: 1px solid var(--border); border-radius: 10px; padding: 6px 10px; font-size: 13px;
              color: var(--text-secondary); box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06); }
.ctx-toggle button { font: inherit; border: 1px solid var(--border); background: var(--surface-1);
                     border-radius: 6px; padding: 2px 10px; margin-left: 4px; cursor: pointer; }
.ctx-toggle button.on { background: #2a78d6; color: #fff; border-color: #2a78d6; }
.rule { background: #eef4fc; border-left: 4px solid #2a78d6; border-radius: 6px; padding: 10px 16px;
        margin: 16px 0; }
.rule p { margin: 4px 0; }
.caveat { background: rgba(227, 73, 72, 0.07); border-left: 4px solid #e34948; border-radius: 6px;
          padding: 10px 16px; margin: 16px 0; font-size: 15px; }
.caveat p { margin: 4px 0; }
.caveat-label { color: #c0302f; font-size: 12px; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.06em; }
.rule-label { color: #2a78d6; font-size: 12px; font-weight: 700; text-transform: uppercase;
              letter-spacing: 0.06em; }
.dtable { font-size: 13px; width: 100%; }
.formulas { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
            padding: 12px 18px; margin: 14px 0; }
.f-row { display: grid; grid-template-columns: 170px 18px auto; column-gap: 6px; align-items: baseline;
         padding: 6px 0; }
.f-row + .f-row { border-top: 1px solid var(--grid); }
.f-lhs { text-align: right; font-weight: 600; white-space: nowrap; }
.f-def .f-lhs { font-weight: 400; }
.f-eq { text-align: center; color: var(--text-secondary); }
.f-rhs { font-family: "Cambria Math", "STIX Two Math", Georgia, serif; font-size: 17px; }
.f-lhs i, .f-lhs sub, .f-rhs i, .dtable th i, .dtable th sub { font-family: "Cambria Math", "STIX Two Math", Georgia, serif; }
.f-def .f-lhs { font-size: 17px; }
.f-note { grid-column: 3; color: var(--muted); font-size: 13px; }
.dtable caption { caption-side: bottom; text-align: left; color: var(--text-secondary); font-size: 13px;
                  padding-top: 6px; }
.rk-chart { font-size: 12px; width: min(1180px, calc(100vw - 48px)); position: relative; left: 50%;
            transform: translateX(-50%); }
.rk-filters { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 4px 0 8px; }
.rk-filters > span { color: var(--muted); margin: 0 2px 0 12px; }
.rk-filters > span:first-child { margin-left: 0; }
.rk-chip { font: inherit; border: 1px solid var(--border); background: var(--surface-1); border-radius: 12px;
           padding: 1px 10px; cursor: pointer; color: var(--muted); text-decoration: line-through; }
.rk-chip.on { color: var(--text-primary); border-color: var(--axis); text-decoration: none; }
.rk-sort { font: inherit; border: 1px solid var(--border); background: var(--surface-1); border-radius: 12px;
           padding: 1px 10px; cursor: pointer; color: var(--text-secondary); }
.rk-sort.on { background: #2a78d6; color: #fff; border-color: #2a78d6; }

.rk-legend { display: flex; flex-wrap: wrap; gap: 8px 10px; margin: 6px 0 10px; color: var(--text-secondary); }
.rk-leg-group { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: center; border: 1.5px solid;
                border-radius: 6px; padding: 4px 10px; }
.rk-leg-item { display: inline-flex; gap: 5px; align-items: center; }
.rk-chart .sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px; flex: none; }
.rk-row { display: grid; grid-template-columns: 24px 330px 1fr 56px 130px; gap: 8px; align-items: center;
          padding: 1px 4px; border-radius: 4px; }
.rk-row[hidden] { display: none; }
.rk-row:not(.rk-head):hover, .rk-row:focus-visible { background: var(--wash); outline: none; }
.rk-row .r { text-align: right; white-space: nowrap; }
.rk-n { color: var(--muted); text-align: right; }
.rk-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rk-head { color: var(--muted); font-size: 11px; position: sticky; top: 0; background: var(--page); z-index: 2;
           padding-top: 4px; padding-bottom: 4px; }
.rk-axis { position: relative; height: 14px; }
.rk-axis span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
.rk-axis span:first-child { transform: none; }
.rk-bar { position: relative; display: block; height: 18px;
          background: linear-gradient(to right, var(--grid) 1px, transparent 1px) 0 0 / calc(100% / 6) 100%; }
.rk-grp { position: absolute; top: 1px; bottom: 1px; overflow: hidden; border-radius: 3px;
          border-left: 2px solid var(--page); box-sizing: border-box; }
.rk-grp:first-child { border-left: 0; }
.rk-grp::after { content: ""; position: absolute; inset: 0; border-radius: 3px; box-shadow: inset 0 0 0 1.5px var(--ring); }
.rk-inner { display: flex; height: 100%; }
.rk-inner > span { height: 100%; box-shadow: inset -1px 0 0 rgba(252, 252, 251, 0.9); }
.rk-clipped .rk-grp:last-child { border-top-right-radius: 0; border-bottom-right-radius: 0; }
.rk-mem { display: flex; gap: 8px; align-items: center; color: var(--text-secondary); white-space: nowrap; }
.rk-gauge { width: 60px; height: 6px; background: var(--grid); border-radius: 3px; overflow: hidden; flex: none; }
.rk-gauge span { display: block; height: 100%; background: var(--muted); }
.rk-table { margin-top: 10px; color: var(--text-secondary); }
.rk-table summary { cursor: pointer; }
#rk-tip { position: fixed; pointer-events: none; background: #fcfcfb; border: 1px solid rgba(11, 11, 11, 0.16);
          border-radius: 8px; padding: 8px 12px; font: 12px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
          max-width: 420px; z-index: 10; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.16); color: #0b0b0b; }
#rk-tip i { font-style: normal; font-weight: 600; display: inline-block; margin-top: 6px; }
#rk-tip .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px;
              vertical-align: -1px; }
.dtable td:not(:first-child), .dtable th:not(:first-child) { text-align: center; }
.dtable .sub { display: block; color: var(--muted); font-size: 11px; font-weight: 400; }
.dtable td { white-space: nowrap; }
.dtable th { white-space: normal; vertical-align: bottom; }
.dtable td, .dtable th { padding: 6px 8px; }
.placeholder { border: 1px dashed var(--axis); border-radius: 10px; padding: 18px; margin: 14px 0;
               color: var(--muted); text-align: center; font-size: 14px; }
.notes { border-top: 1px dashed var(--grid); margin-top: 18px; padding-top: 6px; color: var(--muted);
         font-size: 14px; }
.notes-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
.toc ol { list-style: none; padding-left: 0; margin: 6px 0; }
.toc li { margin: 3px 0; }
.toc a { color: var(--text-primary); text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.section-sep { border: 0; border-top: 1px solid var(--grid); margin: 40px 0; }
.foot { color: var(--muted); font-size: 12px; margin: 40px 0; }
"""

PLACEHOLDER = re.compile(r"^:::\s*(chart|table)\s+(\S+)\s*$", re.M)
FIGURE = re.compile(r"^:::\s*figure\s+(\S+)\s*$", re.M)


SECTION_KEY = re.compile(r"^(?:\d+[a-z]?_|appendix_)")
REF = re.compile(r"\[\[([a-z0-9_]+)(?:\.(\d+(?:\.\d+)?))?\]\]")


def section_numbers(files: list[Path]) -> dict[str, str]:
    """File key (stem without its order prefix) -> section label: the first
    file is the title (unnumbered), appendix_* files get letters, the rest 1, 2, ..."""
    labels, n, a = {}, 0, 0
    for i, p in enumerate(files):
        key = SECTION_KEY.sub("", p.stem)
        if i == 0:
            labels[key] = ""
        elif p.stem.startswith("appendix_"):
            labels[key] = f"Appendix {chr(ord('A') + a)}"
            a += 1
        else:
            n += 1
            labels[key] = str(n)
    return labels


SMALL_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "nor",
    "for",
    "so",
    "yet",
    "as",
    "at",
    "by",
    "in",
    "of",
    "on",
    "to",
    "up",
    "via",
    "vs",
    "per",
    "from",
    "into",
    "with",
    "over",
    "than",
}


def title_case(text: str) -> str:
    """Capitalize each word except short prepositions, articles and
    conjunctions (unless first or last); words already containing a capital
    or digit (KDA, GB200, 192) are left as written."""
    words = text.split(" ")
    out = []
    for i, w in enumerate(words):
        core = w.strip("()[],:;.?!\"'")
        if not core or any(c.isupper() or c.isdigit() for c in core):
            out.append(w)
        elif core.lower() in SMALL_WORDS and 0 < i < len(words) - 1:
            out.append(w.lower())
        else:
            j = w.index(core[0])
            out.append(w[:j] + w[j].upper() + w[j + 1 :])
    return " ".join(out)


def number_headings(body: str, key: str, label: str) -> tuple[str, str]:
    """Prefix the file's '#' heading with its number, its '##' headings with
    number.sub and its '###' headings with number.sub.subsub; returns the body
    and the section title (for the contents)."""
    title, sub, subsub, out = "", 0, 0, []
    for line in body.split("\n"):
        if line.startswith("# ") and not title:
            title = title_case(line[2:].strip())
            head = (
                f"{label}. {title}"
                if label and not label.startswith("Appendix")
                else (f"{label}: {title}" if label else title)
            )
            line = f'<h1 id="sec-{key}">{html.escape(head)}</h1>'
        elif line.startswith("## ") and label:
            sub += 1
            num = (
                f"{label}.{sub}"
                if not label.startswith("Appendix")
                else f"{label[-1]}.{sub}"
            )
            line = f'<h2 id="sec-{key}-{sub}">{html.escape(num + " " + title_case(line[3:].strip()))}</h2>'
            subsub = 0
        elif line.startswith("### ") and label and not label.startswith("Appendix") and sub:
            subsub += 1
            num = f"{label}.{sub}.{subsub}"
            line = f'<h3 id="sec-{key}-{sub}-{subsub}">{html.escape(num + " " + title_case(line[4:].strip()))}</h3>'
        out.append(line)
    return "\n".join(out), title


def resolve_refs(body: str, labels: dict[str, str]) -> str:
    """[[memory]] -> a link 'Section 3'; [[memory.2]] -> 'Section 3.2';
    [[memory.2.1]] -> 'Section 3.2.1'."""

    def ref(m):
        key, sub = m.group(1), m.group(2)
        if key not in labels:
            return f"**[unknown section: {key}]**"
        lab = labels[key]
        word = lab if lab.startswith("Appendix") else f"Section {lab}"
        if sub:
            word += f".{sub}" if not lab.startswith("Appendix") else f" {sub}"
        return f"[{word}](#sec-{key}{'-' + sub.replace('.', '-') if sub else ''})"

    return REF.sub(ref, body)


RULE_LABEL = "Insight"


def rule_boxes(body: str) -> str:
    """A blockquote starting with '> RULE:' becomes a highlighted rule-of-thumb box."""
    lines, out, i = body.split("\n"), [], 0
    while i < len(lines):
        if lines[i].startswith(("> RULE:", "> INSIGHT:")):
            block = [lines[i].split(":", 1)[1].strip()]
            i += 1
            while i < len(lines) and lines[i].startswith(">"):
                block.append(lines[i][1:].strip())
                i += 1
            inner = markdown.markdown(" ".join(block))
            out.append(
                f'\n<div class="rule"><div class="rule-label">{RULE_LABEL}</div>{inner}</div>\n'
            )
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def render_section(text: str, key: str, labels: dict[str, str]) -> tuple[str, str]:
    body, _, notes = text.partition("\n## Notes")
    body = re.sub(r"<!--.*?-->\s*", "", body, flags=re.S)  # authoring comments stay out of the page
    body, title = number_headings(body, key, labels[key])
    body = resolve_refs(body, labels)
    body = rule_boxes(body)
    # figures are raw HTML blocks, so markdown leaves them alone
    body = FIGURE.sub(
        lambda m: f'\n<div class="figure">{blog_figures.FIGURES[m.group(1)]()}</div>\n'
        if m.group(1) in blog_figures.FIGURES
        else f'<div class="placeholder">figure: {m.group(1)} (unknown)</div>',
        body,
    )
    body = PLACEHOLDER.sub(
        lambda m: f'<div class="placeholder">{m.group(1)}: {html.escape(m.group(2))} (to come)</div>',
        body,
    )
    out = markdown.markdown(body, extensions=["tables", "fenced_code", "sane_lists"])
    notes = notes.strip()
    if notes:
        out += (
            '<div class="notes"><div class="notes-label">Author notes (not published)</div>'
            + markdown.markdown(
                resolve_refs(notes, labels), extensions=["tables", "sane_lists"]
            )
            + "</div>"
        )
    return out, title


def build() -> None:
    importlib.reload(blog_figures)  # pick up figure edits in --watch mode
    files = sorted(p for p in BLOG.glob("*.md") if p.name != "README.md")
    labels = section_numbers(files)
    rendered = [
        (
            SECTION_KEY.sub("", p.stem),
            *render_section(p.read_text(), SECTION_KEY.sub("", p.stem), labels),
        )
        for p in files
    ]
    toc = "".join(
        f'<li><a href="#sec-{k}">{html.escape(labels[k] + (": " if labels[k].startswith("Appendix") else ". ") + t)}</a></li>'
        for k, _, t in rendered[1:]
    )
    parts = [
        rendered[0][1],
        f'<nav class="toc"><div class="notes-label">Contents</div><ol>{toc}</ol></nav>',
    ]
    parts += [r[1] for r in rendered[1:]]
    sections = '\n<hr class="section-sep">\n'.join(parts)
    OUT.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Roofline Analysis for Kimi K3</title>
<style>
{PAGE_CSS}
{EXTRA_CSS}
</style>
</head>
<body>
<div class="viz-root"><div class="wrap"><div class="blog">
<div class="ctx-toggle">Charts at <button data-ctx="8K" class="on">8K</button><button data-ctx="32K">32K</button></div>
{sections}
<p class="foot">Draft built {time.strftime("%Y-%m-%d %H:%M")} by scripts/memory_study/kimi_k3_blog.py from blog/*.md.</p>
</div></div></div>
<script>
// context toggle: show each context-dependent chart at the selected context
for (const btn of document.querySelectorAll(".ctx-toggle button"))
  btn.addEventListener("click", () => {{
    for (const b of document.querySelectorAll(".ctx-toggle button")) b.classList.toggle("on", b === btn);
    for (const f of document.querySelectorAll(".ctx-fig")) f.hidden = f.dataset.ctx !== btn.dataset.ctx;
  }});
// ranking chart: hover tooltip and filters (FSDP rows follow the family filter only)
const tip = document.createElement("div");
tip.id = "rk-tip"; tip.hidden = true; document.body.append(tip);
for (const row of document.querySelectorAll(".rk-row[data-tip]")) {{
  const show = e => {{
    tip.innerHTML = row.dataset.tip; tip.hidden = false;
    const r = row.getBoundingClientRect(), x = e && e.clientX ? e.clientX : r.left + r.width / 2;
    const left = Math.min(x + 16, window.innerWidth - tip.offsetWidth - 8);
    const top = r.bottom + 6 + tip.offsetHeight > window.innerHeight ? r.top - tip.offsetHeight - 6 : r.bottom + 6;
    tip.style.left = left + "px"; tip.style.top = top + "px";
  }};
  row.addEventListener("mousemove", show); row.addEventListener("focus", () => show());
  row.addEventListener("mouseleave", () => tip.hidden = true); row.addEventListener("blur", () => tip.hidden = true);
}}
// sort ascending, the biggest last; both contexts' charts follow the same choice
for (const btn of document.querySelectorAll(".rk-sort"))
  btn.addEventListener("click", () => {{
    const key = btn.dataset.sort;
    for (const b of document.querySelectorAll(".rk-sort")) b.classList.toggle("on", b.dataset.sort === key);
    for (const box of document.querySelectorAll(".rk-rows")) {{
      const rows = [...box.children];
      rows.sort((a, b) => (parseFloat(a.dataset[key]) - parseFloat(b.dataset[key]))
                          || parseFloat(a.dataset.step) - parseFloat(b.dataset.step));
      for (const r of rows) box.append(r);
    }}
  }});
const rkOn = (key, val) => document.querySelector(`.rk-chip[data-key="${{key}}"][data-val="${{val}}"]`).classList.contains("on");
for (const chip of document.querySelectorAll(".rk-chip"))
  chip.addEventListener("click", () => {{
    const on = !chip.classList.contains("on");
    for (const c of document.querySelectorAll(`.rk-chip[data-key="${{chip.dataset.key}}"][data-val="${{chip.dataset.val}}"]`)) {{
      c.classList.toggle("on", on); c.setAttribute("aria-pressed", on);
    }}
    for (const row of document.querySelectorAll(".rk-row[data-fam]"))
      row.hidden = !rkOn("fam", row.dataset.fam) ||
        (row.dataset.fam !== "1" && !rkOn("v", row.dataset.v));
  }});
</script>
</body>
</html>
"""
    )
    print(f"Wrote {OUT}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--watch", action="store_true", help="rebuild whenever a blog/*.md file changes"
    )
    parser.add_argument(
        "--blog",
        type=Path,
        default=None,
        help="after each build, import the page into this Hugo blog repo's page "
        "bundle (e.g. ~/local/blog), so its local preview stays current; "
        "publishing (commit, push) stays a separate step, see the repo's AGENTS.md",
    )
    args = parser.parse_args()

    def build_and_import() -> None:
        build()
        if args.blog:
            blog = args.blog.expanduser()
            subprocess.run(
                [sys.executable, str(blog / "scripts" / "import_article.py"), str(OUT),
                 str(blog / "content" / "posts" / "kimi-k3-roofline")],
                check=True,
            )

    build_and_import()
    if args.watch:
        seen = {}
        me = Path(__file__).resolve()
        # the builder and the modules the figures import: restart on any change,
        # so no stale module survives
        code = [me, HERE / "blog_figures.py", HERE / "kimi_k3_step_time.py"]
        started = [p.stat().st_mtime for p in code]
        while True:
            if [p.stat().st_mtime for p in code] != started:
                os.execv(sys.executable, [sys.executable, str(me), *sys.argv[1:]])
            now = {p: p.stat().st_mtime for p in BLOG.glob("*.md")}
            if seen and now != seen:
                build_and_import()
            seen = now
            time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
