# Kimi K3 on 512 GB200s -- blog drafts

**Published.** The article is live at
https://anijain2305.github.io/posts/kimi-k3-roofline/. These files stay the
source of truth; `kimi_k3_blog.py --blog ~/local/blog` (or `--watch --blog
~/local/blog`) regenerates the page and imports it into the Hugo repo's page
bundle. Publishing (lastmod, errata, build check, commit, push, verify) follows
`~/local/blog/AGENTS.md`. Author notes (`## Notes`) are dropped on import.

One file per section of the blog, in reading order. Write freely; the drafts
below are a starting point and will be revised as round two goes deeper.

Conventions (for when these files get built into the page):

- `{{name}}` -- a number that will come from the model or a benchmark (for now
  most numbers are written out literally, with their source in brackets).
- `::: chart <name>` / `::: table <name>` -- where an interactive chart or table
  from the current pages goes.
- `> TODO:` -- open questions or measurements still needed.
- Sections are numbered automatically at build time (don't type numbers in
  headings). Refer to another section by its file name without the order
  prefix: `[[memory]]` -> "Section 3", `[[memory.2]]` -> "Section 3.2" (a link).
- Headings are title-cased at build time (small words stay lowercase;
  acronyms and numbers are left as written).
- `::: figure <name>` -- an SVG figure from ../blog_figures.py.
- `> INSIGHT: ...` (or `> RULE: ...`; a blockquote, may span lines) -- a
  highlighted "Insight" box: a short conceptual takeaway worth internalizing for
  any roofline exercise (name still to be decided). Longer ones go in
  appendix_rules_of_thumb.md and are referenced with `[[rules_of_thumb.N]]`.
- `> CAVEAT: ...` -- a red caveat box (a limitation or deliberate choice the
  reader should keep in mind).
- `## Notes` at the end of each file -- scratch space, not published.

Files:

| File | Section |
|---|---|
| 00_intro.md | Why this study; the one-paragraph answer |
| 01_setup.md | Kimi K3 at a glance: attention, MoE, attention residuals, parameter split |
| 01b_cluster.md | The cluster and the training run |
| 02_memory.md | The memory problem: persistent state, stashed activations, remat, DistMoE, MXFP8 |
| 02b_sharding.md | Using parallelism to fit the model: which configs fit in 192 GB |
| 03_compute_roofline.md | FLOPs per step and the roofline |
| 05_communication.md | Links, collectives, overlap |
| 05b_step.md | Pipelines: what a hop carries, how 1F1B and interleaved schedules expose or hide hops, the schedule simulation |
| 06_ranking.md | The ranking, and why: FSDP vs PP, PP 8 vs PP 16, interleaving, transports, overlapped TP / CP |
| 10_open_problems.md | Next directions: the ideal roofline of DistMuon, other pipeline schedules |
| appendix_methodology.md | How each number was obtained |
