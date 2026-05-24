# Presentations

[Marp](https://marp.app/) markdown decks for collaborator briefings and design
reviews. Marp lets us keep slides as plain markdown (diff-friendly, lives next
to the source docs it cites) and render to HTML or PDF on demand.

## Files

- `2026-collaborator-overview-gfv2-spatial-targets.slides.md` — first
  collaborator briefing on the pipeline as a whole + per-target inspection
  findings against the `gfv2-spatial-targets` project, used to drive
  consensus on (a) period of record per target group and (b) datasets per
  target group. References figures saved to
  `../figures/{consolidated,aggregated}/gfv2-spatial-targets/`.

## Naming convention

Decks are project-specific (figures embedded reference one fabric's
HRU geometry). Filename pattern:
``<YYYY>-<topic>-<project_dir_name>.slides.md``. A future briefing for the
next-generation fabric would land at
``2026-collaborator-overview-gfv11-spatial-targets.slides.md`` with its
own figures under ``../figures/{consolidated,aggregated}/gfv11-spatial-targets/``.

## Rendering

Pick whichever fits your environment. The `pixi run -e marp` path is the
recommended choice for HPC operators where no system browser is installed —
chromium is bundled via the `marp` pixi feature.

### Pixi `marp` feature (recommended for HPC / fresh checkouts)

Two-step install — chromium isn't on conda-forge, so PDF rendering also needs
puppeteer's chrome-headless-shell:

```bash
# Step 1 — pulls nodejs + Linux system libs (libgbm + alsa-lib) (~100 MB)
pixi install -e marp

# Step 2 — downloads chrome-headless-shell into ~/.cache/puppeteer (~150 MB)
# Required for --pdf / --pptx / --png / --preview output; HTML + server work
# without it. Re-run safely to upgrade.
pixi run -e marp marp-setup
```

Then render any deck via the `render-deck` task (which delegates to
`scripts/render_deck.py`, a thin wrapper that resolves chrome from the
puppeteer cache and sets `MARP_USER=root` for the sandboxless HPC chrome):

```bash
# Render to PDF
pixi run -e marp render-deck docs/presentations/2026-05-aggregated-targets-overview-or-spatial-targets.slides.md --pdf

# Render to HTML (no chromium required at runtime — fast)
pixi run -e marp render-deck docs/presentations/2026-05-aggregated-targets-overview-or-spatial-targets.slides.md --html

# Live-reload server (open the printed URL — works through SSH port-forward)
pixi run -e marp render-deck docs/presentations/ --server
```

`--allow-local-files` is added automatically by the wrapper (slides reference
figures in `../figures/{consolidated,aggregated,targets}/<project>/` via
relative paths).

### VSCode (interactive, alternative)

Install the **Marp for VS Code** extension. Open any `*.slides.md` file
and click the "Open Preview" button in the editor toolbar. Export from the
extension's command palette: `Marp: Export slide deck...` → choose PDF /
HTML / PPTX. Works on workstations with Chrome installed; not viable on
headless HPC.

### Command line (one-shot via npx, no pixi)

Workstation-only — requires Node.js + a system Chrome/Edge/Firefox for PDF
output. `--html` works without a browser. The `pixi run -e marp` path
above is the portable alternative.

```bash
# PDF (needs Chrome/Edge/Firefox)
npx --yes @marp-team/marp-cli docs/presentations/2026-05-aggregated-targets-overview-or-spatial-targets.slides.md \
    --pdf --allow-local-files

# HTML (no browser needed)
npx --yes @marp-team/marp-cli docs/presentations/2026-05-aggregated-targets-overview-or-spatial-targets.slides.md \
    --html --allow-local-files
```

### Docker (no Node install)

```bash
docker run --rm -v "$PWD:/home/marp/app" \
    marpteam/marp-cli docs/presentations/2026-05-aggregated-targets-overview-or-spatial-targets.slides.md \
    --pdf --allow-local-files
```

## Editing conventions

- Each `---` on its own line is a slide break.
- The YAML front-matter at the top of each deck controls theme, paginate,
  size, and inline custom CSS — don't strip it.
- Figures are referenced by relative path from the slide-deck file, e.g.
  `![](../figures/aggregated/gfv2-spatial-targets/aet_normalized_comparison.png)`.
  The figure pipeline is the inspection notebooks under
  `notebooks/{consolidated,aggregated}/`; flip
  `_helpers.SAVE_FIGURES = True` and `_helpers.PROJECT = PROJECT_DIR.name`
  near the top of each notebook, re-run the relevant cells (or
  `pixi run -e dev render-figures` / `sbatch inspect_*.slurm`), then
  commit the updated PNGs.
