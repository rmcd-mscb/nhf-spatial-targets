# Presentations

[Marp](https://marp.app/) markdown decks for collaborator briefings and design
reviews. Marp lets us keep slides as plain markdown (diff-friendly, lives next
to the source docs it cites) and render to HTML or PDF on demand.

## Files

- `2026-collaborator-overview.slides.md` — first collaborator briefing on the
  pipeline as a whole + per-target inspection findings, used to drive
  consensus on (a) period of record per target group and (b) datasets per
  target group. References figures saved to
  `../figures/inspect_aggregated/`.

## Rendering

Marp tooling is intentionally *not* added to `pixi.toml` (the pipeline
itself doesn't depend on it). Use whichever of these fits your environment:

### VSCode (recommended — interactive)

Install the **Marp for VS Code** extension. Open any `*.slides.md` file
and click the "Open Preview" button in the editor toolbar. Export from the
extension's command palette: `Marp: Export slide deck...` → choose PDF /
HTML / PPTX.

### Command line (one-shot via npx)

Requires Node.js (no global install needed):

```bash
# Render to PDF
npx --yes @marp-team/marp-cli docs/presentations/2026-collaborator-overview.slides.md \
    --pdf --allow-local-files

# Render to HTML
npx --yes @marp-team/marp-cli docs/presentations/2026-collaborator-overview.slides.md \
    --html --allow-local-files
```

`--allow-local-files` is required because slides reference images in
`../figures/inspect_aggregated/` via relative paths.

### Docker (no Node install)

```bash
docker run --rm -v "$PWD:/home/marp/app" \
    marpteam/marp-cli docs/presentations/2026-collaborator-overview.slides.md \
    --pdf --allow-local-files
```

## Editing conventions

- Each `---` on its own line is a slide break.
- The YAML front-matter at the top of each deck controls theme, paginate,
  size, and inline custom CSS — don't strip it.
- Figures are referenced by relative path from the slide-deck file, e.g.
  `![](../figures/inspect_aggregated/aet_normalized_comparison.png)`. The
  figure pipeline is the inspection notebooks under
  `notebooks/inspect_aggregated/`; flip `_helpers.SAVE_FIGURES = True` and
  re-run the relevant cells to refresh, then commit the updated PNGs.
