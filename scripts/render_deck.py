"""Wrapper around marp-cli for rendering Marp `.slides.md` decks.

Solves the "HPC has no system browser" problem by resolving chromium from the
pixi `marp` feature (`$CONDA_PREFIX/bin/chromium`) and setting `MARP_USER=root`
so marp-cli accepts the sandboxless chromium that caldera-style environments
require (no `setuid` sandbox, no user namespace). With those two env vars set,
`npx --yes @marp-team/marp-cli` rendering to PDF/PPTX/PNG works the same way
on HPC as it does on a workstation with Chrome installed.

The script appends ``--allow-local-files`` automatically (Marp decks reference
figures via relative paths and refuse to read them without this flag) and
otherwise passes through every argument to marp-cli verbatim.

Usage::

    # PDF render of one deck
    pixi run -e marp render-deck docs/presentations/<file>.slides.md --pdf

    # HTML render (no browser needed; chromium is resolved but unused)
    pixi run -e marp render-deck docs/presentations/<file>.slides.md --html

    # Live-reload server (open the printed URL in any browser; works through
    # an SSH port-forward on HPC)
    pixi run -e marp render-deck docs/presentations/ --server
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_chromium() -> str:
    """Locate a chromium/chrome binary, preferring the puppeteer-installed one.

    Discovery order:
    1. ``CHROME_PATH`` env var (if the user explicitly set one).
    2. Puppeteer's download cache (``~/.cache/puppeteer/chrome/<rev>/chrome-linux64/chrome``,
       populated by ``pixi run -e marp marp-setup`` / ``npx puppeteer browsers install chrome``).
       Honors ``PUPPETEER_CACHE_DIR`` override if set.
    3. ``$CONDA_PREFIX/bin/chrome*`` — for any future world where chrome ends up
       on conda-forge in the pixi env directly.
    4. System PATH — workstation Chrome / Edge / Firefox installs (macOS, Windows,
       desktop Linux).

    Raises ``SystemExit`` with the install hint if none found.
    """
    # 1. Explicit env override
    explicit = os.environ.get("CHROME_PATH", "").strip()
    if explicit and Path(explicit).exists():
        return explicit

    # 2. Puppeteer cache — where `npx puppeteer browsers install chrome` writes.
    # Default cache dir is ~/.cache/puppeteer on Linux/macOS, %LOCALAPPDATA%\puppeteer
    # on Windows. PUPPETEER_CACHE_DIR overrides.
    cache_root = os.environ.get("PUPPETEER_CACHE_DIR", "").strip()
    if not cache_root:
        if sys.platform == "win32":
            cache_root = os.path.join(
                os.environ.get("LOCALAPPDATA", str(Path.home())), "puppeteer"
            )
        else:
            cache_root = str(Path.home() / ".cache" / "puppeteer")
    # Search both chrome-headless-shell (the smaller variant marp-setup pulls
    # by default — works on bare HPC) and chrome (full GUI variant — operator
    # may have installed it manually for a workstation render path). Prefer
    # headless-shell since marp-cli is happy with either.
    for browser_dir, layout_candidates in (
        (
            "chrome-headless-shell",
            (
                Path("chrome-headless-shell-linux64") / "chrome-headless-shell",
                Path("chrome-headless-shell-mac-arm64") / "chrome-headless-shell",
                Path("chrome-headless-shell-mac-x64") / "chrome-headless-shell",
                Path("chrome-headless-shell-win64") / "chrome-headless-shell.exe",
            ),
        ),
        (
            "chrome",
            (
                Path("chrome-linux64") / "chrome",
                Path("chrome-mac-arm64") / "Google Chrome for Testing.app"
                / "Contents" / "MacOS" / "Google Chrome for Testing",
                Path("chrome-mac-x64") / "Google Chrome for Testing.app"
                / "Contents" / "MacOS" / "Google Chrome for Testing",
                Path("chrome-win64") / "chrome.exe",
            ),
        ),
    ):
        browser_root = Path(cache_root) / browser_dir
        if not browser_root.is_dir():
            continue
        # Each install lands under a revision-tagged subdir; pick the newest
        # so a later `marp-setup` re-run automatically supersedes the old one.
        revisions = sorted(
            (p for p in browser_root.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for rev in revisions:
            for relative in layout_candidates:
                candidate = rev / relative
                if candidate.exists():
                    return str(candidate)

    # 3. Pixi env (forward compatibility — conda-forge has no chrome today)
    env_prefix = os.environ.get("CONDA_PREFIX", "")
    if env_prefix:
        for name in ("chromium", "chrome", "google-chrome"):
            candidate = Path(env_prefix) / "bin" / name
            if candidate.exists():
                return str(candidate)

    # 4. System PATH (workstation Chrome / Edge / Firefox)
    for name in ("chromium", "chromium-browser", "chrome", "google-chrome", "msedge"):
        path = shutil.which(name)
        if path:
            return path

    raise SystemExit(
        "ERROR: No chrome/chromium binary found.\n\n"
        "On HPC (or any fresh pixi checkout): run the two-step install:\n"
        "  pixi install -e marp\n"
        "  pixi run -e marp marp-setup\n\n"
        "The second step downloads chrome (~150 MB) via puppeteer into\n"
        "~/.cache/puppeteer/. Or set CHROME_PATH=/path/to/chrome explicitly."
    )


_BROWSER_FLAGS = ("--pdf", "--pdf-notes", "--pptx", "--png", "--jpeg", "--preview")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__, file=sys.stderr)
        return 2

    # Marp refuses to embed local images without this flag, and every deck
    # under docs/presentations/ references figures via relative paths. Adding
    # it unconditionally is harmless for HTML/server output (no images would
    # break either way).
    if "--allow-local-files" not in args:
        args.append("--allow-local-files")

    env = dict(os.environ)
    # Only browser-backed outputs (PDF/PPTX/PNG/JPEG/preview) need chromium.
    # HTML and the live-reload server work without it; skip the resolve so
    # the script is useful even on a fresh pixi install before chromium has
    # been pulled.
    needs_browser = any(arg in _BROWSER_FLAGS for arg in args)
    if needs_browser:
        env["CHROME_PATH"] = _find_chromium()
        # MARP_USER=root tells marp-cli "yes, I know this chromium has no
        # setuid sandbox; render anyway." Required on HPC where user
        # namespaces and chrome's normal sandbox can't run. The variable
        # name is misleading (we're not root) but is what marp-cli checks.
        env.setdefault("MARP_USER", "root")

    cmd = ["npx", "--yes", "@marp-team/marp-cli", *args]
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
