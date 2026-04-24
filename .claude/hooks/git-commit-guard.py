#!/usr/bin/env python3
"""PreToolUse guard that blocks bare `git commit` and directs Claude to use `pixi run git commit`.

The repo's pre-commit config runs ruff and pytest via `pixi run`. Invoking
`git commit` outside an activated pixi shell forces every hook to re-resolve
the pixi environment, which is slow and has intermittently stalled the
pytest hook. Running the commit via `pixi run git commit` keeps the pre-
commit invocations inside the already-resolved pixi environment.

Reads the hook payload on stdin. If the tool_input.command begins with
`git commit` at a word boundary, prints the permission-deny JSON to
stdout and exits 0. Otherwise exits silently so the tool call proceeds.

Matches (blocked):
  git commit
  git commit -m "msg"
  <leading whitespace>git commit --amend

Does NOT match (allowed):
  pixi run git commit ...          (doesn't start with `git`)
  cat <<EOF ... git commit ... EOF (doesn't start with `git`)
  git branch -D foo                (second token isn't `commit`)
  for b in ...; do git branch ...  (doesn't start with `git`)
"""

from __future__ import annotations

import json
import re
import sys

PATTERN = re.compile(r"^\s*git\s+commit(\s|$)")

REASON = (
    "This repo runs git commit through pixi so pre-commit hooks "
    "(ruff, pytest) resolve inside the pixi dev env instead of the "
    "parent shell. Re-run the same command with pixi run prepended "
    "(pixi run git commit -m ...). See CLAUDE.md -> Pre-commit Quality Gate."
)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    cmd = payload.get("tool_input", {}).get("command", "")
    if not PATTERN.match(cmd):
        return
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": REASON,
            }
        },
        sys.stdout,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
