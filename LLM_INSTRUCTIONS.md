# Instructions for LLMs/agents working in this repo

## Do not directly edit `notebooks/final_experiment.ipynb` (or other actively-run notebooks)

This notebook is round-tripped through Google Colab: the user runs cells there and the
`.ipynb` file gets re-synced back to this repo with full cell outputs embedded
(matplotlib PNGs as base64, print output, etc). That makes the file:

- **Large** — outputs alone can push it past 400KB, well over what a single tool read
  can return without truncation.
- **Constantly changing on disk** — every time the user re-runs cells in Colab and it
  syncs back, the file's bytes change underneath whatever the agent last read, which
  trips "file modified since read" guards on structured editors.
- **Expensive to fix when those guards trip** — the retry loop (re-read → fails on
  size → strip outputs via a script → re-read → edit → still fails because the user
  synced again) burns tokens for no benefit, since none of that back-and-forth touches
  the actual code being added.

**Do this instead:** when asked to add or modify a notebook cell, write the cell's
source code to a markdown file in the repo root — `claude_result.md` or
`opencode_result.md` — as a fenced ` ```python ` code block, with:

1. **EXACT WHICH CELL** — state the cell number clearly (e.g. "**Cell 9b**", "**Cell
   9c**", "**New cell between Cell 1 and Cell 3**") at the start of the section.
2. **Replace or Insert** — clarify whether it replaces an existing cell or is a new
   cell inserted at a specific position.
3. **Brief what changed** — 1-2 bullet points so the user knows what to expect.

Example format:
```
## Cell 9b (REPLACE) — Training loop with per-epoch logging + tqdm

Perubahan:
   - Print tiap epoch (instead of tiap 20 epoch)
   - Tambah tqdm progress bar di batch loop

[code block]
```

Let the human paste it into Colab themselves. Do not attempt to patch the `.ipynb`
JSON directly for this file unless the user explicitly asks you to edit the notebook
file itself despite the cost.

If you only need to *read* something from the notebook (not edit it), prefer targeted
inspection over a full read — e.g. via Bash:

```bash
python3 -c "
import json
nb = json.load(open('notebooks/final_experiment.ipynb'))
for c in nb['cells']:
    print(c.get('id'), '|', ''.join(c['source']).split(chr(10))[0][:80])
"
```

or `grep`/`jq` for a specific cell's source, instead of loading the whole file.

## Scope

This guidance applies to `notebooks/final_experiment.ipynb` specifically (and any
other notebook the user is actively running/syncing from Colab during a session).
Static, rarely-touched notebooks in this repo don't need this treatment — normal
notebook editing tools are fine for those.
