# CLAUDE.md — Standing rules for Claude Code (Scout)

The canonical specification is `docs/SPEC.md` (Sections 2–10). These rules are Section 6 of that
spec. The git rules are Udit's standing preference and are not negotiable.

## Git

- Never run `git commit` or `git push`.
- At the end of each phase (or sub-prompt), stage only the files that phase touched, by explicit
  path: `git add <path> <path> ...`. Never `git add .`.
- Write the commit message to `.commit-msg.txt` at the repo root (gitignored), then print the exact
  command for Udit: `git commit -F .commit-msg.txt`.
- Messages use Conventional Commits, for example `feat(agent): add critic-driven replanning`.
- Never add Co-Authored-By trailers for Claude.

## Secrets

- Real keys live only in `.env`, which is gitignored from the very first commit.
- Never print, log or echo a key.
- Before staging, scan the staged diff for key patterns (`gsk_`, `AIza`, `tvly-`, `sk-`) and stop
  if any appear.

## Code

- Type hints everywhere, small functions, a docstring on every public function.
- Prompts live in `scout/prompts/*.md`, never as long inline strings.
- Settings and budgets come only from `scout/config.py`.
- Library code emits events or uses `logging`; no stray `print`.
- `ruff check .` passes before every report.

## Tests

- Every phase adds tests. Tests never touch the network; they use `FakeLLM` and `FakeSearch` from
  `tests/fakes.py`.
- `pytest -q` must be green before the PHASE REPORT.
- Real API runs belong in scripts and manual checks, not in the test suite.

## Dependencies

- Pin versions in `requirements.txt`.
- Before using a library, check its current package name and API (for example `ddgs`,
  `tavily-python`, `trafilatura`, `openai`). Never invent an API.

## Scope and honesty

- Build only the current phase. If the spec looks wrong, implement the closest faithful version and
  explain under "Deviations" in the report.
- Never fake metrics, outputs or example runs. Everything in `examples/` and `evals/` comes from
  real runs.
- End every phase with the PHASE REPORT block from `docs/SPEC.md` §8.

## Environment

- WSL Ubuntu, virtualenv at `.venv`, all commands run from the repo root (`~/Projects/techvruk`).

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
ruff check .
python scripts/smoke.py   # real API calls; needs .env
```
