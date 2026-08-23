# Posting kit — dsh-turbo 0.1.0

Everything needed to announce dsh-turbo. Written so the body can be copied,
the named screenshots attached, and posted.

**Status of the facts in here** — verified on 2026-08-23:

- The install path (`dsh plugin add dsh-turbo`, model selector, `DEEPSEEK_API_KEY`)
  was run on a clean DSH home with a stock `@deepseek-ai/dsh@0.1.1-rc.2` from npm
  (`scripts/verify-mount-e2e.mjs`).
- Every fan-out number (steps, candidates, degraded count, verifier calls, cached
  tokens) is read from `$DSH_HOME/dsh-turbo/decisions.jsonl` of the two live runs,
  not estimated. The file is reproduced in the example.
- Both games are the model's unedited output from those runs.
- The Terminal-Bench 79% → 88% figure is the *paper's* result on the paper's
  harness. It is always attributed as such; we have not run our own A/B.

## Files

| File | What it is |
|---|---|
| `post-dsh-community.md` | Main post for the DSH repo's "Show Your Plugins!" discussions |
| `post-xiaohongshu.md` | Xiaohongshu title options and ready-to-post Chinese body |
| `assets/` | Screenshots referenced by both posts |

## Screenshots

| File | Shows |
|---|---|
| `assets/seed-maze.png` | The image pasted into the composer (the prompt never names it) |
| `assets/01-dungeon-crawler.png` | The first-person dungeon crawler running in a browser: torch-lit corridor, orb, mini-map |
| `assets/02-maze-strike.png` | The CS conversion's start screen: radar with hostiles, AK-47 viewmodel, HP/armor/ammo HUD |
| `assets/03-cs-hud.png` | The CS build mid-session with the round timer and HUD live |
