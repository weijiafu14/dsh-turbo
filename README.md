# dsh-turbo

Best-of-N sampling with an **LLM-as-a-Verifier** judge for [DeepSeek Harness](https://www.npmjs.com/package/@deepseek-ai/dsh) — as one ordinary model route.

Pick the `deepseek-turbo` provider in your model selector and every model call samples **N candidates in parallel**, scores them with **fine-grained logprob verification**, and hands your agent the **tournament winner**. Faithful implementation of [LLM-as-a-Verifier](https://scalingintelligence.stanford.edu/pubs/llm_as_a_verifier/) (Stanford Scaling Intelligence Lab): sampling 5 candidates with DeepSeek and ranking them with the same model lifted Terminal-Bench accuracy from 79% to 88% in the paper — outperforming closed frontier models at a fraction of the cost.

## Install

```bash
dsh plugin add dsh-turbo
```

That's it if you already use the official DeepSeek route: dsh-turbo reads the **same `DEEPSEEK_API_KEY` credential** (credentials service first, environment second). Otherwise:

```bash
export DEEPSEEK_API_KEY=sk-...
```

Restart `dsh`, open the model selector, choose **DeepSeek Turbo → DeepSeek-V4-Flash-Vision-Exp (Turbo)**. Done — no other configuration exists between you and best-of-5.

Default models, in order:

| Model | Input | Notes |
|---|---|---|
| `deepseek-v4-flash-vision-exp` | text + images | default; DeepSeek's multimodal model |
| `deepseek-v4-flash` | text | |
| `deepseek-v4-pro` | text | |

Images flow end to end: candidates receive them as real multimodal input, and the verifier sees the same pixels when judging.

## How it works

Every conversation model call (auxiliary calls like compaction/session titles pass straight through):

1. **Sample** — N=5 parallel generations against the DeepSeek API, spread over a temperature ladder.
2. **Verify** — pairwise comparative prompts; the verifier answers with integer scores 1–20 in `<score_A>`/`<score_B>` tags, requested with `logprobs: true, top_logprobs: 20`. The continuous reward is the **expectation over the score token's probability distribution**, not the argmax — the paper's key signal (one logprob evaluation ≈ 16 discrete-judge evaluations). Scores decompose over three criteria (specification adherence / output match / error signals) and can repeat K times.
3. **Select** — a **Probabilistic Pivot Tournament**: one ring pass over a random circuit (each candidate judged once as A and once as B, cancelling position bias), top ring scorers become pivots, everyone else is compared against the pivots, and Bradley-Terry preference probabilities accumulate into win totals. Sub-quadratic in N.
4. **Replay** — the winner's stream chunks replay to the agent loop, which sees one ordinary model call. The winner's own token usage is reported (so context accounting stays correct); the extra spend of losing candidates and verifier calls is tracked separately.

Verifier prompts share a byte-identical prefix (system + task context) across all pairs and criteria of a step, so DeepSeek prefix caching absorbs most of the verification input cost.

## The `/turbo` command

- `/turbo` — last decision (per-candidate scores, tournament outcome, previews) and cumulative extra spend.
- `/turbo off` / `/turbo on` — temporarily pass single calls through without switching models.

Decision records also append to `$DSH_HOME/dsh-turbo/decisions.jsonl` for offline analysis.

## Configuration (optional)

Everything lives in the `dsh-turbo:` section of `$DSH_HOME/settings.yaml` (hot-reloaded):

```yaml
dsh-turbo:
  n: 5                  # candidates per call; 1 = plain passthrough
  repeats: 1            # K verifier repeats per pair and criterion
  pivots: 2             # ring leaders anchoring the comparison round
  criteria: coding      # coding | general
  temperatures: [0.2, 0.5, 0.8, 1.0, 1.2]
  verifierModel: deepseek-v4-flash-vision-exp   # default: the candidate model
  verifierTemperature: 1
  verifierMaxTokens: 2000
  apiKeyEnv: DEEPSEEK_API_KEY
  baseURL: https://api.deepseek.com   # any OpenAI-compatible endpoint that returns logprobs (vLLM!)
  decisionLog: true
```

Pointing `baseURL` at a vLLM (or any OpenAI-compatible, logprob-returning) server gives your self-hosted models the same best-of-N treatment.

## Cost and latency

Best-of-N is a deliberate trade: with defaults, one step costs ~5 candidate generations plus ~36 short verifier calls (most of whose input is prefix-cache hits). Responses arrive after all candidates and verification complete — the status line shows sampling/verifying progress while you wait. Budget option: `n: 3, pivots: 1` roughly halves the extra spend.

## Notes

- A candidate that fails (transport error, empty response) is dropped; survivors still compete. If **verification** itself fails, the first surviving candidate is used and the decision is recorded — and reported — as degraded.
- `/turbo` and the status line appear only on surfaces that provide the commands / TUI status services; everything else works headless.

## License

MIT
