/**
 * LLM-as-a-Verifier scoring, aligned with the reference implementation
 * (llm_verifier/fine_grained_reward.py, upstream v0.2.0, 2026-08-14) and the
 * paper it comes from (arXiv 2607.05391): comparative prompts over candidate
 * pairs, a 20-point scale whose anchors describe how likely the candidate
 * actually solved the task, and a CONTINUOUS reward taken as the expectation
 * over the score token's logprob distribution rather than the argmax — plus
 * repeated evaluation (mean over K calls) and criteria decomposition (mean
 * over C criteria).
 *
 * Two deliberate deviations from upstream, both measured on DeepSeek:
 *
 *  - Upstream scores with letters A-T because a single letter is reliably one
 *    token on every backend. On DeepSeek the integers 1-20 also arrive as one
 *    token, and they carry MORE usable mass: probing the live API, the score
 *    position offered 18/20 legal integer alternatives versus 5/20 legal
 *    letter alternatives (DeepSeek fuses the closing `>` into the letter
 *    token, so most alternatives at that position are unusable). Integers win
 *    here, so this implementation keeps them — and still strips a fused `>`
 *    prefix, exactly as upstream must.
 *  - Rewards are normalized to [0, 1] as upstream does, so a missing or
 *    unparseable evaluation can default to a neutral 0.5 instead of being
 *    dropped, and Bradley-Terry sees the same scale the reference does.
 *
 * @module dsh-turbo/verifier
 */

import type { Criterion, CriteriaSet } from './criteria.ts'
import { completeOnce } from './transport.ts'
import type { TransportConnection } from './transport.ts'
import type { WireCompletion, WireContentPart, WireLogprobEntry, WireMessage, WireRequest, WireTopLogprob } from './types.ts'

/** Lowest and highest legal integer score (the paper's finest granularity, g = 20). */
export const SCORE_MIN = 1
export const SCORE_MAX = 20
/** Neutral reward for an evaluation that produced nothing usable. */
export const NEUTRAL_REWARD = 0.5

/** One pairwise verification outcome: normalized rewards in [0, 1] for both sides. */
export interface PairScores {
  a: number
  b: number
}

/** Everything one verifier call needs beyond the pair itself. */
export interface VerifierCallOptions {
  connection: TransportConnection
  apiKey: string
  model: string
  temperature: number
  maxTokens: number
  signal: AbortSignal
  /** Observed token accounting for cost reporting. */
  onUsage?: (inputTokens: number, outputTokens: number, cacheReadTokens: number) => void
  /** Reports evaluations that fell back to the neutral reward. */
  onNeutral?: (reason: 'call-failed' | 'unparseable') => void
  /** Reports evaluations that produced a real score. */
  onScored?: () => void
}

/**
 * The rating scale, anchored on how likely the candidate actually solved the
 * task — upstream's semantics, carried over verbatim in meaning. A generic
 * "quality" scale invites the verifier to reward polish; this one asks for
 * evidence of success.
 */
const SCALE_DESCRIPTION = 'Rate how likely the candidate correctly solved the task on a '
  + '20-point scale:\n'
  + '  20 = clearly and completely succeeded, with verified output (best)\n'
  + '  17-19 = succeeded with only minor issues\n'
  + '  14-16 = above average, mostly correct with some issues\n'
  + '  11-13 = uncertain, leans toward success\n'
  + '  8-10 = uncertain, leans toward failure\n'
  + '  5-7 = below average, significant issues remain\n'
  + '  2-4 = failed with some partial progress\n'
  + '  1 = clearly and completely failed (worst)'

const SYSTEM_PROMPT = 'You are a meticulous verifier. You will see the context of a task and two '
  + 'candidate responses, Candidate A and Candidate B, produced independently for that exact '
  + 'context. Evaluate both candidates ONLY on the single criterion you are given.\n\n'
  + `${SCALE_DESCRIPTION}\n\n`
  + 'First write a brief analysis (a few sentences), then output your verdict as two integers '
  + `between ${SCORE_MIN} and ${SCORE_MAX}, in exactly this format and nothing after it:\n`
  + '<score_A>N</score_A>\n<score_B>N</score_B>'

/**
 * Build the verifier wire messages.
 *
 * Message layering is deliberate cache economics, matching upstream's
 * "keep criterion-specific text strictly at the end": system + task context are
 * byte-identical across the whole step, and the candidate-pair message is
 * byte-identical across this pair's C x K criterion calls — so provider prefix
 * caching pays for the candidates (often whole generated files) once per pair
 * instead of once per call.
 */
export function buildPairMessages(
  context: WireContentPart[],
  candidateA: string,
  candidateB: string,
  criterion: Criterion,
  groundTruthNote: string,
): WireMessage[] {
  return [
    { role: 'system', content: SYSTEM_PROMPT },
    {
      role: 'user',
      content: [
        { type: 'text' as const, text: `=== How to judge ===\n${groundTruthNote}\n\n=== Task context ===\n` },
        ...context,
      ],
    },
    {
      role: 'user',
      content: `=== Candidate A ===\n${candidateA}\n\n=== Candidate B ===\n${candidateB}`,
    },
    {
      role: 'user',
      content: `=== Criterion: ${criterion.name} ===\n${criterion.question}\n\n`
        + `Score both candidates on this criterion alone (${SCORE_MIN}-${SCORE_MAX}).`,
    },
  ]
}

/** Strip a fused closing bracket: DeepSeek emits `>20` as one token after the tag. */
function normalizeToken(token: string): string {
  const trimmed = token.trim()
  return trimmed.startsWith('>') ? trimmed.slice(1).trim() : trimmed
}

/** Parse a token as a legal integer score. */
function scoreValue(token: string): number | undefined {
  const text = normalizeToken(token)
  if (!/^\d{1,2}$/.test(text)) return undefined
  const value = Number(text)
  return Number.isInteger(value) && value >= SCORE_MIN && value <= SCORE_MAX ? value : undefined
}

/** Map a raw 1-20 score onto the normalized [0, 1] reward scale. */
export function normalizeScore(value: number): number {
  return (value - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)
}

/**
 * The alternatives the provider reported at the score position of `tag`.
 *
 * The LAST occurrence wins: the verdict is the score block at the end of the
 * reply, and a model that quotes the output format mid-analysis would
 * otherwise shadow it. Whitespace-only tokens are skipped so they cannot make
 * the tag appear to match twice at the same place — both behaviours are
 * upstream's, learned from its issues #5/#10/#11.
 * @param entries - per-token logprob entries of the completion.
 * @param tag - the opening tag whose following position holds the score.
 * @returns the alternatives at that position, or undefined.
 */
export function findTagLogprobs(
  entries: readonly WireLogprobEntry[],
  tag: string,
): readonly WireTopLogprob[] | undefined {
  if (entries.length === 0) return undefined
  // Some tokenizers fuse the closing '>' with the score, so try the exact tag
  // first and then the tag without it.
  for (const suffix of [tag, tag.slice(0, -1)]) {
    let found: readonly WireTopLogprob[] | undefined
    let textSoFar = ''
    for (let i = 0; i < entries.length; i++) {
      const token = entries[i]?.token ?? ''
      textSoFar += token
      if (token.trim().length === 0) continue
      if (textSoFar.trimEnd().endsWith(suffix)) {
        const next = entries[i + 1]
        if (next !== undefined) {
          found = next.top_logprobs !== undefined && next.top_logprobs.length > 0
            ? next.top_logprobs
            : [{ token: next.token, logprob: next.logprob }]
        }
      }
    }
    if (found !== undefined) return found
  }
  return undefined
}

/**
 * The paper's continuous reward: the expectation of the score value under the
 * probability distribution over legal score tokens at the score position,
 * renormalized over the alternatives the provider reported.
 * @param alternatives - the alternatives at the score position.
 * @returns a normalized reward in [0, 1], or undefined when no alternative is a legal score.
 */
export function scoreExpectation(alternatives: readonly WireTopLogprob[] | undefined): number | undefined {
  const mass = new Map<number, number>()
  for (const alternative of alternatives ?? []) {
    const value = scoreValue(alternative.token)
    if (value === undefined) continue
    const probability = Math.exp(alternative.logprob)
    if (!Number.isFinite(probability) || probability <= 0) continue
    // The same value in several spellings (" 17", ">17") is one score, not
    // double weight.
    mass.set(value, Math.max(mass.get(value) ?? 0, probability))
  }
  let total = 0
  let weighted = 0
  for (const [value, probability] of mass) {
    total += probability
    weighted += probability * value
  }
  if (total <= 0) return undefined
  return normalizeScore(weighted / total)
}

/** Last literal `<tag>N</tag>` in the text, as a normalized reward. */
function literalScore(content: string, tag: 'score_A' | 'score_B'): number | undefined {
  const matches = [...content.matchAll(new RegExp(`<${tag}>\\s*(\\d{1,2})\\s*</${tag}>`, 'giu'))]
  const last = matches[matches.length - 1]
  if (last?.[1] === undefined) return undefined
  const value = Number(last[1])
  if (!Number.isInteger(value) || value < SCORE_MIN || value > SCORE_MAX) return undefined
  return normalizeScore(value)
}

/**
 * Extract both normalized rewards from one verifier completion: the logprob
 * expectation at each tag, falling back to the literal parsed score, and to a
 * neutral 0.5 when neither is available (upstream's behaviour — a criterion
 * that could not be read must not silently drop out of the average).
 * @param completion - the non-streaming verifier response with logprobs.
 * @returns rewards in [0, 1] for both slots, and whether anything was usable.
 */
export function extractPairScores(completion: WireCompletion): { scores: PairScores; usable: boolean } {
  const choice = completion.choices?.[0]
  const content = typeof choice?.message?.content === 'string' ? choice.message.content : ''
  const entries = choice?.logprobs?.content ?? []
  const read = (tag: 'score_A' | 'score_B'): number | undefined =>
    scoreExpectation(findTagLogprobs(entries, `<${tag}>`)) ?? literalScore(content, tag)
  const a = read('score_A')
  const b = read('score_B')
  return {
    scores: { a: a ?? NEUTRAL_REWARD, b: b ?? NEUTRAL_REWARD },
    usable: a !== undefined && b !== undefined,
  }
}

function mapVerifierUsage(completion: WireCompletion, options: VerifierCallOptions): void {
  const usage = completion.usage
  if (usage == null || options.onUsage === undefined) return
  const cacheRead = usage.prompt_tokens_details?.cached_tokens ?? usage.prompt_cache_hit_tokens ?? 0
  options.onUsage(usage.prompt_tokens - cacheRead, usage.completion_tokens, cacheRead)
}

/**
 * One pairwise evaluation: C criteria x K repeats, each an independent
 * verifier call, averaged into one reward per side (the paper's Monte Carlo
 * mean over repeats and arithmetic mean over criteria).
 *
 * A failed or unreadable evaluation contributes the neutral 0.5 rather than
 * being dropped, matching upstream: dropping would let one broken criterion
 * silently re-weight the rest.
 * @param context - shared task-context parts (cache-stable prefix).
 * @param candidateA - rendered candidate A.
 * @param candidateB - rendered candidate B.
 * @param set - the criteria set, including the standing ground-truth note.
 * @param repeats - K independent evaluations per criterion.
 * @param call - transport facts for the verifier calls.
 * @returns mean normalized rewards in [0, 1] for both sides.
 */
export async function scorePair(
  context: WireContentPart[],
  candidateA: string,
  candidateB: string,
  set: CriteriaSet,
  repeats: number,
  call: VerifierCallOptions,
): Promise<PairScores> {
  const requests: Promise<PairScores>[] = []
  for (const criterion of set.criteria) {
    const messages = buildPairMessages(context, candidateA, candidateB, criterion, set.groundTruthNote)
    const body: WireRequest = {
      model: call.model,
      messages,
      stream: false,
      thinking: { type: 'disabled' },
      temperature: call.temperature,
      max_tokens: call.maxTokens,
      logprobs: true,
      top_logprobs: 20,
    }
    for (let k = 0; k < repeats; k++) {
      requests.push(
        completeOnce(body, call.connection, call.apiKey, call.signal)
          .then((completion) => {
            mapVerifierUsage(completion, call)
            const { scores, usable } = extractPairScores(completion)
            if (usable) call.onScored?.()
            else call.onNeutral?.('unparseable')
            return scores
          })
          .catch((error: unknown) => {
            if (call.signal.aborted) throw error
            call.onNeutral?.('call-failed')
            return { a: NEUTRAL_REWARD, b: NEUTRAL_REWARD }
          }),
      )
    }
  }
  const scored = await Promise.all(requests)
  const meanA = scored.reduce((sum, entry) => sum + entry.a, 0) / scored.length
  const meanB = scored.reduce((sum, entry) => sum + entry.b, 0) / scored.length
  return { a: meanA, b: meanB }
}
