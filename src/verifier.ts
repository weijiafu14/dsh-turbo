/**
 * LLM-as-a-Verifier scoring, faithful to the published method
 * (arXiv 2607.05391): comparative prompts over candidate pairs, integer
 * scores on a 1-20 scale emitted in `<score_A>`/`<score_B>` tags, and a
 * CONTINUOUS reward computed as the expectation over the score token's
 * logprob distribution rather than the argmax — plus repeated evaluation
 * (mean over K independent calls) and criteria decomposition (mean over C
 * criteria), the paper's three verification-scaling dimensions.
 *
 * @module dsh-turbo/verifier
 */

import { LlmError } from '@deepseek-ai/dsh-llm'
import type { Criterion } from './criteria.ts'
import { completeOnce } from './transport.ts'
import type { TransportConnection } from './transport.ts'
import type { WireCompletion, WireContentPart, WireLogprobEntry, WireMessage, WireRequest, WireTopLogprob } from './types.ts'

/** Lowest and highest legal integer score (the paper's finest granularity). */
export const SCORE_MIN = 1
export const SCORE_MAX = 20

/** One pairwise verification outcome: continuous rewards for both sides. */
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
}

const SYSTEM_PROMPT = 'You are a meticulous verifier. You will see the context of a task and two '
  + 'candidate responses, Candidate A and Candidate B, produced independently for that exact '
  + 'context. Evaluate both candidates ONLY on the single criterion you are given. Observed '
  + 'evidence in the context outranks any candidate\'s claims about itself. First write a brief '
  + 'analysis (a few sentences), then output your verdict as two integer scores between '
  + `${SCORE_MIN} and ${SCORE_MAX} (${SCORE_MAX} = flawless on this criterion), in exactly this `
  + 'format and nothing after it:\n<score_A>N</score_A>\n<score_B>N</score_B>'

/**
 * Build the verifier wire messages. The system prompt and the shared task
 * context lead and are byte-identical across every pair/criterion/repeat of
 * one step, so DeepSeek prefix caching absorbs most of the verification input
 * cost; only the final message (pair + criterion) varies.
 */
export function buildPairMessages(
  context: WireContentPart[],
  candidateA: string,
  candidateB: string,
  criterion: Criterion,
): WireMessage[] {
  return [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: [{ type: 'text' as const, text: '=== Task context ===\n' }, ...context] },
    {
      role: 'user',
      content: `=== Candidate A ===\n${candidateA}\n\n=== Candidate B ===\n${candidateB}\n\n`
        + `=== Criterion: ${criterion.name} ===\n${criterion.question}\n\n`
        + `Score both candidates on this criterion alone (${SCORE_MIN}-${SCORE_MAX}).`,
    },
  ]
}

/** Concatenated-token character offsets, for locating a score inside the logprob stream. */
function tokenOffsets(entries: readonly WireLogprobEntry[]): number[] {
  const offsets: number[] = []
  let at = 0
  for (const entry of entries) {
    offsets.push(at)
    at += entry.token.length
  }
  return offsets
}

/** Parse a token as a legal integer score, tolerating surrounding whitespace. */
function scoreValue(token: string): number | undefined {
  if (!/^\s*\d{1,2}\s*$/.test(token)) return undefined
  const value = Number(token.trim())
  return Number.isInteger(value) && value >= SCORE_MIN && value <= SCORE_MAX ? value : undefined
}

/**
 * The paper's continuous reward: the expectation of the score value under the
 * probability distribution over legal score tokens at the score position,
 * renormalized over the alternatives the provider reported.
 * @param sampled - the token actually emitted at the score position.
 * @param alternatives - the `top_logprobs` alternatives at that position.
 * @param fallback - the parsed integer, used when no distribution is usable.
 * @returns a continuous score in [SCORE_MIN, SCORE_MAX].
 */
export function scoreExpectation(
  sampled: WireTopLogprob | undefined,
  alternatives: readonly WireTopLogprob[] | undefined,
  fallback: number,
): number {
  const seen = new Map<number, number>()
  const consider = (candidate: WireTopLogprob | undefined): void => {
    if (candidate === undefined) return
    const value = scoreValue(candidate.token)
    if (value === undefined) return
    const mass = Math.exp(candidate.logprob)
    if (!Number.isFinite(mass) || mass <= 0) return
    // Keep the larger mass when the same value appears in several spellings
    // (" 17" and "17"): they are one score, not double weight.
    seen.set(value, Math.max(seen.get(value) ?? 0, mass))
  }
  consider(sampled)
  for (const alternative of alternatives ?? []) consider(alternative)
  let total = 0
  let weighted = 0
  for (const [value, mass] of seen) {
    total += mass
    weighted += mass * value
  }
  if (total <= 0) return fallback
  return weighted / total
}

/** One tag's parse result: the integer and its digit offset in the content. */
interface ParsedTag {
  value: number
  digitStart: number
  digitLength: number
}

function parseTag(content: string, tag: 'score_A' | 'score_B'): ParsedTag | undefined {
  const match = new RegExp(`<${tag}>\\s*(\\d{1,2})\\s*</${tag}>`).exec(content)
  if (match === null || match[1] === undefined) return undefined
  const value = Number(match[1])
  if (!Number.isInteger(value) || value < SCORE_MIN || value > SCORE_MAX) return undefined
  return {
    value,
    digitStart: match.index + match[0].indexOf(match[1]),
    digitLength: match[1].length,
  }
}

/**
 * Extract both continuous scores from one verifier completion. The argmax
 * integers come from the `<score_A>`/`<score_B>` tags; when the provider's
 * logprob stream covers the score position with a token spelling the WHOLE
 * number, the continuous expectation replaces the argmax (a partial token —
 * "1" of "17" — would make single-digit alternatives incomparable, so those
 * positions keep the parsed integer).
 * @param completion - the non-streaming verifier response with logprobs.
 * @returns both scores, or `undefined` when the tags are missing/invalid.
 */
export function extractPairScores(completion: WireCompletion): PairScores | undefined {
  const choice = completion.choices?.[0]
  const content = choice?.message?.content
  if (typeof content !== 'string') return undefined
  const tagA = parseTag(content, 'score_A')
  const tagB = parseTag(content, 'score_B')
  if (tagA === undefined || tagB === undefined) return undefined

  const entries = choice?.logprobs?.content ?? undefined
  const refine = (tag: ParsedTag): number => {
    if (entries === undefined || entries.length === 0) return tag.value
    const offsets = tokenOffsets(entries)
    // Token concatenation must reproduce the content for offsets to be
    // trustworthy; providers that diverge fall back to the argmax.
    const concatenated = entries.map(entry => entry.token).join('')
    if (concatenated !== content) return tag.value
    let index = -1
    for (let i = 0; i < entries.length; i++) {
      const start = offsets[i] ?? 0
      const token = entries[i]?.token ?? ''
      if (tag.digitStart >= start && tag.digitStart < start + token.length) {
        index = i
        break
      }
    }
    if (index === -1) return tag.value
    const entry = entries[index]
    if (entry === undefined) return tag.value
    // The sampled token must spell the whole number (modulo whitespace) for
    // the alternatives at this position to be commensurable scores.
    if (entry.token.trim() !== content.slice(tag.digitStart, tag.digitStart + tag.digitLength)) {
      return tag.value
    }
    return scoreExpectation(
      { token: entry.token, logprob: entry.logprob },
      entry.top_logprobs,
      tag.value,
    )
  }

  return { a: refine(tagA), b: refine(tagB) }
}

function mapVerifierUsage(completion: WireCompletion, options: VerifierCallOptions): void {
  const usage = completion.usage
  if (usage == null || options.onUsage === undefined) return
  const cacheRead = usage.prompt_tokens_details?.cached_tokens ?? usage.prompt_cache_hit_tokens ?? 0
  options.onUsage(usage.prompt_tokens - cacheRead, usage.completion_tokens, cacheRead)
}

/**
 * One pairwise evaluation: C criteria x K repeats, each an independent
 * verifier call, averaged into one continuous reward per side (the paper's
 * Monte Carlo mean over repeats and arithmetic mean over criteria).
 * A single failed or unparseable call is skipped; the pair fails only when
 * every call failed.
 * @param context - shared task-context parts (cache-stable prefix).
 * @param candidateA - rendered candidate A.
 * @param candidateB - rendered candidate B.
 * @param criteria - the decomposed criteria set.
 * @param repeats - K independent evaluations per criterion.
 * @param call - transport facts for the verifier calls.
 * @returns mean continuous scores for both sides.
 */
export async function scorePair(
  context: WireContentPart[],
  candidateA: string,
  candidateB: string,
  criteria: readonly Criterion[],
  repeats: number,
  call: VerifierCallOptions,
): Promise<PairScores> {
  const requests: Promise<PairScores | undefined>[] = []
  for (const criterion of criteria) {
    const messages = buildPairMessages(context, candidateA, candidateB, criterion)
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
            return extractPairScores(completion)
          })
          .catch((error: unknown) => {
            if (call.signal.aborted) throw error
            return undefined
          }),
      )
    }
  }
  const settled = await Promise.all(requests)
  const scored = settled.filter((entry): entry is PairScores => entry !== undefined)
  if (scored.length === 0) {
    throw new LlmError('dsh-turbo: every verifier call for a pair failed or returned no scores', 'VERIFIER_FAILED')
  }
  const meanA = scored.reduce((sum, entry) => sum + entry.a, 0) / scored.length
  const meanB = scored.reduce((sum, entry) => sum + entry.b, 0) / scored.length
  return { a: meanA, b: meanB }
}
