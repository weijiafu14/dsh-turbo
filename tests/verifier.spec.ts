import { describe, expect, it } from 'vitest'
import {
  buildPairMessages,
  extractPairScores,
  findTagLogprobs,
  NEUTRAL_REWARD,
  normalizeScore,
  scoreExpectation,
} from '../src/verifier.ts'
import { CODING_CRITERIA } from '../src/criteria.ts'
import type { WireCompletion, WireLogprobEntry } from '../src/types.ts'

/** Tokenize content into logprob entries, keeping digit runs whole. */
function entriesFor(content: string, at: Record<string, { logprob: number; top: [string, number][] }>): WireLogprobEntry[] {
  const tokens = content.match(/\d+|\D+/g) ?? []
  return tokens.map((token) => {
    const spec = at[token]
    return {
      token,
      logprob: spec?.logprob ?? -0.01,
      ...spec === undefined
        ? {}
        : { top_logprobs: spec.top.map(([t, lp]) => ({ token: t, logprob: lp })) },
    }
  })
}

const lp = (p: number) => Math.log(p)

describe('scoreExpectation', () => {
  it('computes the renormalized expectation, mapped to [0, 1]', () => {
    // p(9) = 0.6, p(8) = 0.3, illegal token ignored -> E = (0.6*9 + 0.3*8) / 0.9
    const raw = (0.6 * 9 + 0.3 * 8) / 0.9
    const result = scoreExpectation([
      { token: '9', logprob: lp(0.6) },
      { token: '8', logprob: lp(0.3) },
      { token: 'ok', logprob: lp(0.1) },
    ])
    expect(result).toBeCloseTo(normalizeScore(raw), 6)
  })

  it('strips a fused closing bracket, as DeepSeek emits it', () => {
    // ">17" is how DeepSeek fuses the tag's '>' into the score token.
    const result = scoreExpectation([{ token: '>17', logprob: lp(0.9) }])
    expect(result).toBeCloseTo(normalizeScore(17), 6)
  })

  it('collapses one value spelled several ways instead of double-counting it', () => {
    const result = scoreExpectation([
      { token: ' 17', logprob: lp(0.5) },
      { token: '17', logprob: lp(0.2) },
      { token: '>17', logprob: lp(0.1) },
    ])
    expect(result).toBeCloseTo(normalizeScore(17), 6)
  })

  it('ignores out-of-range integers', () => {
    const result = scoreExpectation([
      { token: '20', logprob: lp(0.5) },
      { token: '99', logprob: lp(0.4) },
      { token: '0', logprob: lp(0.1) },
    ])
    expect(result).toBeCloseTo(normalizeScore(20), 6)
  })

  it('returns undefined when no alternative is a legal score', () => {
    expect(scoreExpectation(undefined)).toBeUndefined()
    expect(scoreExpectation([{ token: 'x', logprob: -1 }])).toBeUndefined()
  })

  it('maps the scale endpoints to 0 and 1', () => {
    expect(normalizeScore(1)).toBe(0)
    expect(normalizeScore(20)).toBe(1)
  })
})

describe('findTagLogprobs', () => {
  it('takes the LAST occurrence, so a quoted format cannot shadow the verdict', () => {
    // The model restates the output format mid-analysis, then gives its verdict.
    const content = 'I will answer as <score_A>N</score_A>.\nVerdict:\n<score_A>18</score_A>'
    const entries = entriesFor(content, {
      // The literal 'N' position carries a decoy distribution.
      '18': { logprob: lp(0.9), top: [['18', lp(0.9)]] },
    })
    const found = findTagLogprobs(entries, '<score_A>')
    expect(found).toBeDefined()
    expect(found?.some(alt => alt.token.includes('18'))).toBe(true)
  })

  it('skips whitespace-only tokens when locating the tag', () => {
    const entries: WireLogprobEntry[] = [
      { token: '<score_A>', logprob: -0.1 },
      { token: '   ', logprob: -0.1 },
      { token: '12', logprob: -0.1, top_logprobs: [{ token: '12', logprob: lp(0.9) }] },
    ]
    const found = findTagLogprobs(entries, '<score_A>')
    // The whitespace token must not make the tag look like it matched twice and
    // shadow the real distribution that follows.
    expect(found?.[0]?.token).toBe('   ')
  })

  it('returns undefined without entries', () => {
    expect(findTagLogprobs([], '<score_A>')).toBeUndefined()
  })
})

describe('extractPairScores', () => {
  const content = 'A is closer to the spec.\n<score_A>9</score_A>\n<score_B>3</score_B>'

  it('refines both scores with logprob expectations', () => {
    const completion: WireCompletion = {
      choices: [{
        message: { content },
        logprobs: {
          content: entriesFor(content, {
            '9': { logprob: lp(0.7), top: [['9', lp(0.7)], ['8', lp(0.2)]] },
            '3': { logprob: lp(0.8), top: [['3', lp(0.8)], ['4', lp(0.1)]] },
          }),
        },
      }],
    }
    const { scores, usable } = extractPairScores(completion)
    expect(usable).toBe(true)
    expect(scores.a).toBeCloseTo(normalizeScore((0.7 * 9 + 0.2 * 8) / 0.9), 6)
    expect(scores.b).toBeCloseTo(normalizeScore((0.8 * 3 + 0.1 * 4) / 0.9), 6)
  })

  it('falls back to the literal score when logprobs carry nothing usable', () => {
    const completion: WireCompletion = {
      choices: [{ message: { content }, logprobs: { content: [{ token: 'mismatch', logprob: -0.5 }] } }],
    }
    const { scores, usable } = extractPairScores(completion)
    expect(usable).toBe(true)
    expect(scores.a).toBeCloseTo(normalizeScore(9), 6)
    expect(scores.b).toBeCloseTo(normalizeScore(3), 6)
  })

  it('takes the last literal score block, not a quoted example', () => {
    const quoted = 'Format: <score_A>N</score_A>. Verdict:\n<score_A>15</score_A>\n<score_B>4</score_B>'
    const { scores } = extractPairScores({ choices: [{ message: { content: quoted } }] })
    expect(scores.a).toBeCloseTo(normalizeScore(15), 6)
  })

  it('returns the neutral reward, flagged unusable, when nothing parses', () => {
    const { scores, usable } = extractPairScores({ choices: [{ message: { content: 'no tags at all' } }] })
    expect(usable).toBe(false)
    expect(scores).toEqual({ a: NEUTRAL_REWARD, b: NEUTRAL_REWARD })
  })

  it('rejects out-of-range literal scores', () => {
    const { usable } = extractPairScores({
      choices: [{ message: { content: '<score_A>25</score_A><score_B>3</score_B>' } }],
    })
    expect(usable).toBe(false)
  })

  it('handles an empty completion', () => {
    const { scores, usable } = extractPairScores({})
    expect(usable).toBe(false)
    expect(scores).toEqual({ a: NEUTRAL_REWARD, b: NEUTRAL_REWARD })
  })
})

describe('buildPairMessages', () => {
  it('layers messages for prefix caching and carries the ground-truth note', () => {
    const context = [{ type: 'text' as const, text: 'task transcript' }]
    const criteria = CODING_CRITERIA.criteria
    const note = CODING_CRITERIA.groundTruthNote
    const first = buildPairMessages(context, 'cand-A', 'cand-B', criteria[0]!, note)
    const other = buildPairMessages(context, 'cand-C', 'cand-D', criteria[1]!, note)
    const samePairOtherCriterion = buildPairMessages(context, 'cand-A', 'cand-B', criteria[1]!, note)
    expect(first).toHaveLength(4)
    // System + note/context identical across everything in the step.
    expect(JSON.stringify(first[0])).toBe(JSON.stringify(other[0]))
    expect(JSON.stringify(first[1])).toBe(JSON.stringify(other[1]))
    expect(JSON.stringify(first[1])).toContain('How to judge')
    // The pair message is stable across criteria of the same pair (cache!),
    // and differs across pairs.
    expect(JSON.stringify(first[2])).toBe(JSON.stringify(samePairOtherCriterion[2]))
    expect(JSON.stringify(first[2])).not.toBe(JSON.stringify(other[2]))
    // Only the final message carries the criterion.
    expect(JSON.stringify(first[3])).not.toBe(JSON.stringify(samePairOtherCriterion[3]))
    const criterion = first[3] as { content: string }
    expect(criterion.content).toContain(criteria[0]!.name)
  })

  it('states the success-likelihood scale rather than generic quality', () => {
    const [system] = buildPairMessages([], 'a', 'b', CODING_CRITERIA.criteria[0]!, 'note')
    const text = (system as { content: string }).content
    expect(text).toContain('how likely the candidate correctly solved the task')
    expect(text).toContain('20 = clearly and completely succeeded')
    expect(text).toContain('1 = clearly and completely failed')
  })
})
