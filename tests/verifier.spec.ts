import { describe, expect, it } from 'vitest'
import { buildPairMessages, extractPairScores, scoreExpectation } from '../src/verifier.ts'
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

describe('scoreExpectation', () => {
  it('computes the renormalized expectation over legal score tokens', () => {
    // p(9) = 0.6, p(8) = 0.3, p(nonsense) ignored -> E = (0.6*9 + 0.3*8) / 0.9
    const result = scoreExpectation(
      { token: '9', logprob: Math.log(0.6) },
      [
        { token: '9', logprob: Math.log(0.6) },
        { token: '8', logprob: Math.log(0.3) },
        { token: 'ok', logprob: Math.log(0.1) },
      ],
      9,
    )
    expect(result).toBeCloseTo((0.6 * 9 + 0.3 * 8) / 0.9, 6)
  })

  it('ignores out-of-range integers and whitespace-tolerates spellings', () => {
    const result = scoreExpectation(
      { token: ' 17', logprob: Math.log(0.5) },
      [
        { token: '17', logprob: Math.log(0.2) },
        { token: '99', logprob: Math.log(0.2) },
        { token: '0', logprob: Math.log(0.1) },
      ],
      17,
    )
    // " 17" and "17" collapse to one value with the larger mass; 99 and 0 are out of range.
    expect(result).toBeCloseTo(17, 6)
  })

  it('falls back to the parsed integer when no distribution is usable', () => {
    expect(scoreExpectation(undefined, [], 12)).toBe(12)
    expect(scoreExpectation({ token: 'x', logprob: -1 }, [{ token: 'y', logprob: -1 }], 4)).toBe(4)
  })
})

describe('extractPairScores', () => {
  const content = 'A is closer to the spec.\n<score_A>9</score_A>\n<score_B>3</score_B>'

  it('refines both argmax scores with logprob expectations', () => {
    const completion: WireCompletion = {
      choices: [{
        message: { content },
        logprobs: {
          content: entriesFor(content, {
            '9': { logprob: Math.log(0.7), top: [['9', Math.log(0.7)], ['8', Math.log(0.2)]] },
            '3': { logprob: Math.log(0.8), top: [['3', Math.log(0.8)], ['4', Math.log(0.1)]] },
          }),
        },
      }],
    }
    const scores = extractPairScores(completion)
    expect(scores).toBeDefined()
    expect(scores!.a).toBeCloseTo((0.7 * 9 + 0.2 * 8) / 0.9, 6)
    expect(scores!.b).toBeCloseTo((0.8 * 3 + 0.1 * 4) / 0.9, 6)
  })

  it('keeps the argmax when token concatenation does not reproduce the content', () => {
    const completion: WireCompletion = {
      choices: [{
        message: { content },
        logprobs: { content: [{ token: 'mismatch', logprob: -0.5 }] },
      }],
    }
    const scores = extractPairScores(completion)
    expect(scores).toEqual({ a: 9, b: 3 })
  })

  it('keeps the argmax when the score is split across tokens', () => {
    const two = 'ok\n<score_A>17</score_A>\n<score_B>12</score_B>'
    // Split "17" into "1" + "7": single-digit alternatives are incommensurable.
    const tokens = (two.match(/\d|\D+/g) ?? []).map(token => ({ token, logprob: -0.1 }))
    const completion: WireCompletion = {
      choices: [{ message: { content: two }, logprobs: { content: tokens } }],
    }
    const scores = extractPairScores(completion)
    expect(scores).toEqual({ a: 17, b: 12 })
  })

  it('rejects missing tags and out-of-range scores', () => {
    expect(extractPairScores({ choices: [{ message: { content: 'no tags' } }] })).toBeUndefined()
    expect(extractPairScores({
      choices: [{ message: { content: '<score_A>25</score_A><score_B>3</score_B>' } }],
    })).toBeUndefined()
    expect(extractPairScores({})).toBeUndefined()
  })

  it('works without logprobs at all (argmax only)', () => {
    const scores = extractPairScores({ choices: [{ message: { content } }] })
    expect(scores).toEqual({ a: 9, b: 3 })
  })
})

describe('buildPairMessages', () => {
  it('layers messages for prefix caching: context, then pair, then criterion', () => {
    const context = [{ type: 'text' as const, text: 'task transcript' }]
    const first = buildPairMessages(context, 'cand-A', 'cand-B', CODING_CRITERIA[0]!)
    const second = buildPairMessages(context, 'cand-C', 'cand-D', CODING_CRITERIA[1]!)
    const samePairOtherCriterion = buildPairMessages(context, 'cand-A', 'cand-B', CODING_CRITERIA[1]!)
    expect(first).toHaveLength(4)
    // System + context identical across everything in the step.
    expect(JSON.stringify(first[0])).toBe(JSON.stringify(second[0]))
    expect(JSON.stringify(first[1])).toBe(JSON.stringify(second[1]))
    // The pair message is stable across criteria of the same pair (cache!),
    // and differs across pairs.
    expect(JSON.stringify(first[2])).toBe(JSON.stringify(samePairOtherCriterion[2]))
    expect(JSON.stringify(first[2])).not.toBe(JSON.stringify(second[2]))
    // Only the final message carries the criterion.
    expect(JSON.stringify(first[3])).not.toBe(JSON.stringify(samePairOtherCriterion[3]))
    const pair = first[2] as { content: string }
    expect(pair.content).toContain('cand-A')
    expect(pair.content).toContain('cand-B')
    const criterion = first[3] as { content: string }
    expect(criterion.content).toContain(CODING_CRITERIA[0]!.name)
  })
})
