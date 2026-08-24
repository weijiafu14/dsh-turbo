import { describe, expect, it } from 'vitest'
import { pivotRoundPairs, preferenceProbability, ringCycle, runTournament } from '../src/tournament.ts'
import type { PairScores } from '../src/verifier.ts'

/** Deterministic LCG so tournament structure is reproducible. */
function seeded(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0
    return state / 2 ** 32
  }
}

describe('preferenceProbability', () => {
  it('is 0.5 on equal rewards and monotone in the difference', () => {
    expect(preferenceProbability(0.5, 0.5)).toBeCloseTo(0.5)
    expect(preferenceProbability(0.9, 0.1)).toBeGreaterThan(0.5)
    expect(preferenceProbability(0.1, 0.9)).toBeLessThan(0.5)
  })

  it('is complementary across orientations', () => {
    expect(preferenceProbability(0.8, 0.3) + preferenceProbability(0.3, 0.8)).toBeCloseTo(1)
  })
})

describe('ringCycle', () => {
  it('gives every candidate exactly one turn in each slot', () => {
    const ring = ringCycle(5, seeded(3))
    expect(ring).toHaveLength(5)
    expect(ring.map(([a]) => a).sort()).toEqual([0, 1, 2, 3, 4])
    expect(ring.map(([, b]) => b).sort()).toEqual([0, 1, 2, 3, 4])
  })

  it('scores two candidates in BOTH orientations', () => {
    const ring = ringCycle(2, seeded(1))
    expect(ring).toHaveLength(2)
    expect(ring[0]?.[0]).toBe(ring[1]?.[1])
    expect(ring[0]?.[1]).toBe(ring[1]?.[0])
  })
})

describe('pivotRoundPairs', () => {
  it('puts non-pivots in slot A and pairs the pivots by ascending index', () => {
    const pairs = pivotRoundPairs(5, [3, 1])
    // 3 non-pivots x 2 pivots + 1 pivot pair
    expect(pairs).toHaveLength(7)
    for (const [a, b] of pairs.slice(0, 6)) {
      expect([3, 1]).toContain(b)
      expect([0, 2, 4]).toContain(a)
    }
    expect(pairs.at(-1)).toEqual([1, 3])
  })
})

describe('runTournament', () => {
  const byQuality = (quality: number[]) =>
    (a: number, b: number): Promise<PairScores> =>
      Promise.resolve({ a: quality[a] as number, b: quality[b] as number })

  it('rejects fewer than two candidates', async () => {
    await expect(runTournament(1, byQuality([0.5]), { pivots: 2 })).rejects.toThrow(/at least two/)
  })

  it('picks the clearly best of five and stays sub-quadratic', async () => {
    const outcome = await runTournament(5, byQuality([0.4, 0.6, 0.95, 0.2, 0.5]), { pivots: 2, random: seeded(42) })
    expect(outcome.winner).toBe(2)
    // ring 5 + non-pivots 3x2 + pivot pair 1 = 12, versus 20 for a directed round robin
    expect(outcome.pairs).toHaveLength(12)
    expect(outcome.pivots).toContain(2)
  })

  it('ranks by MEAN preference, so extra pivot comparisons cannot buy a win', async () => {
    // A mediocre candidate that happens to become a pivot takes part in many
    // more comparisons; ranking by total wins would hand it the tournament.
    const quality = [0.55, 0.54, 0.53, 0.52, 0.95]
    const outcome = await runTournament(5, byQuality(quality), { pivots: 2, random: seeded(7) })
    expect(outcome.winner).toBe(4)
    const pivotTotals = outcome.pivots.map(p => outcome.wins[p] as number)
    const winnerTotal = outcome.wins[outcome.winner] as number
    if (!outcome.pivots.includes(outcome.winner)) {
      // The evidence for the guard: a pivot really does accumulate more raw wins.
      expect(Math.max(...pivotTotals)).toBeGreaterThan(winnerTotal)
    }
    // Mean preference is what decides, and the best candidate leads on it.
    const best = outcome.meanPreference[4] as number
    for (const [i, value] of outcome.meanPreference.entries()) {
      if (i !== 4) expect(best).toBeGreaterThan(value)
    }
  })

  it('is winner-stable across seeds when one candidate dominates', async () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      const outcome = await runTournament(5, byQuality([0.5, 0.5, 0.5, 0.98, 0.5]), { pivots: 2, random: seeded(seed) })
      expect(outcome.winner).toBe(3)
    }
  })

  it('counts every comparison for both sides', async () => {
    const outcome = await runTournament(4, byQuality([0.3, 0.6, 0.8, 0.5]), { pivots: 2, random: seeded(11) })
    const totalSlots = outcome.comparisons.reduce((s, c) => s + c, 0)
    expect(totalSlots).toBe(outcome.pairs.length * 2)
    const totalWins = outcome.wins.reduce((s, w) => s + w, 0)
    expect(totalWins).toBeCloseTo(outcome.pairs.length)
  })

  it('separates the ring pass from the pivot round in the record', async () => {
    const outcome = await runTournament(4, byQuality([0.3, 0.6, 0.8, 0.5]), { pivots: 2, random: seeded(5) })
    expect(outcome.pairs.filter(p => p.phase === 'ring')).toHaveLength(4)
    expect(outcome.pairs.filter(p => p.phase === 'pivot').length).toBeGreaterThan(0)
  })

  it('reports monotone progress up to the final total', async () => {
    const seen: [number, number][] = []
    await runTournament(4, byQuality([0.3, 0.6, 0.8, 0.5]), {
      pivots: 2,
      random: seeded(11),
      onProgress: (done, total) => seen.push([done, total]),
    })
    const [lastDone, lastTotal] = seen[seen.length - 1] as [number, number]
    expect(lastDone).toBe(lastTotal)
  })
})
