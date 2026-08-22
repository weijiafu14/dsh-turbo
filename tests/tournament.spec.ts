import { describe, expect, it } from 'vitest'
import { preferenceProbability, runTournament } from '../src/tournament.ts'
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
  it('is 0.5 on equal scores and monotone in the difference', () => {
    expect(preferenceProbability(10, 10)).toBeCloseTo(0.5)
    expect(preferenceProbability(12, 10)).toBeGreaterThan(0.5)
    expect(preferenceProbability(8, 10)).toBeLessThan(0.5)
    expect(preferenceProbability(20, 1)).toBeGreaterThan(0.99)
  })

  it('is complementary across orientations', () => {
    expect(preferenceProbability(14, 9) + preferenceProbability(9, 14)).toBeCloseTo(1)
  })
})

describe('runTournament', () => {
  const byQuality = (quality: number[]) =>
    (a: number, b: number): Promise<PairScores> =>
      Promise.resolve({ a: quality[a] as number, b: quality[b] as number })

  it('rejects fewer than two candidates', async () => {
    await expect(runTournament(1, byQuality([5]), { pivots: 2 })).rejects.toThrow(/at least two/)
  })

  it('decides two candidates with the single ring pair', async () => {
    const outcome = await runTournament(2, byQuality([6, 15]), { pivots: 2, random: seeded(7) })
    expect(outcome.winner).toBe(1)
    expect(outcome.pairs).toHaveLength(1)
  })

  it('picks the clearly best of five and keeps pair counts sub-quadratic', async () => {
    const quality = [8, 12, 18, 6, 10]
    const outcome = await runTournament(5, byQuality(quality), { pivots: 2, random: seeded(42) })
    expect(outcome.winner).toBe(2)
    // Ring (5) + comparison round bounded by non-pivots x pivots + pivot pair,
    // minus ring overlap: never the full C(5,2) x both orientations.
    expect(outcome.pairs.length).toBeGreaterThanOrEqual(5)
    expect(outcome.pairs.length).toBeLessThanOrEqual(12)
    expect(outcome.pivots).toContain(2)
    // Every candidate appears as A exactly once in the ring pass.
    const asA = outcome.pairs.slice(0, 5).map(pair => pair.a).sort()
    expect(asA).toEqual([0, 1, 2, 3, 4])
  })

  it('is winner-stable across seeds when one candidate dominates', async () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      const outcome = await runTournament(5, byQuality([5, 5, 5, 19, 5]), { pivots: 2, random: seeded(seed) })
      expect(outcome.winner).toBe(3)
    }
  })

  it('accumulates Bradley-Terry wins from every evaluated pair', async () => {
    const outcome = await runTournament(3, byQuality([4, 10, 16]), { pivots: 1, random: seeded(9) })
    const total = outcome.wins.reduce((sum, w) => sum + w, 0)
    expect(total).toBeCloseTo(outcome.pairs.length)
    expect(outcome.wins[2]).toBeGreaterThan(outcome.wins[0] as number)
  })

  it('reports monotone progress up to the final total', async () => {
    const seen: [number, number][] = []
    await runTournament(4, byQuality([3, 9, 12, 7]), {
      pivots: 2,
      random: seeded(11),
      onProgress: (done, total) => seen.push([done, total]),
    })
    expect(seen.length).toBeGreaterThan(0)
    const [lastDone, lastTotal] = seen[seen.length - 1] as [number, number]
    expect(lastDone).toBe(lastTotal)
    for (let i = 1; i < seen.length; i++) {
      expect((seen[i] as [number, number])[0]).toBe((seen[i - 1] as [number, number])[0] + 1)
    }
  })
})
