/**
 * Probabilistic Pivot Tournament (PPT): O(Nk) best-of-N selection, aligned
 * with the reference implementation (llm_verifier/pivot_tournament.py,
 * upstream commit line "Release v0.2.0", 2026-08-14).
 *
 *   1) Ring pass — score the N directed adjacent pairs of a random
 *      Hamiltonian cycle, so every candidate appears once in each prompt slot
 *      and the verifier's slot bias cancels around the ring.
 *   2) Pivot selection — the top-k by ring-pass MEAN preference w/c.
 *   3) Pivot rounds — every non-pivot vs pivot and pivot vs pivot, aggregated
 *      into the same w/c, then argmax of the mean.
 *
 * Selection is by MEAN preference, never by total wins: pivots take part in
 * far more comparisons than anyone else, so totals would reward being a pivot
 * rather than being good.
 *
 * @module dsh-turbo/tournament
 */

import type { PairScores } from './verifier.ts'

/** One evaluated ordered pair (`a` was shown as Candidate A). */
export interface EvaluatedPair {
  a: number
  b: number
  /** Normalized reward in [0, 1] for the candidate in slot A. */
  scoreA: number
  /** Normalized reward in [0, 1] for the candidate in slot B. */
  scoreB: number
  /** Bradley-Terry preference for `a` over `b`. */
  preference: number
  phase: 'ring' | 'pivot'
}

/** The tournament's full outcome, kept for decision records. */
export interface TournamentOutcome {
  /** Index of the winning candidate. */
  winner: number
  /** Mean preference per candidate — the quantity selection uses. */
  meanPreference: number[]
  /** Accumulated soft wins per candidate (diagnostic; never the ranking). */
  wins: number[]
  /** Comparison count per candidate (diagnostic). */
  comparisons: number[]
  /** Mean ring-pass preference per candidate, before the pivot rounds. */
  ringPreference: number[]
  /** Candidate indexes selected as pivots. */
  pivots: number[]
  /** Every evaluated pair in evaluation order. */
  pairs: EvaluatedPair[]
}

/** Evaluate one ordered pair; `a` is presented as Candidate A. */
export type PairEvaluator = (a: number, b: number) => Promise<PairScores>

export interface TournamentOptions {
  /** How many ring leaders anchor the comparison round. */
  pivots: number
  /** Uniform [0,1) source; injectable for deterministic tests. */
  random?: () => number
  /** Progress callback: evaluated pair count over the planned total. */
  onProgress?: (done: number, total: number) => void
}

/**
 * Bradley-Terry preference on rewards in [0, 1].
 * @param scoreI - normalized reward of the candidate in slot A.
 * @param scoreJ - normalized reward of the candidate in slot B.
 * @returns p(i beats j).
 */
export function preferenceProbability(scoreI: number, scoreJ: number): number {
  return 1 / (1 + Math.exp(-(scoreI - scoreJ)))
}

/** The N directed adjacent pairs of a random Hamiltonian cycle. */
export function ringCycle(count: number, random: () => number): [number, number][] {
  if (count <= 1) return []
  const perm = Array.from({ length: count }, (_, i) => i)
  for (let i = perm.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1))
    const swap = perm[i] as number
    perm[i] = perm[j] as number
    perm[j] = swap
  }
  // With two candidates the cycle is the pair in BOTH orientations, which is
  // exactly the slot-bias cancellation the ring exists for.
  return perm.map((candidate, at) => [candidate, perm[(at + 1) % count] as number] as [number, number])
}

/** Directed pairs for the pivot round: non-pivots take slot A, pivot pairs by ascending index. */
export function pivotRoundPairs(count: number, pivots: readonly number[]): [number, number][] {
  const pivotSet = new Set(pivots)
  const pairs: [number, number][] = []
  for (let candidate = 0; candidate < count; candidate++) {
    if (pivotSet.has(candidate)) continue
    for (const pivot of pivots) pairs.push([candidate, pivot])
  }
  const sorted = [...pivots].sort((x, y) => x - y)
  for (let x = 0; x < sorted.length; x++) {
    for (let y = x + 1; y < sorted.length; y++) {
      pairs.push([sorted[x] as number, sorted[y] as number])
    }
  }
  return pairs
}

/**
 * Run the tournament over `count` candidates. Pairs within one phase evaluate
 * concurrently; the ring pass completes before pivots are chosen.
 * @param count - number of candidates (>= 2).
 * @param evaluate - ordered-pair evaluator (slot order matters to the model).
 * @param options - pivot count, RNG, progress.
 * @returns the winner and the full evidence trail.
 */
export async function runTournament(
  count: number,
  evaluate: PairEvaluator,
  options: TournamentOptions,
): Promise<TournamentOutcome> {
  if (count < 2) throw new Error('dsh-turbo: a tournament needs at least two candidates')
  const random = options.random ?? Math.random
  const pivotCount = Math.max(1, Math.min(options.pivots, count))

  const wins = new Array<number>(count).fill(0)
  const comparisons = new Array<number>(count).fill(0)
  const pairs: EvaluatedPair[] = []
  let done = 0

  const ring = ringCycle(count, random)
  // The pivot round's size is known only after the ring picks pivots; the
  // planned total is refined then, and progress never exceeds it.
  let total = ring.length + pivotCount * (count - pivotCount) + pivotCount * (pivotCount - 1) / 2

  const runPhase = async (plan: readonly [number, number][], phase: 'ring' | 'pivot'): Promise<void> => {
    const scored = await Promise.all(plan.map(async ([a, b]) => {
      const result = await evaluate(a, b)
      done += 1
      options.onProgress?.(done, total)
      return { a, b, result }
    }))
    for (const { a, b, result } of scored) {
      const preference = preferenceProbability(result.a, result.b)
      pairs.push({ a, b, scoreA: result.a, scoreB: result.b, preference, phase })
      wins[a] = (wins[a] ?? 0) + preference
      comparisons[a] = (comparisons[a] ?? 0) + 1
      wins[b] = (wins[b] ?? 0) + (1 - preference)
      comparisons[b] = (comparisons[b] ?? 0) + 1
    }
  }

  await runPhase(ring, 'ring')

  const mean = (index: number): number => {
    const seen = comparisons[index] ?? 0
    return seen > 0 ? (wins[index] ?? 0) / seen : 0
  }
  const ringPreference = Array.from({ length: count }, (_, i) => mean(i))
  const pivots = Array.from({ length: count }, (_, i) => i)
    .sort((x, y) => (ringPreference[y] ?? 0) - (ringPreference[x] ?? 0) || x - y)
    .slice(0, Math.min(pivotCount, count))

  const plan = pivotRoundPairs(count, pivots)
  total = done + plan.length
  await runPhase(plan, 'pivot')

  const meanPreference = Array.from({ length: count }, (_, i) => mean(i))
  let winner = 0
  for (let candidate = 1; candidate < count; candidate++) {
    if ((meanPreference[candidate] ?? 0) > (meanPreference[winner] ?? 0)) winner = candidate
  }

  return { winner, meanPreference, wins, comparisons, ringPreference, pivots, pairs }
}
