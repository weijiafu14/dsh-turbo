/**
 * Probabilistic Pivot Tournament (PPT), faithful to the published selection
 * algorithm: a ring pass over a random Hamiltonian circuit (each candidate
 * appears exactly once as A and once as B, cancelling position bias), pivot
 * selection by ring score, a comparison round of every non-pivot against every
 * pivot plus the pivot pairs, and Bradley-Terry preference probabilities
 * accumulated into win totals. Avoids full O(N^2) pairing.
 *
 * @module dsh-turbo/tournament
 */

import type { PairScores } from './verifier.ts'

/** One evaluated ordered pair (`a` was shown as Candidate A). */
export interface EvaluatedPair {
  a: number
  b: number
  scoreA: number
  scoreB: number
}

/** The tournament's full outcome, kept for decision records. */
export interface TournamentOutcome {
  /** Index of the winning candidate. */
  winner: number
  /** Bradley-Terry win accumulation per candidate. */
  wins: number[]
  /** Mean ring-pass score per candidate (NaN when a candidate saw no ring pair). */
  ringScores: number[]
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

/** Bradley-Terry preference probability P(i beats j) from continuous rewards. */
export function preferenceProbability(scoreI: number, scoreJ: number): number {
  return 1 / (1 + Math.exp(-(scoreI - scoreJ)))
}

function shuffled(count: number, random: () => number): number[] {
  const order = Array.from({ length: count }, (_, i) => i)
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1))
    const a = order[i] as number
    order[i] = order[j] as number
    order[j] = a
  }
  return order
}

const pairKey = (a: number, b: number): string => a < b ? `${a}:${b}` : `${b}:${a}`

/**
 * Run the tournament over `count` candidates.
 * All pairs of one phase evaluate concurrently; the ring pass completes before
 * pivots are chosen. With two candidates the ring already decides.
 * @param count - number of candidates (>= 2).
 * @param evaluate - ordered-pair evaluator (position matters to the model).
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
  const pivotCount = Math.max(1, Math.min(options.pivots, count - 1))

  const pairs: EvaluatedPair[] = []
  const evaluatedKeys = new Set<string>()
  let done = 0
  // Planned total: ring pairs + upper bound of the comparison round; refined
  // below once pivots are known (overlap with ring pairs is skipped).
  let total = count === 2 ? 1 : count + pivotCount * (count - pivotCount) + pivotCount * (pivotCount - 1) / 2

  const evaluateOrdered = async (a: number, b: number): Promise<EvaluatedPair> => {
    const scores = await evaluate(a, b)
    const entry: EvaluatedPair = { a, b, scoreA: scores.a, scoreB: scores.b }
    pairs.push(entry)
    evaluatedKeys.add(pairKey(a, b))
    done += 1
    options.onProgress?.(done, total)
    return entry
  }

  // Ring pass: adjacent ordered pairs around a random circuit. With exactly
  // two candidates the "circuit" degenerates to the one pair, evaluated once.
  const circuit = shuffled(count, random)
  const ringPlan: [number, number][] = count === 2
    ? [[circuit[0] as number, circuit[1] as number]]
    : circuit.map((candidate, at) => [candidate, circuit[(at + 1) % count] as number] as [number, number])
  await Promise.all(ringPlan.map(([a, b]) => evaluateOrdered(a, b)))

  const ringTotals = new Array<number>(count).fill(0)
  const ringCounts = new Array<number>(count).fill(0)
  for (const pair of pairs) {
    ringTotals[pair.a] = (ringTotals[pair.a] ?? 0) + pair.scoreA
    ringCounts[pair.a] = (ringCounts[pair.a] ?? 0) + 1
    ringTotals[pair.b] = (ringTotals[pair.b] ?? 0) + pair.scoreB
    ringCounts[pair.b] = (ringCounts[pair.b] ?? 0) + 1
  }
  const ringScores = ringTotals.map((sum, i) => (ringCounts[i] ?? 0) > 0 ? sum / (ringCounts[i] as number) : Number.NaN)

  // Pivot selection by ring score (stable on ties by index).
  const byRing = Array.from({ length: count }, (_, i) => i)
    .sort((x, y) => (ringScores[y] ?? 0) - (ringScores[x] ?? 0) || x - y)
  const pivots = count === 2 ? [byRing[0] as number] : byRing.slice(0, pivotCount)
  const pivotSet = new Set(pivots)

  // Comparison round: non-pivots vs every pivot, plus pivot pairs; pairs the
  // ring already scored are not re-bought. Orientation is randomized so the
  // pivot is not systematically Candidate A.
  if (count > 2) {
    const plan: [number, number][] = []
    for (let candidate = 0; candidate < count; candidate++) {
      if (pivotSet.has(candidate)) continue
      for (const pivot of pivots) {
        if (evaluatedKeys.has(pairKey(candidate, pivot))) continue
        plan.push(random() < 0.5 ? [candidate, pivot] : [pivot, candidate])
        evaluatedKeys.add(pairKey(candidate, pivot))
      }
    }
    for (let x = 0; x < pivots.length; x++) {
      for (let y = x + 1; y < pivots.length; y++) {
        const px = pivots[x] as number
        const py = pivots[y] as number
        if (evaluatedKeys.has(pairKey(px, py))) continue
        plan.push(random() < 0.5 ? [px, py] : [py, px])
        evaluatedKeys.add(pairKey(px, py))
      }
    }
    total = done + plan.length
    await Promise.all(plan.map(([a, b]) => evaluateOrdered(a, b)))
  }

  // Bradley-Terry accumulation over every evaluated pair.
  const wins = new Array<number>(count).fill(0)
  for (const pair of pairs) {
    const p = preferenceProbability(pair.scoreA, pair.scoreB)
    wins[pair.a] = (wins[pair.a] ?? 0) + p
    wins[pair.b] = (wins[pair.b] ?? 0) + (1 - p)
  }

  let winner = 0
  for (let candidate = 1; candidate < count; candidate++) {
    const lead = wins[winner] ?? 0
    const challenger = wins[candidate] ?? 0
    if (challenger > lead) {
      winner = candidate
      continue
    }
    if (challenger === lead) {
      const leadRing = ringScores[winner] ?? Number.NEGATIVE_INFINITY
      const challengerRing = ringScores[candidate] ?? Number.NEGATIVE_INFINITY
      if (challengerRing > leadRing) winner = candidate
    }
  }

  return { winner, wins, ringScores, pivots, pairs }
}
