/**
 * Verifier criteria sets (LLM-as-a-Verifier "criteria decomposition"):
 * each criterion is scored in its own evaluation call and the final reward is
 * the arithmetic mean across criteria. The coding set follows the published
 * Terminal-Bench criteria of the upstream framework
 * (github.com/llm-as-a-verifier/llm-as-a-verifier, criteria/terminal_bench.md):
 * Specification Adherence / Output Match / Error Signal Detection, with
 * observed output as ground truth over the agent's self-assessment.
 *
 * @module dsh-turbo/criteria
 */

/** One decomposed evaluation criterion. */
export interface Criterion {
  /** Stable key for logs and decision records. */
  key: string
  /** Display name shown to the verifier model. */
  name: string
  /** The single question this evaluation answers. */
  question: string
}

/** Criteria for agentic coding / terminal work (the default). */
export const CODING_CRITERIA: readonly Criterion[] = [
  {
    key: 'specification',
    name: 'Specification Adherence',
    question: 'Does the candidate meet the EXACT task requirements — file paths, locations, '
      + 'output formats, naming conventions, and every explicit constraint? A candidate that '
      + 'solves a similar problem in the wrong place or wrong format fails this criterion.',
  },
  {
    key: 'output',
    name: 'Output Match',
    question: 'Judge by observable results, not claims: do the concrete outputs (tool results, '
      + 'terminal output, produced content) literally match what the task requires? Observed '
      + 'output is the ground truth and supersedes the candidate\'s own assessment of success.',
  },
  {
    key: 'errors',
    name: 'Error Signal Detection',
    question: 'Scan for unresolved failure indicators: error messages, exceptions, '
      + '"command not found", non-zero exits, compilation failures, contradictions, or actions '
      + 'that would clearly fail. A clean candidate whose steps would succeed scores high; one '
      + 'carrying unresolved errors is broken regardless of claimed success.',
  },
]

/** Criteria for general conversation / analysis requests. */
export const GENERAL_CRITERIA: readonly Criterion[] = [
  {
    key: 'instruction',
    name: 'Instruction Adherence',
    question: 'Does the candidate do exactly what the user asked — scope, format, language, and '
      + 'every explicit constraint — without ignoring parts of the request or adding unrequested work?',
  },
  {
    key: 'correctness',
    name: 'Correctness',
    question: 'Is the content factually and logically correct? Penalize fabrications, internal '
      + 'contradictions, and reasoning errors; reward verifiable, grounded claims.',
  },
  {
    key: 'quality',
    name: 'Clarity and Usefulness',
    question: 'Is the candidate clear, well-organized, and directly useful to the user\'s goal, '
      + 'at an appropriate level of detail?',
  },
]

/** Resolve a configured criteria set name. */
export function criteriaSet(name: 'coding' | 'general'): readonly Criterion[] {
  return name === 'general' ? GENERAL_CRITERIA : CODING_CRITERIA
}
