/**
 * Verifier criteria sets (LLM-as-a-Verifier "criteria decomposition"): each
 * criterion is scored in its own call and the rewards are averaged. Aligned
 * with the reference implementation's criteria files, which pair the criteria
 * with a ground-truth note the verifier sees on every call — the note carries
 * the standing rule for judging evidence, and the criteria stay specific.
 *
 * The coding set follows the published Terminal-Bench criteria of the upstream
 * framework (github.com/llm-as-a-verifier/llm-as-a-verifier,
 * criteria/terminal_bench.md): specification adherence, output match, and
 * error-signal detection, with observed output outranking self-assessment.
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

/** A criteria set: the standing note plus the criteria scored one by one. */
export interface CriteriaSet {
  /** Paragraph shown to the verifier on every call, before the criterion. */
  groundTruthNote: string
  criteria: readonly Criterion[]
}

const CODING_NOTE = 'Judge only by evidence visible in the transcript. Observed '
  + 'results — tool results, command output, produced files — are the ground truth, '
  + 'and they outrank any claim a candidate makes about its own success. A candidate '
  + 'that asserts it finished without evidence is worth less than one whose output '
  + 'shows the work. Absent or truncated evidence is not credit.'

/** Criteria for agentic coding / terminal work (the default). */
export const CODING_CRITERIA: CriteriaSet = {
  groundTruthNote: CODING_NOTE,
  criteria: [
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
      question: 'Do the concrete outputs (tool results, terminal output, produced content) '
        + 'literally match what the task requires, character for character where the task says so?',
    },
    {
      key: 'errors',
      name: 'Error Signal Detection',
      question: 'Scan for unresolved failure indicators: error messages, exceptions, '
        + '"command not found", non-zero exits, compilation failures, contradictions, or actions '
        + 'that would clearly fail. A clean candidate whose steps would succeed scores high; one '
        + 'carrying unresolved errors is broken regardless of claimed success.',
    },
  ],
}

const GENERAL_NOTE = 'Judge only by what the candidate actually produced. A confident '
  + 'tone is not evidence, and unrequested extra work is not credit. Where the request is '
  + 'specific, the specific requirement outranks general quality.'

/** Criteria for general conversation / analysis requests. */
export const GENERAL_CRITERIA: CriteriaSet = {
  groundTruthNote: GENERAL_NOTE,
  criteria: [
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
  ],
}

/** Resolve a configured criteria set name. */
export function criteriaSet(name: 'coding' | 'general'): CriteriaSet {
  return name === 'general' ? GENERAL_CRITERIA : CODING_CRITERIA
}
