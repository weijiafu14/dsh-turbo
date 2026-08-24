/**
 * Orchestration contract tests: the adapter is exercised against a stubbed
 * fetch that speaks the real DeepSeek wire shapes (SSE for candidates, JSON
 * with logprobs for verifier calls). Nothing here claims end-to-end proof —
 * that runs against the live API — but the fan-out/verify/replay contract is
 * pinned: chunk ordering, winner selection, usage accounting, degradation.
 */

import { afterEach, describe, expect, it, vi } from 'vitest'
import { resolveRetryPolicy } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import { TurboAdapter } from '../src/adapter.ts'
import type { TurboConnectionOptions, TurboDecision } from '../src/adapter.ts'
import type { WireLogprobEntry, WireRequest } from '../src/types.ts'

function seeded(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0
    return state / 2 ** 32
  }
}

const connection = (overrides: Partial<TurboConnectionOptions> = {}): TurboConnectionOptions => ({
  baseURL: 'https://fake.test',
  apiKeyEnv: credentialRef('DEEPSEEK_API_KEY'),
  defaults: {},
  maxTokens: 1000,
  defaultContextWindow: 100_000,
  models: [{ id: 'deepseek-v4-flash-vision-exp', vision: true }],
  streamIdleTimeoutMs: 5_000,
  retryPolicy: resolveRetryPolicy(undefined, 'test'),
  n: 3,
  repeats: 1,
  pivots: 1,
  temperatures: [0.2, 0.6, 1.0],
  criteria: 'coding',
  verifierTemperature: 1,
  verifierMaxTokens: 500,
  ...overrides,
})

const request = (): GenerateOptions => ({
  provider: 'deepseek-turbo',
  model: 'deepseek-v4-flash-vision-exp',
  messages: [{ role: 'user', content: [{ type: 'text', text: 'solve it' }] } as never],
})

function sseResponse(text: string, outputTokens: number): Response {
  const payloads = [
    JSON.stringify({ choices: [{ delta: { content: text } }] }),
    JSON.stringify({
      choices: [{ delta: {}, finish_reason: 'stop' }],
      usage: { prompt_tokens: 10, completion_tokens: outputTokens },
    }),
    '[DONE]',
  ]
  const body = payloads.map(payload => `data: ${payload}\n\n`).join('')
  return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
}

/** Tokenize verifier content keeping digit runs whole, so offsets line up. */
function verifierEntries(content: string, scores: Record<string, number>): WireLogprobEntry[] {
  return (content.match(/\d+|\D+/g) ?? []).map((token) => {
    const p = scores[token]
    return {
      token,
      logprob: p === undefined ? -0.01 : Math.log(p),
      ...p === undefined ? {} : { top_logprobs: [{ token, logprob: Math.log(p) }] },
    }
  })
}

function verifierResponse(scoreA: number, scoreB: number): Response {
  const content = `analysis.\n<score_A>${scoreA}</score_A>\n<score_B>${scoreB}</score_B>`
  return new Response(JSON.stringify({
    choices: [{
      message: { content },
      logprobs: { content: verifierEntries(content, { [String(scoreA)]: 0.9, [String(scoreB)]: 0.9 }) },
      finish_reason: 'stop',
    }],
    usage: { prompt_tokens: 50, completion_tokens: 20, prompt_tokens_details: { cached_tokens: 30 } },
  }), { status: 200, headers: { 'content-type': 'application/json' } })
}

/** Stub fetch speaking the wire: candidates by temperature, verifier by content. */
function stubWire(byTemperature: Record<string, Response | (() => Response)>): { calls: WireRequest[] } {
  const calls: WireRequest[] = []
  vi.stubGlobal('fetch', vi.fn(async (_url: unknown, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body)) as WireRequest
    calls.push(body)
    if (body.stream) {
      const key = String(body.temperature)
      const entry = byTemperature[key]
      if (entry === undefined) throw new Error(`unexpected candidate temperature ${key}`)
      return typeof entry === 'function' ? entry() : entry.clone()
    }
    // Verifier call: the GOOD candidate always wins, whichever side it is on.
    const pairMessage = body.messages
      .map(message => (message as { content: unknown }).content)
      .filter((content): content is string => typeof content === 'string')
      .find(content => content.includes('=== Candidate A ==='))
    if (pairMessage === undefined) throw new Error('verifier request carries no candidate pair message')
    const aStart = pairMessage.indexOf('=== Candidate A ===')
    const bStart = pairMessage.indexOf('=== Candidate B ===')
    const aText = pairMessage.slice(aStart, bStart)
    const goodOnA = aText.includes('GOOD')
    return goodOnA ? verifierResponse(17, 4) : verifierResponse(4, 17)
  }))
  return { calls }
}

const collect = async (chunks: AsyncIterable<StreamChunk>): Promise<StreamChunk[]> => {
  const all: StreamChunk[] = []
  for await (const chunk of chunks) all.push(chunk)
  return all
}

const finalText = (chunks: StreamChunk[]): string =>
  chunks
    .filter(chunk => chunk.type === 'block-end')
    .map(chunk => chunk.block)
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('')

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('TurboAdapter fan-out', () => {
  const build = (conn: TurboConnectionOptions) => {
    const decisions: TurboDecision[] = []
    const adapter = new TurboAdapter({
      options: () => conn,
      resolveApiKey: () => Promise.resolve('test-key'),
      resolveImages: () => Promise.resolve(new Map()),
      record: decision => decisions.push(decision),
      random: seeded(42),
    })
    return { adapter, decisions }
  }

  it('samples N, verifies, and replays exactly the winning candidate', async () => {
    const { calls } = stubWire({
      '0.2': () => sseResponse('BAD-ONE', 11),
      '0.6': () => sseResponse('GOOD answer', 22),
      '1': () => sseResponse('BAD-TWO', 33),
    })
    const { adapter, decisions } = build(connection())
    const chunks = await collect(adapter.stream(request()))

    expect(finalText(chunks)).toBe('GOOD answer')
    // Winner's own usage replays for the loop's context accounting.
    const usage = chunks.find(chunk => chunk.type === 'usage')
    expect(usage).toMatchObject({ usage: { outputTokens: 22 } })
    // usage precedes finish, and finish is terminal.
    const kinds = chunks.map(chunk => chunk.type)
    expect(kinds.indexOf('usage')).toBeLessThan(kinds.indexOf('finish'))
    expect(kinds[kinds.length - 1]).toBe('finish')

    // 3 candidate calls + at least one verifier call went over the wire.
    expect(calls.filter(call => call.stream).length).toBe(3)
    const verifierCalls = calls.filter(call => !call.stream)
    expect(verifierCalls.length).toBeGreaterThan(0)
    // Verifier calls carry the paper's signal switches.
    for (const call of verifierCalls) {
      expect(call.logprobs).toBe(true)
      expect(call.top_logprobs).toBe(20)
      expect(call.thinking).toEqual({ type: 'disabled' })
    }

    const decision = decisions[0]!
    expect(decision.survivors).toBe(3)
    expect(decision.degraded).toBe(false)
    expect(decision.candidates[decision.winner]!.preview).toContain('GOOD')
    expect(decision.neutralEvaluations).toBe(0)
    // Selection is by mean preference, and the winner must lead on it.
    const mean = decision.outcome!.meanPreference
    for (const [i, value] of mean.entries()) {
      if (i !== decision.winner) expect(mean[decision.winner]!).toBeGreaterThan(value)
    }
    // Losing candidates' spend lands in extraUsage (their 11 + 33 output tokens).
    expect(decision.extraUsage.outputTokens).toBeGreaterThanOrEqual(44)
    expect(decision.extraUsage.calls).toBeGreaterThanOrEqual(2)
  })

  it('tolerates a failed candidate and still verifies the survivors', async () => {
    const empty = () => new Response('data: [DONE]\n\n', { status: 200 })
    const { calls } = stubWire({
      '0.2': empty, // EMPTY_RESPONSE error finish -> failed candidate
      '0.6': () => sseResponse('GOOD answer', 22),
      '1': () => sseResponse('meh', 5),
    })
    const { adapter, decisions } = build(connection())
    const chunks = await collect(adapter.stream(request()))

    expect(finalText(chunks)).toBe('GOOD answer')
    expect(decisions[0]!.survivors).toBe(2)
    expect(decisions[0]!.failed).toBe(1)
    expect(calls.filter(call => !call.stream).length).toBeGreaterThan(0)
  })

  it('throws when every candidate fails', async () => {
    const broken = () => new Response('data: [DONE]\n\n', { status: 200 })
    stubWire({ '0.2': broken, '0.6': broken, '1': broken })
    const { adapter } = build(connection())
    await expect(collect(adapter.stream(request()))).rejects.toThrow(/candidate/i)
  })

  it('degrades to the first survivor when verification breaks, and says so', async () => {
    vi.stubGlobal('fetch', vi.fn(async (_url: unknown, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as WireRequest
      if (body.stream) return sseResponse(`answer-at-${body.temperature}`, 7)
      return new Response('{"error":{"message":"verifier down"}}', { status: 500 })
    }))
    const { adapter, decisions } = build(connection())
    const chunks = await collect(adapter.stream(request()))
    expect(finalText(chunks)).toMatch(/^answer-at-/)
    // Every evaluation fell back to neutral, so the ranking carried no
    // information and the run must say so rather than dress a tie-break up as
    // a verdict.
    expect(decisions[0]!.degraded).toBe(true)
    expect(decisions[0]!.usableEvaluations).toBe(0)
    expect(decisions[0]!.neutralEvaluations).toBeGreaterThan(0)
  })

  it('passes auxiliary calls straight through with a single request', async () => {
    const { calls } = stubWire({ undefined: () => sseResponse('Title', 3) })
    const { adapter } = build(connection())
    const chunks = await collect(adapter.stream({ ...request(), purpose: 'session-title' }))
    expect(finalText(chunks)).toBe('Title')
    expect(calls).toHaveLength(1)
    expect(calls[0]!.thinking).toEqual({ type: 'disabled' })
  })

  it('passes through when the runtime kill switch is off', async () => {
    const { calls } = stubWire({ undefined: () => sseResponse('single', 3) })
    const adapter = new TurboAdapter({
      options: () => connection(),
      resolveApiKey: () => Promise.resolve('k'),
      resolveImages: () => Promise.resolve(new Map()),
      bypass: () => true,
    })
    const chunks = await collect(adapter.stream(request()))
    expect(finalText(chunks)).toBe('single')
    expect(calls).toHaveLength(1)
  })
})

describe('TurboAdapter catalog', () => {
  it('declares image input for vision models and text-only otherwise', async () => {
    const adapter = new TurboAdapter({
      options: () => connection({
        models: [
          { id: 'deepseek-v4-flash-vision-exp', vision: true },
          { id: 'deepseek-v4-flash' },
        ],
      }),
      resolveApiKey: () => Promise.resolve('k'),
      resolveImages: () => Promise.resolve(new Map()),
    })
    const vision = await adapter.resolveModel('deepseek-turbo', 'deepseek-v4-flash-vision-exp')
    expect(vision.inputModalities).toEqual(['text', 'image'])
    const text = await adapter.resolveModel('deepseek-turbo', 'deepseek-v4-flash')
    expect(text.inputModalities).toEqual(['text'])
    // Uncatalogued models must declare NEGATIVE image capability.
    const unknown = await adapter.resolveModel('deepseek-turbo', 'mystery')
    expect(unknown.inputModalities).toEqual(['text'])
  })
})
