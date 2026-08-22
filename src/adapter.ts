/**
 * `TurboAdapter`: the best-of-N provider route. One harness model call fans
 * out into N parallel candidate generations against DeepSeek, the
 * LLM-as-a-Verifier tournament picks the winner, and the winner's buffered
 * StreamChunks replay to the loop — which sees exactly one ordinary model
 * call. Auxiliary calls (compaction, session titles) and n<=1 pass straight
 * through as a live stream. Watchdog/error normalization adapted from
 * @deepseek-ai/dsh-llm-deepseek `src/adapter.ts` (0.1.1-rc.2).
 *
 * @module dsh-turbo/adapter
 */

import { LlmAdapter, LlmError, ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import type {
  ContentBlock,
  FinishReason,
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  ResolvedRetryPolicy,
  StreamChunk,
  TokenUsage,
} from '@deepseek-ai/dsh-llm'
import type { CredentialRef } from '@deepseek-ai/dsh-credentials'
import { idleWatchdog, timeoutOf } from '@deepseek-ai/dsh-timeout'
import { criteriaSet } from './criteria.ts'
import { renderCandidate, renderContext } from './render.ts'
import { serializeRequest } from './serialize.ts'
import type { ImageDataIndex, RequestDefaults } from './serialize.ts'
import { runTournament } from './tournament.ts'
import type { TournamentOutcome } from './tournament.ts'
import { translate } from './translate.ts'
import { streamCompletion } from './transport.ts'
import { scorePair } from './verifier.ts'
import type { WireRequest } from './types.ts'

/** One model entry advertised by the turbo route. */
export interface TurboCatalogModel {
  id: string
  name?: string
  description?: string
  contextWindow?: number
  maxTokens?: number
  /** Whether the wire model accepts image input. */
  vision?: boolean
}

/** Validated connection + verification facts for one operation. */
export interface TurboConnectionOptions {
  baseURL: string
  apiKeyEnv: CredentialRef
  defaults: RequestDefaults
  maxTokens: number
  defaultContextWindow: number
  models: readonly TurboCatalogModel[]
  streamIdleTimeoutMs: number
  retryPolicy: ResolvedRetryPolicy
  /** Candidates per model call; n<=1 is plain passthrough. */
  n: number
  /** K: independent verifier calls per pair and criterion. */
  repeats: number
  /** Ring leaders anchoring the comparison round. */
  pivots: number
  /** Candidate temperature ladder; generated evenly over [0.2, 1.2] when omitted. */
  temperatures?: readonly number[]
  /** Which decomposed criteria set the verifier applies. */
  criteria: 'coding' | 'general'
  /** Verifier model; defaults to the candidate model of the request. */
  verifierModel?: string
  verifierTemperature: number
  verifierMaxTokens: number
}

/** Aggregate token cost of everything the winner's usage chunk does NOT cover. */
export interface TurboExtraUsage {
  inputTokens: number
  outputTokens: number
  cacheReadTokens: number
  calls: number
}

/** One recorded fan-out decision, for the /turbo command and the decision log. */
export interface TurboDecision {
  at: string
  model: string
  verifierModel: string
  requested: number
  survivors: number
  failed: number
  winner: number
  /** Whether verification failed and the first survivor was used instead. */
  degraded: boolean
  outcome?: TournamentOutcome
  candidates: {
    temperature: number | undefined
    finish: string
    outputTokens: number | undefined
    preview: string
  }[]
  extraUsage: TurboExtraUsage
  durationMs: number
}

/** Operation-local hooks the registering plugin owns. */
export interface TurboAdapterOptions {
  /** Current validated facts; called once per operation. */
  options: () => TurboConnectionOptions
  /** Resolve the bearer token for one request's facts. */
  resolveApiKey: (connection: TurboConnectionOptions) => Promise<string>
  /**
   * Resolve every image attachment of the request to a base64 data URL.
   * Throws `LlmError('UNSUPPORTED_CONTENT')` when images are present but no
   * attachment store is available.
   */
  resolveImages: (options: GenerateOptions, signal: AbortSignal | undefined) => Promise<ImageDataIndex>
  /** Runtime kill switch (the /turbo off command); true = pass through. */
  bypass?: () => boolean
  /** Status-line reporter; called with undefined to clear. */
  report?: (text: string | undefined) => void
  /** Decision sink (ring buffer + JSONL). */
  record?: (decision: TurboDecision) => void
  /** Uniform [0,1) source; injectable for deterministic tests. */
  random?: () => number
}

const STREAM_IDLE_TIMEOUT_CODE = 'LLM_STREAM_IDLE_TIMEOUT'
const OFF_REASONING_EFFORT = ReasoningEffortId('off')
const HIGH_REASONING_EFFORT = ReasoningEffortId('high')
const MAX_REASONING_EFFORT = ReasoningEffortId('max')
const REASONING_EFFORTS = [
  { id: OFF_REASONING_EFFORT, name: 'Off' },
  { id: HIGH_REASONING_EFFORT, name: 'High' },
  { id: MAX_REASONING_EFFORT, name: 'Max' },
] as const

/** Evenly spaced candidate temperatures over [0.2, 1.2]. */
export function temperatureLadder(n: number): number[] {
  if (n <= 1) return [1]
  return Array.from({ length: n }, (_, i) => Number((0.2 + i * (1 / (n - 1))).toFixed(3)))
}

function modelInfo(provider: string, model: TurboCatalogModel): LlmModelInfo {
  return {
    provider,
    id: model.id,
    name: model.name ?? model.id,
    ...model.description === undefined ? {} : { description: model.description },
    inputModalities: model.vision === true ? ['text', 'image'] : ['text'],
  }
}

/** One buffered candidate generation. */
interface Candidate {
  temperature: number | undefined
  chunks: StreamChunk[]
  blocks: ContentBlock[]
  usage: TokenUsage | undefined
  finish: FinishReason | undefined
  error: unknown
}

function candidateFailed(candidate: Candidate): boolean {
  if (candidate.error !== undefined) return true
  const kind = candidate.finish?.kind
  return kind !== 'stop' && kind !== 'tool-calls' && kind !== 'max-tokens'
}

export class TurboAdapter extends LlmAdapter {
  constructor(private readonly config: TurboAdapterOptions) {
    super()
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: 'DeepSeek Turbo (best-of-N)' }
  }

  override providerRetryPolicy(_provider: string): ResolvedRetryPolicy {
    return this.config.options().retryPolicy
  }

  override listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    return Promise.resolve(this.config.options().models.map(model => modelInfo(provider, model)))
  }

  override resolveModel(
    provider: string,
    model: string,
    _signal?: AbortSignal,
  ): Promise<LlmResolvedModelInfo> {
    const connection = this.config.options()
    const configured = connection.models.find(entry => entry.id === model)
    const contextWindow = configured?.contextWindow ?? connection.defaultContextWindow
    return Promise.resolve({
      // Uncatalogued models declare text-only: "unknown" would let the host
      // accept images the wire model may then reject.
      ...configured === undefined
        ? { provider, id: model, name: model, inputModalities: ['text' as const] }
        : modelInfo(provider, configured),
      context: { contextWindow },
      defaultMaxTokens: configured?.maxTokens ?? connection.maxTokens,
      reasoning: {
        efforts: REASONING_EFFORTS,
        defaultEffort: connection.defaults.reasoningEffort === 'off'
          ? OFF_REASONING_EFFORT
          : connection.defaults.reasoningEffort === 'max'
            ? MAX_REASONING_EFFORT
            : HIGH_REASONING_EFFORT,
      },
    })
  }

  async* stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    // One resolution per stream call: facts and credential freeze here so an
    // in-flight fan-out never observes a configuration change.
    const connection = this.config.options()
    const apiKey = await this.config.resolveApiKey(connection)
    const images = await this.config.resolveImages(options, options.signal)

    const passthrough = options.purpose !== undefined
      || connection.n <= 1
      || this.config.bypass?.() === true
    if (passthrough) {
      yield* this.passthrough(options, connection, apiKey, images)
      return
    }
    yield* this.fanOut(options, connection, apiKey, images)
  }

  /** Live single-candidate streaming (auxiliary calls, n<=1, /turbo off). */
  private async* passthrough(
    options: GenerateOptions,
    connection: TurboConnectionOptions,
    apiKey: string,
    images: ImageDataIndex,
  ): AsyncIterable<StreamChunk> {
    const body = serializeRequest(options, connection.defaults, images)
    const consumer = new AbortController()
    const upstream = options.signal === undefined
      ? consumer.signal
      : AbortSignal.any([options.signal, consumer.signal])
    using watchdog = idleWatchdog(upstream, connection.streamIdleTimeoutMs, STREAM_IDLE_TIMEOUT_CODE)
    const iterator = translate(
      streamCompletion(body, connection, apiKey, watchdog.signal, () => { watchdog.pulse() }),
    )[Symbol.asyncIterator]()
    let exhausted = false
    try {
      while (true) {
        const result = await watchdog.next(iterator)
        if (result.done) {
          exhausted = true
          return
        }
        yield result.value
      }
    } catch (error: unknown) {
      throw this.normalizeStreamError(error, watchdog.signal, options, connection)
    } finally {
      consumer.abort('dsh-turbo stream consumer stopped')
      if (!exhausted && iterator.return !== undefined) {
        try {
          await iterator.return(undefined)
        } catch (_teardown) {
          // The consumer controller already owns termination.
        }
      }
    }
  }

  private normalizeStreamError(
    error: unknown,
    watchdogSignal: AbortSignal,
    options: GenerateOptions,
    connection: TurboConnectionOptions,
  ): LlmError {
    if (timeoutOf(watchdogSignal, STREAM_IDLE_TIMEOUT_CODE) !== undefined) {
      return new LlmError(
        `DeepSeek stream idle timeout after ${connection.streamIdleTimeoutMs}ms`,
        'TIMEOUT',
        { cause: error },
      )
    }
    if (options.signal?.aborted) {
      return new LlmError('dsh-turbo request aborted by caller', 'ABORTED', { cause: error })
    }
    if (error instanceof LlmError) return error
    return new LlmError(`DeepSeek API stream from ${connection.baseURL} failed`, 'TRANSPORT', { cause: error })
  }

  /** One buffered candidate: collect every chunk, never surface mid-stream. */
  private async runCandidate(
    body: WireRequest,
    temperature: number | undefined,
    connection: TurboConnectionOptions,
    apiKey: string,
    upstream: AbortSignal,
  ): Promise<Candidate> {
    const candidate: Candidate = {
      temperature,
      chunks: [],
      blocks: [],
      usage: undefined,
      finish: undefined,
      error: undefined,
    }
    using watchdog = idleWatchdog(upstream, connection.streamIdleTimeoutMs, STREAM_IDLE_TIMEOUT_CODE)
    const iterator = translate(
      streamCompletion(body, connection, apiKey, watchdog.signal, () => { watchdog.pulse() }),
    )[Symbol.asyncIterator]()
    try {
      while (true) {
        const result = await watchdog.next(iterator)
        if (result.done) break
        const chunk = result.value
        candidate.chunks.push(chunk)
        if (chunk.type === 'block-end') candidate.blocks.push(chunk.block)
        if (chunk.type === 'usage') candidate.usage = chunk.usage
        if (chunk.type === 'finish') candidate.finish = chunk.reason
      }
    } catch (error: unknown) {
      candidate.error = error
      if (upstream.aborted) throw error
      try {
        await iterator.return?.(undefined)
      } catch (_teardown) {
        // Failed candidates release their transport quietly.
      }
    }
    return candidate
  }

  /** The best-of-N path: N candidates, tournament verification, winner replay. */
  private async* fanOut(
    options: GenerateOptions,
    connection: TurboConnectionOptions,
    apiKey: string,
    images: ImageDataIndex,
  ): AsyncIterable<StreamChunk> {
    const startedAt = Date.now()
    const report = this.config.report ?? (() => {})
    const n = connection.n
    const ladder = connection.temperatures !== undefined && connection.temperatures.length > 0
      ? Array.from({ length: n }, (_, i) => connection.temperatures![i % connection.temperatures!.length] as number)
      : temperatureLadder(n)

    try {
      report(`turbo: sampling 0/${n} candidates`)
      let sampled = 0
      const candidates = await Promise.all(ladder.map(async (temperature) => {
        const body = serializeRequest(options, connection.defaults, images, temperature)
        const candidate = await this.runCandidate(body, temperature, connection, apiKey, options.signal ?? new AbortController().signal)
        sampled += 1
        report(`turbo: sampling ${sampled}/${n} candidates`)
        return candidate
      }))
      if (options.signal?.aborted) {
        throw new LlmError('dsh-turbo request aborted by caller', 'ABORTED')
      }

      const survivors = candidates.filter(candidate => !candidateFailed(candidate))
      const failed = candidates.length - survivors.length
      if (survivors.length === 0) {
        const first = candidates.find(candidate => candidate.error !== undefined)
        const firstFinish = candidates.find(candidate => candidate.finish?.kind === 'error')?.finish
        if (first?.error instanceof LlmError) throw first.error
        if (firstFinish?.kind === 'error') {
          throw new LlmError(`dsh-turbo: every candidate failed: ${firstFinish.failure.message}`, firstFinish.failure.code)
        }
        throw new LlmError('dsh-turbo: every candidate generation failed', 'ALL_CANDIDATES_FAILED', {
          ...first?.error !== undefined ? { cause: first.error } : {},
        })
      }

      const extraUsage: TurboExtraUsage = { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, calls: 0 }
      const verifierModel = connection.verifierModel ?? options.model
      let winnerAt = 0
      let outcome: TournamentOutcome | undefined
      let degraded = false

      if (survivors.length > 1) {
        const context = renderContext(options, images)
        const rendered = survivors.map(candidate => renderCandidate(candidate.blocks))
        const criteria = criteriaSet(connection.criteria)
        const verifierSignal = options.signal ?? new AbortController().signal
        try {
          outcome = await runTournament(
            survivors.length,
            (a, b) => scorePair(context, rendered[a] as string, rendered[b] as string, criteria, connection.repeats, {
              connection,
              apiKey,
              model: verifierModel,
              temperature: connection.verifierTemperature,
              maxTokens: connection.verifierMaxTokens,
              signal: verifierSignal,
              onUsage: (input, output, cacheRead) => {
                extraUsage.inputTokens += input
                extraUsage.outputTokens += output
                extraUsage.cacheReadTokens += cacheRead
                extraUsage.calls += 1
              },
            }),
            {
              pivots: connection.pivots,
              ...this.config.random === undefined ? {} : { random: this.config.random },
              onProgress: (done, total) => { report(`turbo: verifying ${done}/${total} pairs`) },
            },
          )
          winnerAt = outcome.winner
        } catch (error: unknown) {
          if (options.signal?.aborted) throw error
          // Verification breaking must not waste N good candidates: degrade
          // to the first survivor and record the degradation honestly.
          degraded = true
          winnerAt = 0
        }
      }

      const winner = survivors[winnerAt] as Candidate
      // Losing candidates and verifier calls are real spend the winner's
      // usage chunk does not cover; they go to the decision record, while the
      // replayed usage keeps the loop's context accounting correct.
      for (const candidate of candidates) {
        if (candidate === winner || candidate.usage === undefined) continue
        extraUsage.inputTokens += candidate.usage.inputTokens
        extraUsage.outputTokens += candidate.usage.outputTokens
        extraUsage.cacheReadTokens += candidate.usage.cacheReadTokens ?? 0
        extraUsage.calls += 1
      }

      this.config.record?.({
        at: new Date().toISOString(),
        model: options.model,
        verifierModel,
        requested: n,
        survivors: survivors.length,
        failed,
        winner: winnerAt,
        degraded,
        ...outcome === undefined ? {} : { outcome },
        candidates: survivors.map(candidate => ({
          temperature: candidate.temperature,
          finish: candidate.finish?.kind ?? 'unknown',
          outputTokens: candidate.usage?.outputTokens,
          preview: renderCandidate(candidate.blocks).slice(0, 200),
        })),
        extraUsage,
        durationMs: Date.now() - startedAt,
      })
      const winnerWins = outcome === undefined ? undefined : outcome.wins[winnerAt]
      report(
        degraded
          ? 'turbo: verification failed, used first candidate'
          : survivors.length > 1
            ? `turbo: winner #${winnerAt + 1}/${survivors.length} (BT ${winnerWins === undefined ? '?' : winnerWins.toFixed(2)})`
            : 'turbo: single surviving candidate',
      )

      yield* winner.chunks
    } finally {
      report(undefined)
    }
  }
}
