/**
 * dsh-turbo: best-of-N sampling with an LLM-as-a-Verifier judge, as one
 * ordinary DSH provider route. Install the plugin, have DEEPSEEK_API_KEY set
 * (the same credential the official DeepSeek route uses), pick the
 * `deepseek-turbo` provider in the model selector — every model call then
 * samples N candidates, verifies them with fine-grained logprob scoring, and
 * the loop receives the tournament winner.
 *
 * Registration/config layering follows @deepseek-ai/dsh-llm-deepseek
 * `src/index.ts` (0.1.1-rc.2): entry config layers under the optional
 * `dsh-turbo` settings section, the API key resolves through the credential
 * seam per request, and only the registration-captured retry policy forces an
 * in-place route replace.
 *
 * @module dsh-turbo
 */

import { appendFile, mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { assertUsableApiKey, LlmError, resolveRetryPolicy, RetryPolicySchema } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, RetryPolicyConfig } from '@deepseek-ai/dsh-llm'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import { launchEnvironmentOf } from '@deepseek-ai/dsh-launch-environment'
import type { LaunchEnvironmentSnapshot } from '@deepseek-ai/dsh-launch-environment'
import { deepEqualJson, installSettingsSection, settingsNamespace } from '@deepseek-ai/dsh-settings'
import { MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout'
import { TurboAdapter } from './adapter.ts'
import type { TurboCatalogModel, TurboConnectionOptions, TurboDecision } from './adapter.ts'
import type { ImageDataIndex } from './serialize.ts'

export { TurboAdapter, temperatureLadder } from './adapter.ts'
export type { TurboAdapterOptions, TurboCatalogModel, TurboConnectionOptions, TurboDecision } from './adapter.ts'
export { CODING_CRITERIA, GENERAL_CRITERIA, criteriaSet } from './criteria.ts'
export type { Criterion } from './criteria.ts'
export { renderCandidate, renderContext } from './render.ts'
export { serializeMessages, serializeRequest } from './serialize.ts'
export type { ImageDataIndex, RequestDefaults } from './serialize.ts'
export { preferenceProbability, runTournament } from './tournament.ts'
export type { EvaluatedPair, TournamentOutcome } from './tournament.ts'
export { mapFinishReason, mapUsage, translate } from './translate.ts'
export { buildPairMessages, extractPairScores, scoreExpectation, scorePair, SCORE_MAX, SCORE_MIN } from './verifier.ts'
export type * from './types.ts'

export const name = 'dsh-turbo'
export const inject = ['llm']

const NS = settingsNamespace('dsh-turbo')
const DEFAULT_API_KEY_ENV = 'DEEPSEEK_API_KEY'
/** The single provider route this plugin owns. */
const PROVIDER = 'deepseek-turbo'
/** Public API default; a trusted-layer $DEEPSEEK_BASE_URL overrides it. */
export const PUBLIC_BASE_URL = 'https://api.deepseek.com'
const BASE_URL_ENV = 'DEEPSEEK_BASE_URL'
const DEFAULT_CONTEXT_WINDOW = 1_000_000
const DEFAULT_MAX_TOKENS = 256_000
const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000

const DEFAULT_MODELS: TurboCatalogModel[] = [
  {
    id: 'deepseek-v4-flash-vision-exp',
    name: 'DeepSeek-V4-Flash-Vision-Exp (Turbo)',
    description: 'Multimodal, best-of-N verified',
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    vision: true,
  },
  {
    id: 'deepseek-v4-flash',
    name: 'DeepSeek-V4-Flash (Turbo)',
    description: 'Best-of-N verified',
    contextWindow: DEFAULT_CONTEXT_WINDOW,
  },
  {
    id: 'deepseek-v4-pro',
    name: 'DeepSeek-V4-Pro (Turbo)',
    description: 'Best-of-N verified',
    contextWindow: DEFAULT_CONTEXT_WINDOW,
  },
]

/**
 * Plugin config, doubling as the `dsh-turbo` settings-section shape. Every
 * field is optional: with nothing configured the route works as best-of-5 on
 * the official endpoint with `DEEPSEEK_API_KEY`.
 */
export interface Config {
  /** Credential reference (environment-variable name); defaults to `DEEPSEEK_API_KEY`. */
  apiKeyEnv?: string
  /** Endpoint base; any OpenAI-compatible server that returns logprobs works (vLLM etc.). */
  baseURL?: string
  /** Candidate thinking policy passthrough. */
  thinking?: 'enabled' | 'disabled'
  reasoningEffort?: 'off' | 'high' | 'max'
  maxTokens?: number
  defaultContextWindow?: number
  models?: TurboCatalogModel[]
  streamIdleTimeoutMs?: number
  retryPolicy?: RetryPolicyConfig
  /** Candidates per model call (default 5); 1 = plain passthrough. */
  n?: number
  /** K: verifier repeats per pair and criterion (default 1 — logprob scoring already smooths). */
  repeats?: number
  /** Ring leaders anchoring the comparison round (default 2). */
  pivots?: number
  /** Candidate temperature ladder; defaults to an even spread over [0.2, 1.2]. */
  temperatures?: number[]
  /** Verifier criteria set (default `coding`). */
  criteria?: 'coding' | 'general'
  /** Verifier model id; defaults to the candidate model of each request. */
  verifierModel?: string
  verifierTemperature?: number
  verifierMaxTokens?: number
  /** Write decision records to $DSH_HOME/dsh-turbo/decisions.jsonl (default true). */
  decisionLog?: boolean
}

const catalogModel: z<TurboCatalogModel> = z.object({
  id: z.string().required(),
  name: z.string(),
  description: z.string(),
  contextWindow: z.number().step(1).min(1),
  maxTokens: z.number().step(1).min(1),
  vision: z.boolean(),
})

export const Config: z<Config> = z.object({
  apiKeyEnv: z.string().role('credential-ref').default(DEFAULT_API_KEY_ENV),
  baseURL: z.string(),
  thinking: z.union(['enabled', 'disabled']),
  reasoningEffort: z.union(['off', 'high', 'max']),
  maxTokens: z.number().step(1).min(1).max(Number.MAX_SAFE_INTEGER).default(DEFAULT_MAX_TOKENS),
  defaultContextWindow: z.number().step(1).min(1).default(DEFAULT_CONTEXT_WINDOW),
  models: z.array(catalogModel).default(DEFAULT_MODELS),
  streamIdleTimeoutMs: z.number().min(Number.MIN_VALUE).max(MAX_TIMER_DELAY_MS).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS),
  retryPolicy: RetryPolicySchema,
  n: z.number().step(1).min(1).max(16).default(5),
  repeats: z.number().step(1).min(1).max(16).default(1),
  pivots: z.number().step(1).min(1).max(8).default(2),
  temperatures: z.array(z.number().min(0).max(2)),
  criteria: z.union(['coding', 'general']).default('coding'),
  verifierModel: z.string(),
  verifierTemperature: z.number().min(0).max(2).default(1),
  verifierMaxTokens: z.number().step(1).min(64).max(32_000).default(2_000),
  decisionLog: z.boolean().default(true),
})

/**
 * The one explicit resolve step from raw config to validated facts, re-judged
 * for the composition entry at load (fail loud) and for each settings
 * snapshot at first use.
 */
export function resolveAdapterOptions(
  config: Config,
  environment?: LaunchEnvironmentSnapshot,
): TurboConnectionOptions {
  const n = config.n ?? 5
  if (!Number.isInteger(n) || n < 1 || n > 16) {
    throw new Error('dsh-turbo: n must be an integer between 1 and 16')
  }
  const repeats = config.repeats ?? 1
  if (!Number.isInteger(repeats) || repeats < 1 || repeats > 16) {
    throw new Error('dsh-turbo: repeats must be an integer between 1 and 16')
  }
  const pivots = config.pivots ?? 2
  if (!Number.isInteger(pivots) || pivots < 1) {
    throw new Error('dsh-turbo: pivots must be a positive integer')
  }
  if (config.temperatures !== undefined
    && config.temperatures.some(value => !Number.isFinite(value) || value < 0 || value > 2)) {
    throw new Error('dsh-turbo: temperatures must be finite numbers between 0 and 2')
  }
  const streamIdleTimeoutMs = config.streamIdleTimeoutMs ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS
  if (!Number.isFinite(streamIdleTimeoutMs) || streamIdleTimeoutMs <= 0 || streamIdleTimeoutMs > MAX_TIMER_DELAY_MS) {
    throw new Error(`dsh-turbo: streamIdleTimeoutMs must be a positive finite number no greater than ${MAX_TIMER_DELAY_MS}`)
  }
  const models = (config.models ?? DEFAULT_MODELS).map((model) => {
    if (model.id.length === 0) throw new Error('dsh-turbo: catalog model ids must be non-empty')
    return { ...model }
  })
  const verifierTemperature = config.verifierTemperature ?? 1
  if (!Number.isFinite(verifierTemperature) || verifierTemperature < 0 || verifierTemperature > 2) {
    throw new Error('dsh-turbo: verifierTemperature must be between 0 and 2')
  }
  const verifierMaxTokens = config.verifierMaxTokens ?? 2_000
  if (!Number.isInteger(verifierMaxTokens) || verifierMaxTokens < 64) {
    throw new Error('dsh-turbo: verifierMaxTokens must be an integer of at least 64')
  }
  return {
    baseURL: config.baseURL ?? environment?.get(BASE_URL_ENV)?.value ?? PUBLIC_BASE_URL,
    apiKeyEnv: credentialRef(config.apiKeyEnv ?? DEFAULT_API_KEY_ENV),
    defaults: {
      thinking: config.thinking,
      reasoningEffort: config.reasoningEffort,
    },
    maxTokens: config.maxTokens ?? DEFAULT_MAX_TOKENS,
    defaultContextWindow: config.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW,
    models,
    streamIdleTimeoutMs,
    retryPolicy: resolveRetryPolicy(config.retryPolicy, 'dsh-turbo: retryPolicy'),
    n,
    repeats,
    pivots,
    ...config.temperatures === undefined ? {} : { temperatures: config.temperatures },
    criteria: config.criteria ?? 'coding',
    ...config.verifierModel === undefined ? {} : { verifierModel: config.verifierModel },
    verifierTemperature,
    verifierMaxTokens,
  }
}

/** The subset of the attachment seam this plugin consumes (structural, optional service). */
interface AttachmentReader {
  readImage(
    ref: { attachmentId: unknown; mediaType: string },
    signal?: AbortSignal,
  ): Promise<{ data: Uint8Array }>
}

/** The subset of the dsh-TUI status seam this plugin consumes (structural, optional service). */
interface TuiStatusLike {
  set(key: string, text: string | undefined): unknown
}

/** The subset of the commands seam this plugin consumes (structural, optional service). */
interface CommandsLike {
  register(definition: {
    name: string
    description: string
    input?: { hint: string }
    handler: (invocation: { rawInput: string }) =>
      | { kind: 'success'; text?: string }
      | { kind: 'error'; text: string }
  }): unknown
}

function collectImageBlocks(blocks: readonly ContentBlock[], out: Extract<ContentBlock, { type: 'image' }>[]): void {
  for (const block of blocks) {
    if (block.type === 'image') out.push(block)
    else if (block.type === 'tool-result') collectImageBlocks(block.content, out)
  }
}

/** Format one decision for the /turbo command. */
function describeDecision(decision: TurboDecision): string {
  const lines = [
    `last decision (${decision.at}):`,
    `  model ${decision.model}, verifier ${decision.verifierModel}`,
    `  candidates ${decision.survivors}/${decision.requested} survived`
    + (decision.failed > 0 ? ` (${decision.failed} failed)` : ''),
    decision.degraded
      ? '  verification FAILED — used first surviving candidate'
      : `  winner #${decision.winner + 1}`
      + (decision.outcome === undefined
        ? ''
        : ` — BT wins [${decision.outcome.wins.map(w => w.toFixed(2)).join(', ')}], pivots [${decision.outcome.pivots.map(p => p + 1).join(', ')}]`),
    `  extra spend: ${decision.extraUsage.inputTokens} in / ${decision.extraUsage.outputTokens} out`
    + ` / ${decision.extraUsage.cacheReadTokens} cached over ${decision.extraUsage.calls} calls, ${decision.durationMs}ms`,
  ]
  for (const [at, candidate] of decision.candidates.entries()) {
    lines.push(`  #${at + 1} t=${candidate.temperature ?? 'default'} ${candidate.finish}`
      + `${candidate.outputTokens === undefined ? '' : ` ${candidate.outputTokens}tok`}: ${candidate.preview.replaceAll('\n', ' ')}`)
  }
  return lines.join('\n')
}

export function apply(ctx: Context, config: Config = {}): void {
  let current: () => Config = () => config
  let lastRaw: Config | undefined
  let lastGood: TurboConnectionOptions | undefined
  const options = (): TurboConnectionOptions => {
    const raw = current()
    if (raw === lastRaw && lastGood !== undefined) return lastGood
    try {
      const next = resolveAdapterOptions(raw, launchEnvironmentOf(ctx))
      lastRaw = raw
      lastGood = next
      return next
    } catch (error) {
      if (lastGood === undefined) throw error
      lastRaw = raw
      ctx.logger.error('dsh-turbo: keeping the last good configuration after an invalid settings section')
      ctx.logger.error(error)
      return lastGood
    }
  }
  options()

  const resolveApiKey = async (connection: TurboConnectionOptions): Promise<string> => {
    const ref = connection.apiKeyEnv
    const credentials = ctx.get('credentials')
    if (credentials !== undefined) {
      const hit = await credentials.resolve(ref)
      if (hit !== undefined) return assertUsableApiKey(hit.value, 'dsh-turbo', ref)
    } else {
      const ambient = launchEnvironmentOf(ctx).get(ref)
      if (ambient !== undefined && ambient.value.length > 0) {
        return assertUsableApiKey(ambient.value, 'dsh-turbo', ref)
      }
    }
    throw new LlmError(
      `dsh-turbo: no API key for provider route "${PROVIDER}"; store ${ref} through the credentials`
      + ` service (the web Models page writes it), or export ${ref} in the launching environment`
      + ` — the same credential the official DeepSeek route uses`,
      'MISSING_CREDENTIAL',
    )
  }

  const resolveImages = async (
    generate: GenerateOptions,
    signal: AbortSignal | undefined,
  ): Promise<ImageDataIndex> => {
    const imageBlocks: Extract<ContentBlock, { type: 'image' }>[] = []
    for (const message of generate.messages) collectImageBlocks(message.content, imageBlocks)
    if (imageBlocks.length === 0) return new Map()
    const attachments = ctx.get('attachments') as AttachmentReader | undefined
    if (attachments === undefined) {
      throw new LlmError(
        'dsh-turbo: the request carries images but no attachment service is available in this composition',
        'UNSUPPORTED_CONTENT',
      )
    }
    const index = new Map<string, string>()
    for (const block of imageBlocks) {
      const key = String(block.attachment.attachmentId)
      if (index.has(key)) continue
      const stored = await attachments.readImage(block.attachment, signal)
      index.set(key, `data:${block.attachment.mediaType};base64,${Buffer.from(stored.data).toString('base64')}`)
    }
    return index
  }

  // Runtime kill switch and decision memory for the /turbo command.
  let fanOutEnabled = true
  const decisions: TurboDecision[] = []
  const totals = { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, calls: 0, steps: 0 }
  let logDirReady: Promise<string | undefined> | undefined
  const decisionLogDir = (): Promise<string | undefined> => {
    logDirReady ??= (async () => {
      if (current().decisionLog === false) return undefined
      const home = launchEnvironmentOf(ctx).get('DSH_HOME')?.value
      if (home === undefined || home.length === 0) return undefined
      const dir = join(home, 'dsh-turbo')
      try {
        await mkdir(dir, { recursive: true })
        return dir
      } catch (error) {
        ctx.logger.warn('dsh-turbo: cannot create the decision log directory; decisions stay in memory only')
        ctx.logger.warn(error)
        return undefined
      }
    })()
    return logDirReady
  }
  const record = (decision: TurboDecision): void => {
    decisions.push(decision)
    if (decisions.length > 20) decisions.shift()
    totals.inputTokens += decision.extraUsage.inputTokens
    totals.outputTokens += decision.extraUsage.outputTokens
    totals.cacheReadTokens += decision.extraUsage.cacheReadTokens
    totals.calls += decision.extraUsage.calls
    totals.steps += 1
    void decisionLogDir().then(async (dir) => {
      if (dir === undefined) return
      await appendFile(join(dir, 'decisions.jsonl'), `${JSON.stringify(decision)}\n`)
    }).catch((error: unknown) => {
      ctx.logger.warn('dsh-turbo: failed to append a decision record')
      ctx.logger.warn(error)
    })
  }

  const report = (text: string | undefined): void => {
    const status = ctx.get('tuiStatus') as TuiStatusLike | undefined
    if (status === undefined) return
    try {
      status.set('dsh-turbo', text)
    } catch (_statusUnavailable) {
      // The status line is presentation only; its failure never touches a request.
    }
  }

  const adapter = new TurboAdapter({
    options,
    resolveApiKey,
    resolveImages,
    bypass: () => !fanOutEnabled,
    report,
    record,
  })
  ctx.llm.registerConfigurableProviders([
    { provider: PROVIDER, displayName: 'DeepSeek Turbo', settingsNs: NS, settingsPath: [] },
  ])
  const registration = ctx.llm.registerAdapter([PROVIDER], adapter)
  let registeredPolicy = options().retryPolicy
  const ensureRegistrationFacts = (): void => {
    const policy = options().retryPolicy
    if (deepEqualJson(policy, registeredPolicy)) return
    registration.replace([PROVIDER])
    registeredPolicy = policy
  }

  installSettingsSection(ctx, NS, Config, config, {
    setSource: (source) => {
      current = source
    },
    onChange: ensureRegistrationFacts,
  })

  // The /turbo command is optional surface: registered only where a commands
  // service exists, headless otherwise.
  ctx.inject(['commands'], (cctx) => {
    const commands = (cctx as unknown as { commands: CommandsLike }).commands
    commands.register({
      name: 'turbo',
      description: 'Best-of-N status: last decision and cumulative extra spend; "on"/"off" toggles fan-out',
      input: { hint: '[on|off]' },
      handler: (invocation) => {
        const argument = invocation.rawInput.trim().toLowerCase()
        if (argument === 'off') {
          fanOutEnabled = false
          return { kind: 'success', text: 'dsh-turbo: fan-out OFF — the route now passes single calls through' }
        }
        if (argument === 'on') {
          fanOutEnabled = true
          return { kind: 'success', text: 'dsh-turbo: fan-out ON' }
        }
        if (argument.length > 0) {
          return { kind: 'error', text: `dsh-turbo: unknown argument "${argument}" (expected "on" or "off")` }
        }
        const header = `dsh-turbo: fan-out ${fanOutEnabled ? 'ON' : 'OFF'}, n=${options().n},`
          + ` verified steps ${totals.steps}, extra spend ${totals.inputTokens} in`
          + ` / ${totals.outputTokens} out / ${totals.cacheReadTokens} cached over ${totals.calls} calls`
        const last = decisions[decisions.length - 1]
        return {
          kind: 'success',
          text: last === undefined ? header : `${header}\n${describeDecision(last)}`,
        }
      },
    })
  })
}
