/**
 * The one HTTP transport dsh-turbo owns: fetch against a DeepSeek
 * (OpenAI-compatible) chat-completions endpoint, streaming for candidate
 * generation and non-streaming (with logprobs) for verifier scoring. Both the
 * fan-out and the judge go through this same client with the same credential —
 * error normalization adapted from @deepseek-ai/dsh-llm-deepseek
 * `src/adapter.ts` (0.1.1-rc.2).
 *
 * @module dsh-turbo/transport
 */

import {
  CONTEXT_WINDOW_EXCEEDED_CODE,
  isContextWindowExceededError,
  isQuotaExceededError,
  LlmError,
  ProviderRequestId,
  QUOTA_EXCEEDED_CODE,
} from '@deepseek-ai/dsh-llm'
import { parseSse } from './sse.ts'
import type { WireCompletion, WireError, WireRequest } from './types.ts'

/** Connection facts one request runs against. */
export interface TransportConnection {
  /** Endpoint base; `/chat/completions` is appended. */
  baseURL: string
}

function providerRetryAfterMs(value: string | null): number | undefined {
  if (value === null) return undefined
  if (/^\d+$/.test(value)) {
    const delay = Number(value) * 1_000
    return Number.isFinite(delay) && delay > 0 ? delay : undefined
  }
  const delay = Date.parse(value) - Date.now()
  return Number.isFinite(delay) && delay > 0 ? delay : undefined
}

function requestId(headers: Headers): ReturnType<typeof ProviderRequestId> | undefined {
  const value = headers.get('x-request-id') ?? headers.get('x-deepseek-request-id')
  return value === null || value.length === 0 ? undefined : ProviderRequestId(value)
}

/** Map an HTTP status to a stable LlmError code. */
export function httpErrorCode(status: number, error?: WireError['error']): string {
  if (status === 401 || status === 403) return 'AUTH'
  const detail = [error?.code, error?.type, error?.message].filter(Boolean).join(' ')
  if (isQuotaExceededError(detail)) return QUOTA_EXCEEDED_CODE
  if (status === 429) return 'RATE_LIMIT'
  if (status === 400) {
    if (isContextWindowExceededError(detail)) return CONTEXT_WINDOW_EXCEEDED_CODE
    return 'INVALID_REQUEST'
  }
  if (status >= 500) return 'SERVER'
  return `HTTP_${status}`
}

async function post(
  body: WireRequest,
  connection: TransportConnection,
  apiKey: string,
  signal: AbortSignal,
  accept: string,
): Promise<Response> {
  const payload = JSON.stringify(body)
  let response: Response
  try {
    response = await fetch(`${connection.baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        'authorization': `Bearer ${apiKey}`,
        'content-type': 'application/json',
        'accept': accept,
      },
      body: payload,
      signal,
    })
  } catch (error: unknown) {
    if (signal.aborted) throw error
    throw new LlmError(
      `DeepSeek API request to ${connection.baseURL} failed`,
      'TRANSPORT',
      { cause: error },
    )
  }

  if (!response.ok) {
    let message = `DeepSeek API error (HTTP ${response.status})`
    let providerError: WireError['error']
    try {
      const parsed = await response.json() as WireError
      providerError = parsed.error
      if (providerError?.message) message = providerError.message
    } catch {
      // Only swallow error-body parsing: the HTTP status still identifies the
      // failure, so malformed gateway JSON must not mask it.
    }
    const delay = providerRetryAfterMs(response.headers.get('retry-after'))
    const id = requestId(response.headers)
    throw new LlmError(message, httpErrorCode(response.status, providerError), {
      status: response.status,
      ...delay === undefined ? {} : { providerRetryAfterMs: delay },
      ...id === undefined ? {} : { requestId: id },
    })
  }
  return response
}

/**
 * One streaming chat-completions call: yields SSE data payloads,
 * `[DONE]`-terminated. Used by candidate generation.
 */
export async function* streamCompletion(
  body: WireRequest,
  connection: TransportConnection,
  apiKey: string,
  signal: AbortSignal,
  onComment?: () => void,
): AsyncGenerator<string> {
  const response = await post(body, connection, apiKey, signal, 'text/event-stream')
  if (!response.body) {
    throw new LlmError('DeepSeek API returned no response body', 'EMPTY_RESPONSE')
  }
  yield* parseSse(response.body, onComment)
}

/**
 * One non-streaming chat-completions call, whole response parsed as JSON.
 * Used by verifier scoring, where the logprobs arrive on the completed choice.
 */
export async function completeOnce(
  body: WireRequest,
  connection: TransportConnection,
  apiKey: string,
  signal: AbortSignal,
): Promise<WireCompletion> {
  const response = await post(body, connection, apiKey, signal, 'application/json')
  try {
    return await response.json() as WireCompletion
  } catch (error: unknown) {
    throw new LlmError('DeepSeek API returned malformed completion JSON', 'MALFORMED_RESPONSE', { cause: error })
  }
}
