/**
 * DeepSeek chat-completions wire format (OpenAI-compatible), extended for
 * dsh-turbo with multimodal content parts (deepseek-v4-flash-vision-exp) and
 * token logprobs (the verifier's fine-grained scoring signal).
 *
 * Base shapes adapted from @deepseek-ai/dsh-llm-deepseek `src/types.ts`
 * (0.1.1-rc.2); multimodal and logprob fields follow the official docs
 * (api/create-chat-completion, guides/vision).
 *
 * @module dsh-turbo/types
 */

/** One part of a multimodal message content array. */
export type WireContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } }

/** Request body for `POST {baseURL}/chat/completions`. */
export interface WireRequest {
  model: string
  messages: WireMessage[]
  /** Candidate calls stream; verifier calls do not (logprobs arrive whole). */
  stream: boolean
  stream_options?: { include_usage: true }
  /** Thinking-mode toggle (top level on the wire). */
  thinking?: { type: 'enabled' | 'disabled' }
  reasoning_effort?: 'high' | 'max'
  tools?: WireTool[]
  temperature?: number
  max_tokens?: number
  stop?: string[]
  /** Verifier calls only: return per-token log probabilities. */
  logprobs?: boolean
  /** 0-20; the alternatives at each position, needed for score expectation. */
  top_logprobs?: number
}

export interface WireSystemMessage {
  role: 'system'
  content: string
}

/** User content is a plain string for text-only turns, parts when images ride along. */
export interface WireUserMessage {
  role: 'user'
  content: string | WireContentPart[]
}

export interface WireToolMessage {
  role: 'tool'
  tool_call_id: string
  content: string
}

export type WireMessage =
  | WireSystemMessage
  | WireUserMessage
  | WireAssistantMessage
  | WireToolMessage

export interface WireAssistantMessage {
  role: 'assistant'
  content: string | null
  /** CoT passback; required on tool-call turns in thinking mode. */
  reasoning_content?: string
  tool_calls?: WireToolCall[]
}

export interface WireToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

export interface WireTool {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: Record<string, unknown>
  }
}

/** One parsed SSE `data:` payload (a chat.completion.chunk). */
export interface WireChunk {
  choices?: WireChoice[]
  usage?: WireUsage | null
}

export interface WireChoice {
  delta?: WireDelta
  finish_reason?: string | null
}

export interface WireDelta {
  role?: string
  content?: string | null
  reasoning_content?: string | null
  tool_calls?: WireToolCallDelta[]
}

export interface WireToolCallDelta {
  index: number
  id?: string
  type?: 'function'
  function?: {
    name?: string
    arguments?: string
  }
}

export interface WireUsage {
  prompt_tokens: number
  completion_tokens: number
  prompt_cache_hit_tokens?: number
  prompt_cache_miss_tokens?: number
  prompt_tokens_details?: { cached_tokens?: number }
  completion_tokens_details?: { reasoning_tokens?: number }
}

/** One alternative token at a position. */
export interface WireTopLogprob {
  token: string
  logprob: number
}

/** Per-token logprob entry of a non-streaming completion. */
export interface WireLogprobEntry {
  token: string
  logprob: number
  top_logprobs?: WireTopLogprob[]
}

/** Non-streaming completion response (verifier calls). */
export interface WireCompletion {
  choices?: {
    message?: { content?: string | null; reasoning_content?: string | null }
    logprobs?: { content?: WireLogprobEntry[] | null } | null
    finish_reason?: string | null
  }[]
  usage?: WireUsage | null
}

/** Non-2xx error body. */
export interface WireError {
  error?: { message?: string; type?: string; code?: string }
}
