/**
 * Serialize harness messages into DeepSeek chat completions, multimodal-aware.
 * Adapted from @deepseek-ai/dsh-llm-deepseek `src/serialize.ts` (0.1.1-rc.2);
 * the delta is image support: user-content image blocks become OpenAI-style
 * `image_url` content parts (base64 data URLs resolved by the adapter before
 * serialization), and tool-result images ride in a follow-up user message
 * because the wire `tool` role only carries text.
 *
 * @module dsh-turbo/serialize
 */

import { LlmError } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, Message } from '@deepseek-ai/dsh-llm'
import type { WireContentPart, WireMessage, WireRequest, WireTool } from './types.ts'

/** Adapter-level request defaults (from plugin config). */
export interface RequestDefaults {
  thinking?: 'enabled' | 'disabled' | undefined
  reasoningEffort?: 'off' | 'high' | 'max' | undefined
}

/** attachmentId -> base64 data URL, resolved once per model call. */
export type ImageDataIndex = ReadonlyMap<string, string>

interface ResolvedThinking {
  thinking?: 'enabled' | 'disabled'
  reasoningEffort?: 'high' | 'max'
}

function reasoningEffort(effort: NonNullable<GenerateOptions['reasoningEffort']>): 'off' | 'high' | 'max' {
  if (effort === 'off' || effort === 'high' || effort === 'max') {
    return effort as 'off' | 'high' | 'max'
  }
  throw new LlmError(
    `DeepSeek does not support reasoning effort "${effort}"`,
    'UNSUPPORTED_REASONING_EFFORT',
  )
}

/** Resolve one legal thinking/effort pair without exposing `off` as a wire effort. */
export function resolveThinking(options: GenerateOptions, defaults: RequestDefaults): ResolvedThinking {
  if (options.purpose === 'session-title') return { thinking: 'disabled' }
  const effort = options.reasoningEffort === undefined
    ? defaults.reasoningEffort
    : reasoningEffort(options.reasoningEffort)
  if (defaults.thinking === 'disabled' && effort !== undefined && effort !== 'off') {
    throw new LlmError(
      `DeepSeek deployment does not support reasoning effort "${effort}"`,
      'UNSUPPORTED_REASONING_EFFORT',
    )
  }
  if (effort === 'off') return { thinking: 'disabled' }
  if (effort === 'high' || effort === 'max') {
    return { thinking: 'enabled', reasoningEffort: effort }
  }
  return defaults.thinking === undefined ? {} : { thinking: defaults.thinking }
}

function flattenText(blocks: readonly ContentBlock[]): string {
  return blocks
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('')
}

/** Resolve one image block to its data-URL content part. */
function imagePart(block: Extract<ContentBlock, { type: 'image' }>, images: ImageDataIndex): WireContentPart {
  const url = images.get(String(block.attachment.attachmentId))
  if (url === undefined) {
    // The adapter resolves every attachment before serializing; a miss here is
    // an internal invariant break, not a user error.
    throw new LlmError(
      `dsh-turbo: image attachment ${String(block.attachment.attachmentId)} was not resolved before serialization`,
      'UNSUPPORTED_CONTENT',
    )
  }
  return { type: 'image_url', image_url: { url } }
}

/** Serialize mixed user content: plain string when text-only, parts when images ride along. */
function userContent(blocks: readonly ContentBlock[], images: ImageDataIndex): string | WireContentPart[] {
  const imageBlocks = blocks.filter(block => block.type === 'image')
  const text = flattenText(blocks)
  if (imageBlocks.length === 0) return text
  const parts: WireContentPart[] = []
  if (text.length > 0) parts.push({ type: 'text', text })
  for (const block of imageBlocks) parts.push(imagePart(block, images))
  return parts
}

function serializeAssistant(message: Message): WireMessage {
  const text = flattenText(message.content)
  const reasoning = message.content
    .filter(block => block.type === 'reasoning')
    .map(block => block.text)
    .join('')
  const toolCalls = message.content
    .filter(block => block.type === 'tool-call')
    .map(block => ({
      id: block.id,
      type: 'function' as const,
      function: { name: block.name, arguments: block.arguments },
    }))

  return {
    role: 'assistant',
    // Text-less turns send "" — never null (the live API rejects null-content
    // assistant messages without tool_calls, and the message sits durably in
    // the session log).
    content: text,
    // Official passback rule: reasoning_content must return on tool-call
    // turns; it is ignored on plain turns, so drop it there to save tokens.
    ...toolCalls.length > 0 && reasoning.length > 0 ? { reasoning_content: reasoning } : {},
    ...toolCalls.length > 0 ? { tool_calls: toolCalls } : {},
  }
}

/**
 * Serialize the conversation. `tool-result` blocks become standalone
 * `{role: 'tool'}` messages; images inside a tool result cannot ride the
 * text-only tool role, so they follow as one labeled user message.
 * @param messages - the harness conversation, in order.
 * @param images - resolved data URLs for every image attachment in `messages`.
 * @returns the wire messages, order preserved.
 */
export function serializeMessages(messages: Message[], images: ImageDataIndex): WireMessage[] {
  const wire: WireMessage[] = []
  for (const message of messages) {
    if (message.role === 'system') {
      wire.push({ role: 'system', content: flattenText(message.content) })
      continue
    }
    if (message.role === 'assistant') {
      wire.push(serializeAssistant(message))
      continue
    }
    // user role: tool results ride in user messages in the harness
    // vocabulary, but DeepSeek wants them as role:'tool' messages.
    const toolResults = message.content.filter(block => block.type === 'tool-result')
    const direct = message.content.filter(block => block.type !== 'tool-result')
    const content = userContent(direct, images)
    const hasDirect = typeof content === 'string' ? content.length > 0 : content.length > 0
    if (hasDirect || toolResults.length === 0) {
      wire.push({ role: 'user', content })
    }
    for (const result of toolResults) {
      wire.push({
        role: 'tool',
        tool_call_id: result.toolCallId,
        // Empty tool output still needs SOME content on the wire.
        content: flattenText(result.content) || '(no output)',
      })
      const resultImages = result.content.filter(block => block.type === 'image')
      if (resultImages.length > 0) {
        wire.push({
          role: 'user',
          content: [
            { type: 'text', text: `(image output of tool call ${result.toolCallId})` },
            ...resultImages.map(block => imagePart(block, images)),
          ],
        })
      }
    }
  }
  return wire
}

/**
 * Build one candidate wire request: always streaming, usage reporting on,
 * optional fields omitted rather than null so provider defaults apply.
 * @param options - the harness request.
 * @param defaults - adapter-level thinking defaults.
 * @param images - resolved data URLs for every image attachment.
 * @param temperature - candidate-specific sampling temperature (the fan-out
 * ladder); `undefined` keeps the request's own value.
 * @returns the chat-completions request body.
 */
export function serializeRequest(
  options: GenerateOptions,
  defaults: RequestDefaults,
  images: ImageDataIndex,
  temperature?: number,
): WireRequest {
  const messages: WireMessage[] = []
  if (options.system !== undefined) {
    messages.push({ role: 'system', content: options.system })
  }
  messages.push(...serializeMessages(options.messages, images))

  const tools: WireTool[] | undefined = options.tools?.map(tool => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }))
  const resolvedThinking = resolveThinking(options, defaults)
  const resolvedTemperature = temperature ?? options.temperature

  return {
    model: options.model,
    messages,
    stream: true,
    stream_options: { include_usage: true },
    ...resolvedThinking.thinking !== undefined ? { thinking: { type: resolvedThinking.thinking } } : {},
    ...resolvedThinking.reasoningEffort !== undefined
      ? { reasoning_effort: resolvedThinking.reasoningEffort }
      : {},
    ...tools !== undefined && tools.length > 0 ? { tools } : {},
    ...resolvedTemperature !== undefined ? { temperature: resolvedTemperature } : {},
    ...options.maxTokens === undefined ? {} : { max_tokens: options.maxTokens },
    ...options.stop !== undefined ? { stop: options.stop } : {},
  }
}
