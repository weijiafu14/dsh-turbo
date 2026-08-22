/**
 * Decode an SSE byte stream into event `data` payloads. Adapted from
 * @deepseek-ai/dsh-llm-deepseek `src/sse.ts` (0.1.1-rc.2), logic unchanged:
 * framing belongs to `eventsource-parser`, the literal `[DONE]` is yielded so
 * the caller owns final flushing, and EOF before it is truncation.
 *
 * @module dsh-turbo/sse
 */

import { EventSourceParserStream } from 'eventsource-parser/stream'
import { LlmError } from '@deepseek-ai/dsh-llm'

/** The terminal payload DeepSeek (and OpenAI) send after the last chunk. */
export const DONE = '[DONE]'

/**
 * Parse an SSE byte stream into data payloads. Yields `[DONE]` as the final
 * value and returns; throws `LlmError('STREAM_CLOSED')` when the stream ends
 * without it.
 * @param stream - raw SSE bytes; reads may split anywhere.
 * @param onComment - optional transport-activity callback.
 * @returns each event's data payload in arrival order, `[DONE]` last.
 */
export async function* parseSse(
  stream: ReadableStream<BufferSource>,
  onComment?: (comment: string) => void,
): AsyncGenerator<string> {
  const events = stream
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new EventSourceParserStream({ onComment }))
  for await (const { data } of events) {
    yield data
    if (data === DONE) return
  }
  throw new LlmError('SSE stream ended without [DONE]', 'STREAM_CLOSED')
}
