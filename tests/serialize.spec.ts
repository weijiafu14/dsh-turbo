import { describe, expect, it } from 'vitest'
import type { Message } from '@deepseek-ai/dsh-llm'
import { serializeMessages, serializeRequest } from '../src/serialize.ts'
import type { WireContentPart, WireUserMessage } from '../src/types.ts'

const imageBlock = (id: string) => ({
  type: 'image' as const,
  attachment: {
    attachmentId: id,
    mediaType: 'image/png' as const,
    bytes: 3,
    width: 1,
    height: 1,
  },
}) as unknown as Extract<Message['content'][number], { type: 'image' }>

const user = (content: Message['content']): Message =>
  ({ role: 'user', content }) as unknown as Message

describe('serializeMessages', () => {
  it('keeps text-only user content as a plain string', () => {
    const wire = serializeMessages([user([{ type: 'text', text: 'hello' }])], new Map())
    expect(wire).toEqual([{ role: 'user', content: 'hello' }])
  })

  it('turns image-bearing user content into OpenAI content parts', () => {
    const wire = serializeMessages(
      [user([{ type: 'text', text: 'what color?' }, imageBlock('att-1')])],
      new Map([['att-1', 'data:image/png;base64,QUJD']]),
    )
    const message = wire[0] as WireUserMessage
    const parts = message.content as WireContentPart[]
    expect(parts).toEqual([
      { type: 'text', text: 'what color?' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,QUJD' } },
    ])
  })

  it('throws on an unresolved image attachment (internal invariant)', () => {
    expect(() => serializeMessages([user([imageBlock('missing')])], new Map()))
      .toThrow(/was not resolved/)
  })

  it('expands tool results into tool messages and images into a labeled follow-up', () => {
    const message = user([{
      type: 'tool-result',
      toolCallId: 'call-1' as never,
      content: [{ type: 'text', text: 'took screenshot' }, imageBlock('shot')],
    }])
    const wire = serializeMessages([message], new Map([['shot', 'data:image/png;base64,WA==']]))
    expect(wire[0]).toEqual({ role: 'tool', tool_call_id: 'call-1', content: 'took screenshot' })
    const followUp = wire[1] as WireUserMessage
    const parts = followUp.content as WireContentPart[]
    expect(parts[0]).toEqual({ type: 'text', text: '(image output of tool call call-1)' })
    expect(parts[1]).toEqual({ type: 'image_url', image_url: { url: 'data:image/png;base64,WA==' } })
  })

  it('replays assistant tool calls with reasoning passback', () => {
    const assistant = {
      role: 'assistant',
      content: [
        { type: 'reasoning', text: 'thinking...' },
        { type: 'tool-call', id: 'c1', name: 'read', arguments: '{"path":"x"}' },
      ],
    } as unknown as Message
    const wire = serializeMessages([assistant], new Map())
    expect(wire[0]).toEqual({
      role: 'assistant',
      content: '',
      reasoning_content: 'thinking...',
      tool_calls: [{ id: 'c1', type: 'function', function: { name: 'read', arguments: '{"path":"x"}' } }],
    })
  })
})

describe('serializeRequest', () => {
  const base = {
    provider: 'deepseek-turbo',
    model: 'deepseek-v4-flash-vision-exp',
    messages: [user([{ type: 'text', text: 'go' }])],
  } as never

  it('streams with usage reporting and applies the candidate temperature', () => {
    const body = serializeRequest(base, {}, new Map(), 0.7)
    expect(body.stream).toBe(true)
    expect(body.stream_options).toEqual({ include_usage: true })
    expect(body.temperature).toBe(0.7)
    expect(body.logprobs).toBeUndefined()
  })

  it('disables thinking for session-title calls', () => {
    const body = serializeRequest(
      { ...(base as object), purpose: 'session-title' } as never,
      {},
      new Map(),
    )
    expect(body.thinking).toEqual({ type: 'disabled' })
  })
})
