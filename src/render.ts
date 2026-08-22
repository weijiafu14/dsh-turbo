/**
 * Render the task context and candidates for the verifier. The context is a
 * role-labeled transcript of the exact request the candidates answered, with
 * images inlined as real content parts (the verifier model is multimodal, so
 * it judges vision candidates by looking at the same pixels). Candidates are
 * rendered from their assembled blocks: visible text plus tool-call intents;
 * private reasoning is withheld so candidates are judged by what they DO.
 *
 * @module dsh-turbo/render
 */

import type { ContentBlock, GenerateOptions } from '@deepseek-ai/dsh-llm'
import type { ImageDataIndex } from './serialize.ts'
import type { WireContentPart } from './types.ts'

/** Accumulates text runs and image parts into one parts array. */
class PartsBuilder {
  private readonly parts: WireContentPart[] = []
  private text = ''

  append(text: string): void {
    this.text += text
  }

  image(url: string): void {
    this.flush()
    this.parts.push({ type: 'image_url', image_url: { url } })
  }

  private flush(): void {
    if (this.text.length > 0) {
      this.parts.push({ type: 'text', text: this.text })
      this.text = ''
    }
  }

  build(): WireContentPart[] {
    this.flush()
    return this.parts
  }
}

function renderBlocks(
  blocks: readonly ContentBlock[],
  out: PartsBuilder,
  images: ImageDataIndex,
): void {
  for (const block of blocks) {
    switch (block.type) {
      case 'text':
        out.append(block.text)
        break
      case 'image': {
        const url = images.get(String(block.attachment.attachmentId))
        if (url !== undefined) out.image(url)
        else out.append('[image]')
        break
      }
      case 'tool-call':
        out.append(`\n[tool call ${block.name}(${block.arguments})]`)
        break
      case 'tool-result':
        out.append(`\n[tool result for ${block.toolCallId}${block.isError === true ? ' (ERROR)' : ''}]:\n`)
        renderBlocks(block.content, out, images)
        break
      case 'reasoning':
        // Context-side reasoning is model-private; the transcript keeps only a marker.
        break
      default:
        break
    }
  }
}

/**
 * Render the request the candidates answered as one shared, cache-stable
 * context block: system prompt, then each message labeled by role.
 * @param options - the harness request being fanned out.
 * @param images - resolved data URLs for the request's image attachments.
 * @returns content parts to embed in every verifier call of this step.
 */
export function renderContext(options: GenerateOptions, images: ImageDataIndex): WireContentPart[] {
  const out = new PartsBuilder()
  if (options.system !== undefined && options.system.length > 0) {
    out.append(`[system]\n${options.system}\n\n`)
  }
  for (const message of options.messages) {
    out.append(`[${message.role}]\n`)
    renderBlocks(message.content, out, images)
    out.append('\n\n')
  }
  return out.build()
}

/**
 * Render one candidate from its assembled content blocks: visible text and
 * tool-call intents, in emission order. Reasoning is withheld — the verifier
 * judges observable behavior, and reasoning dumps would drown the signal.
 * @param blocks - the candidate's `block-end` contents, in block order.
 * @returns the candidate text shown to the verifier.
 */
export function renderCandidate(blocks: readonly ContentBlock[]): string {
  const lines: string[] = []
  for (const block of blocks) {
    switch (block.type) {
      case 'text':
        lines.push(block.text)
        break
      case 'tool-call':
        lines.push(`[tool call ${block.name}(${block.arguments})]`)
        break
      default:
        break
    }
  }
  const rendered = lines.join('\n').trim()
  return rendered.length > 0 ? rendered : '(the candidate produced no visible output)'
}
