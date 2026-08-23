#!/usr/bin/env node
/**
 * Send a follow-up prompt into the already-running dsh web session and wait
 * for the turn to complete. Used for stress-testing the turbo route with
 * successive requests against the same conversation context.
 *
 * Usage: node scripts/drive-followup.mjs <port> <sessionsRoot> <doneMarker> <promptFile>
 */
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { readdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'

const [port, sessionsRoot, doneMarker, promptFile] = process.argv.slice(2)
const require = createRequire(join(process.cwd(), 'package.json'))
const { chromium } = require('playwright')
const log = m => console.log(`[followup] ${m}`)

async function filesBelow(dir) {
  const out = []
  let entries
  try { entries = await readdir(dir, { withFileTypes: true }) } catch { return out }
  for (const e of entries) {
    const p = join(dir, e.name)
    if (e.isDirectory()) out.push(...await filesBelow(p))
    else if (e.isFile()) out.push(p)
  }
  return out
}
const blocksOf = r => r.data?.message?.content ?? r.data?.content ?? []
async function records() {
  const files = (await filesBelow(sessionsRoot)).filter(p => p.endsWith('session.jsonl'))
  const out = []
  for (const p of files) {
    for (const line of (await readFile(p, 'utf8')).split('\n').filter(Boolean)) {
      try { out.push(JSON.parse(line)) } catch {}
    }
  }
  return out
}

const prompt = await readFile(promptFile, 'utf8')
const before = (await records()).filter(r => r.type === 'turn/end').length
log(`turns before: ${before}`)

const browser = await chromium.launch()
try {
  const page = await browser.newPage({ viewport: { width: 1400, height: 950 }, locale: 'en-US' })
  await page.goto(`http://127.0.0.1:${port}`, { waitUntil: 'domcontentloaded' })
  const notice = page.getByRole('button', { name: 'Continue' })
  if (await notice.waitFor({ state: 'visible', timeout: 15_000 }).then(() => true).catch(() => false)) {
    await notice.click()
    await page.waitForFunction(() => document.querySelectorAll('[class*="_mask_"]').length === 0, undefined, { timeout: 20_000 })
  }
  // Resume the existing conversation: the sidebar's first session entry.
  const entry = page.locator('[role="treeitem"], [data-session-id]').first()
  if (await entry.waitFor({ state: 'visible', timeout: 20_000 }).then(() => true).catch(() => false)) {
    await entry.click()
    log('opened the existing session')
  } else {
    log('WARNING: no session entry found in the sidebar; sending into the current view')
  }
  const composer = page.getByRole('textbox').last()
  await composer.click()
  await composer.fill(prompt)
  assert((await composer.inputValue()).length > 50, 'composer did not accept the prompt')
  await page.getByRole('button', { name: 'Send message' }).click()
  log('prompt sent — best-of-5 running …')

  const deadline = Date.now() + 45 * 60_000
  for (;;) {
    const all = await records()
    const turns = all.filter(r => r.type === 'turn/end')
    const done = all.some(r => r.type === 'assistant/message'
      && blocksOf(r).some(b => b.type === 'text' && b.text.includes(doneMarker)))
    if (turns.length > before && turns.at(-1)?.data?.reason?.kind === 'completed' && done) break
    assert(Date.now() < deadline, 'timed out waiting for the follow-up turn')
    await new Promise(r => setTimeout(r, 3_000))
  }
  log('turn completed durably')
  await page.screenshot({ path: '.artifacts/cs-followup-final.png', fullPage: true })
} finally {
  await browser.close()
}
