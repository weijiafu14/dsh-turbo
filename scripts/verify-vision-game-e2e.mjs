#!/usr/bin/env node
/**
 * Live vision E2E: one image in, one first-person 3D dungeon crawler out,
 * through the real user path.
 *
 *   stock npm DSH CLI -> `dsh plugin add <packed dsh-turbo>` -> dsh web
 *   -> paste maze-map.png into the composer -> ask for a complex-interior
 *      3D game (the community-standard dungeon-crawler discriminator)
 *   -> best-of-5 fan-out + logprob verification on the REAL DeepSeek API
 *   -> assert: the image became a native attachment, the fan-out ran and was
 *      verified (decisions.jsonl), the model actually SAW the image (the
 *      prompt never says "maze" — only the pixels do), and a substantial
 *      three.js game file landed in the workspace.
 *
 * Requires DEEPSEEK_API_KEY in the environment (read-through only, never
 * printed or written anywhere). The agent runs with the same non-interactive
 * permission mode the pi2dsh reference harness uses, so the model can write
 * the game file without a browser approval round trip.
 *
 * Usage: DEEPSEEK_API_KEY=... node scripts/verify-vision-game-e2e.mjs [--keep]
 */

import assert from 'node:assert/strict'
import { execFile as execFileCb, spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { chmod, copyFile, mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

const execFile = promisify(execFileCb)
const projectRoot = join(dirname(fileURLToPath(import.meta.url)), '..')

const DSH_CLI_SPEC = process.env.DSH_TURBO_CLI_SPEC ?? '@deepseek-ai/dsh@0.1.1-rc.2'
const PLAYWRIGHT_FROM = process.env.PLAYWRIGHT_FROM
  ?? join(projectRoot, '..', 'deepseek-harness', 'apps', 'web')
const IMAGE = join(projectRoot, 'test-assets', 'maze-map.png')
const keep = process.argv.includes('--keep')
const log = message => console.log(`[vision-game-e2e] ${message}`)

assert(process.env.DEEPSEEK_API_KEY, 'DEEPSEEK_API_KEY is required (read from the environment only)')

const root = await mkdtemp(join(tmpdir(), 'dsh-turbo-vision-'))
const home = join(root, 'dsh-home')
const cliDir = join(root, 'cli')
const workspace = join(root, 'workspace')
const artifactDir = join(projectRoot, '.artifacts')
await mkdir(home, { recursive: true })
await mkdir(workspace, { recursive: true })

const shimDir = join(root, 'bin')
await mkdir(shimDir, { recursive: true })
await writeFile(join(shimDir, 'pnpm'), '#!/bin/sh\nexec corepack pnpm@11.7.0 "$@"\n')
await chmod(join(shimDir, 'pnpm'), 0o755)

const env = {
  ...process.env,
  DSH_HOME: home,
  PATH: `${shimDir}:${process.env.PATH ?? ''}`,
  CI: '1',
  NO_COLOR: '1',
  DSH_TELEMETRY_DISABLED: '1',
  // Same non-interactive agent permission mode as the pi2dsh reference
  // harness (verify-codex-image-gen-e2e.mjs) — file writes need no approval UI.
  DSH_PERMISSION_MODE: 'danger-full-access',
  PNPM_CONFIG_MINIMUM_RELEASE_AGE: '0',
  npm_config_registry: 'https://registry.npmjs.org',
}

async function filesBelow(directory) {
  const out = []
  let entries
  try {
    entries = await readdir(directory, { withFileTypes: true })
  } catch (error) {
    if (error?.code === 'ENOENT') return out
    throw error
  }
  for (const entry of entries) {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) out.push(...await filesBelow(path))
    else if (entry.isFile()) out.push(path)
  }
  return out
}

async function allSessionRecords(sessionsRoot) {
  const files = (await filesBelow(sessionsRoot)).filter(path => path.endsWith('session.jsonl'))
  const records = []
  for (const path of files) {
    const lines = (await readFile(path, 'utf8')).split('\n').filter(Boolean)
    for (const line of lines) {
      try { records.push(JSON.parse(line)) } catch { /* partial trailing write */ }
    }
  }
  return records
}

let web
let failed = false
try {
  // ---- 1. pack + stock CLI ------------------------------------------------
  log('packing dsh-turbo …')
  const { stdout: packOut } = await execFile('corepack', ['pnpm@11.7.0', 'pack', '--silent'], { cwd: projectRoot, env })
  const tarball = join(projectRoot, packOut.trim().split('\n').pop().trim().split('/').pop())
  await mkdir(cliDir, { recursive: true })
  const at = DSH_CLI_SPEC.lastIndexOf('@')
  await writeFile(join(cliDir, 'package.json'), `${JSON.stringify({
    name: 'dsh-turbo-vision-e2e-cli',
    private: true,
    dependencies: { [DSH_CLI_SPEC.slice(0, at)]: DSH_CLI_SPEC.slice(at + 1) },
  }, null, 2)}\n`)
  await writeFile(join(cliDir, 'pnpm-workspace.yaml'), [
    'minimumReleaseAge: 0',
    'allowBuilds:',
    "  '@deepseek-ai/dsh-subprocess-local': true",
    '  node-pty: true',
    '  koffi: true',
    "  '@google/genai': false",
    '  protobufjs: false',
    '',
  ].join('\n'))
  log(`installing stock CLI ${DSH_CLI_SPEC} …`)
  await execFile('corepack', ['pnpm@11.7.0', 'install'], { cwd: cliDir, env, timeout: 300_000 })
  const dshBin = join(cliDir, 'node_modules', '.bin', 'dsh')

  // ---- 2. web profile with dsh-turbo, durable session logs ----------------
  log('dsh plugin --profile web add <tarball> …')
  await execFile(dshBin, ['plugin', '--profile', 'web', 'add', tarball], {
    env, timeout: 600_000, maxBuffer: 32 * 1024 * 1024,
  })
  const installed = JSON.parse(await readFile(join(home, 'profiles', 'web', 'node_modules', 'dsh-turbo', 'package.json'), 'utf8'))
  log(`profile carries dsh-turbo@${installed.version}`)
  await writeFile(join(home, 'profiles', 'web', 'cordis.patch.yml'), [
    '- id: session-persistence-jsonl',
    '  config:',
    "    root: !!js dshHomePath('sessions-web')",
    '    compression: none',
    '',
  ].join('\n'))
  await writeFile(join(home, 'settings.yaml'), [
    'agent-default-model:',
    '  provider: deepseek-turbo',
    '  model: deepseek-v4-flash-vision-exp',
    '',
  ].join('\n'))

  // ---- 3. launch dsh web --------------------------------------------------
  const port = 5460 + Math.floor(Math.random() * 300)
  const logRef = { value: '' }
  web = spawn(dshBin, ['--profile', 'web', '--port', String(port)], {
    cwd: root, env, stdio: ['ignore', 'pipe', 'pipe'],
  })
  web.stdout.on('data', chunk => { logRef.value += String(chunk) })
  web.stderr.on('data', chunk => { logRef.value += String(chunk) })
  const webDeadline = Date.now() + 120_000
  while (!/dsh web:/u.test(logRef.value) && web.exitCode === null && Date.now() < webDeadline) {
    await new Promise(done => setTimeout(done, 400))
  }
  if (web.exitCode !== null || !/dsh web:/u.test(logRef.value)) {
    throw new Error(`DSH web did not start:\n${logRef.value.slice(-3000)}`)
  }
  log(`dsh web up on :${port}`)

  // ---- 4. drive the real browser -----------------------------------------
  const require = createRequire(join(resolve(PLAYWRIGHT_FROM), 'package.json'))
  const { chromium } = require('playwright')
  const browser = await chromium.launch()
  try {
    const page = await browser.newPage({ viewport: { width: 1360, height: 900 }, locale: 'en-US' })
    await page.goto(`http://127.0.0.1:${port}`, { waitUntil: 'domcontentloaded' })
    const dismissNotice = async () => {
      const notice = page.getByRole('button', { name: 'Continue' })
      if (await notice.waitFor({ state: 'visible', timeout: 15_000 }).then(() => true).catch(() => false)) {
        await notice.click()
        await page.waitForFunction(() => document.querySelectorAll('[class*="_mask_"]').length === 0, undefined, { timeout: 20_000 })
      }
    }
    await dismissNotice()
    const created = await page.evaluate(async (path) => {
      const result = await fetch('/api/workspace.create', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ type: 'client-request', rpcId: 'vision-game-e2e', method: 'workspace.create', payload: { path } }),
      })
      return { status: result.status, body: await result.text() }
    }, workspace)
    assert.equal(created.status, 200, `workspace.create -> ${created.status}: ${created.body.slice(0, 300)}`)
    await page.reload({ waitUntil: 'domcontentloaded' })
    await dismissNotice()
    await page.getByRole('button', { name: 'New session' }).first().click({ timeout: 60_000 })

    // Paste the maze image into the composer — the real user gesture, through
    // the real onPaste -> intakeImages -> attachment-service admission path.
    const imageB64 = (await readFile(IMAGE)).toString('base64')
    const composer = page.getByRole('textbox').last()
    await composer.click()
    await page.evaluate((b64) => {
      const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0))
      const file = new File([bytes], 'maze-map.png', { type: 'image/png' })
      const transfer = new DataTransfer()
      transfer.items.add(file)
      const areas = document.querySelectorAll('textarea')
      const target = areas[areas.length - 1]
      target.focus()
      target.dispatchEvent(new ClipboardEvent('paste', { clipboardData: transfer, bubbles: true, cancelable: true }))
    }, imageB64)
    // The attachment thumbnail proves admission finished before we send.
    await page.locator('img').first().waitFor({ state: 'visible', timeout: 30_000 })
    log('image pasted and admitted')

    // The community-standard complex-interior discriminator (first-person
    // dungeon crawler), seeded by the image. The prompt NEVER names what the
    // image depicts — only the pixels do.
    const prompt = [
      'Look carefully at the attached image. First reply with one line beginning',
      '"IMAGE SHOWS:" describing exactly what kind of structure it depicts.',
      'Then build a FIRST-PERSON 3D exploration game whose level layout',
      'faithfully reproduces that structure: one single self-contained HTML file',
      'named game3d.html in the workspace root, using three.js (CDN <script> is',
      'fine). Requirements for a rich interior environment: the player walks',
      'INSIDE the corridors in first person (pointer-lock mouse look + WASD,',
      'collision with walls); stone-textured walls (procedural canvas textures),',
      'floor and ceiling; flickering torch point-lights along the corridors plus',
      'fog and ambient light; a few collectible glowing orbs that update a HUD',
      'counter; a mini-map overlay in a corner showing the layout and player',
      'position; the green marker from the image is the spawn point and the red',
      'marker is the goal — reaching it shows a win screen. After writing the',
      'file, reply with exactly GAME_DONE.',
    ].join(' ')
    await composer.fill(prompt)
    await page.getByRole('button', { name: 'Send message' }).click()
    log('prompt sent — best-of-5 running on the real API …')

    // Completion: durable turn/end AND the GAME_DONE marker in the session.
    const deadline = Date.now() + 30 * 60_000
    let records
    for (;;) {
      records = await allSessionRecords(join(home, 'sessions-web'))
      const turnEnd = records.findLast(record => record.type === 'turn/end')
      const doneText = records.some(record => record.type === 'assistant/message'
        && (record.data?.message?.content ?? []).some(block => block.type === 'text' && /GAME_DONE/u.test(block.text)))
      if (turnEnd?.data?.reason?.kind === 'completed' && doneText) break
      assert(Date.now() < deadline, 'timed out waiting for the completed 3D-game turn')
      assert(web.exitCode === null, `dsh web exited mid-turn:\n${logRef.value.slice(-3000)}`)
      await new Promise(done => setTimeout(done, 2_000))
    }
    log('turn completed durably')
    await mkdir(artifactDir, { recursive: true })
    await page.screenshot({ path: join(artifactDir, 'vision-game-final.png'), fullPage: true })

    // ---- 5. assertions ----------------------------------------------------
    // (a) The image went in as a native attachment on the user message.
    const userImage = records.find(record => record.type === 'user/message'
      && (record.data?.message?.content ?? []).some(block => block.type === 'image' && block.attachment?.attachmentId))
    assert(userImage, 'the user message carries no native image attachment — the paste path did not work')

    // (b) The model actually SAW the image: the reply must name the structure
    // the prompt never mentioned.
    const assistantText = records
      .filter(record => record.type === 'assistant/message')
      .flatMap(record => record.data?.message?.content ?? [])
      .filter(block => block.type === 'text')
      .map(block => block.text)
      .join('\n')
    assert.match(assistantText, /IMAGE SHOWS:/u, 'the reply never described the image')
    assert.match(assistantText, /maze|labyrinth|迷宫/iu,
      'the reply does not identify the maze — vision input did not reach the model')

    // (c) A substantial three.js game file landed in the workspace.
    const files = await filesBelow(workspace)
    const gameFile = files.find(path => path.endsWith('game3d.html')) ?? files.find(path => path.endsWith('.html'))
    assert(gameFile, `no game HTML file in the workspace (files: ${files.join(', ') || 'none'})`)
    const game = await readFile(gameFile, 'utf8')
    assert.match(game, /three/iu, 'the game file does not reference three.js')
    assert(game.length > 8_000, `the game file is implausibly small for the requested scope (${game.length} bytes)`)
    // Interior-scene richness: report which requested features made it in
    // (informative), but hard-require the first-person core.
    assert.match(game, /pointerlock|mousemove/iu, 'no first-person look controls in the game file')
    const features = {
      pointerLock: /pointerlock/iu.test(game),
      pointLights: /PointLight/u.test(game),
      fog: /fog/iu.test(game),
      texture: /CanvasTexture|createElement\('canvas'\)|createElement\("canvas"\)/u.test(game),
      minimap: /mini-?map/iu.test(game),
      hud: /hud|collect/iu.test(game),
      winScreen: /win|victory|goal/iu.test(game),
    }
    await copyFile(gameFile, join(artifactDir, 'game3d.html'))

    // (d) The fan-out really ran and was verified: dsh-turbo's own decision log.
    const decisionsRaw = await readFile(join(home, 'dsh-turbo', 'decisions.jsonl'), 'utf8')
    const decisions = decisionsRaw.split('\n').filter(Boolean).map(line => JSON.parse(line))
    assert(decisions.length > 0, 'no best-of-N decisions were recorded')
    const verified = decisions.filter(d => d.survivors >= 2 && d.degraded === false && d.outcome !== undefined)
    assert(verified.length > 0,
      `no non-degraded multi-candidate decision found (${decisions.length} decisions: ${decisions.map(d => `${d.survivors}s/${d.degraded ? 'degraded' : 'ok'}`).join(', ')})`)
    const verifierCalls = decisions.reduce((sum, d) => sum + d.extraUsage.calls, 0)
    assert(verifierCalls > 0, 'no verifier/extra calls were accounted')
    await copyFile(join(home, 'dsh-turbo', 'decisions.jsonl'), join(artifactDir, 'decisions.jsonl'))

    const summary = {
      schemaVersion: 1,
      status: 'passed',
      cli: DSH_CLI_SPEC,
      plugin: `dsh-turbo@${installed.version}`,
      model: 'deepseek-turbo/deepseek-v4-flash-vision-exp',
      steps: decisions.length,
      verifiedSteps: verified.length,
      degradedSteps: decisions.filter(d => d.degraded).length,
      totalExtraCalls: verifierCalls,
      totalExtraTokens: decisions.reduce((sum, d) => sum + d.extraUsage.inputTokens + d.extraUsage.outputTokens, 0),
      cacheReadTokens: decisions.reduce((sum, d) => sum + d.extraUsage.cacheReadTokens, 0),
      sampleDecision: {
        requested: verified[0].requested,
        survivors: verified[0].survivors,
        winner: verified[0].winner,
        wins: verified[0].outcome.wins,
        pivots: verified[0].outcome.pivots,
        pairs: verified[0].outcome.pairs.length,
      },
      gameFile: gameFile.split('/').pop(),
      gameBytes: game.length,
      gameFeatures: features,
      visionProof: (assistantText.match(/IMAGE SHOWS:[^\n]*/u) ?? ['(missing)'])[0].slice(0, 300),
    }
    await writeFile(join(artifactDir, 'vision-game-e2e.json'), `${JSON.stringify(summary, null, 2)}\n`)
    log('PASS — image -> best-of-5 -> verified winner -> three.js dungeon crawler')
    log(`evidence -> ${join(artifactDir, 'vision-game-e2e.json')}`)
    console.log(JSON.stringify(summary, null, 2))
  } finally {
    await browser.close()
  }
} catch (error) {
  failed = true
  console.error('[vision-game-e2e] FAIL:', error.message ?? error)
} finally {
  if (web !== undefined && web.exitCode === null) {
    web.kill('SIGTERM')
    await new Promise(done => setTimeout(done, 2_000))
    if (web.exitCode === null) web.kill('SIGKILL')
  }
  if (keep || failed) log(`kept for inspection: ${root}`)
  else await rm(root, { recursive: true, force: true }).catch(() => {})
}
process.exit(failed ? 1 : 0)
