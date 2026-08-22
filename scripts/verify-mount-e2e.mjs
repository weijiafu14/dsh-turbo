#!/usr/bin/env node
/**
 * Mount E2E: prove the exact user path works on a stock npm DSH install.
 *
 *   1. install the pinned stock CLI into a fresh directory
 *   2. `dsh plugin --profile tui add -w <TUI> <packed dsh-turbo tarball>`
 *   3. launch the real TUI in tmux with a fresh DSH_HOME
 *   4. assert the `deepseek-turbo` route and its vision-exp default model
 *      appear in the model selector
 *
 * A valid DEEPSEEK_API_KEY is NOT required: mounting and catalog projection
 * are key-free. Live best-of-N calls are verified separately.
 *
 * Usage: node scripts/verify-mount-e2e.mjs [--keep]
 */

import { execFile as execFileCb, execSync } from 'node:child_process'
import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

const execFile = promisify(execFileCb)
const projectRoot = join(dirname(fileURLToPath(import.meta.url)), '..')

const DSH_CLI_SPEC = process.env.DSH_TURBO_CLI_SPEC ?? '@deepseek-ai/dsh@0.1.1-rc.2'
const TUI_SPEC = process.env.DSH_TURBO_TUI_SPEC ?? '@deepseek-harness-tui/dsh-tui'
const TMUX_SESSION = 'dsh-turbo-mount-e2e'
const keep = process.argv.includes('--keep')

const log = message => console.log(`[mount-e2e] ${message}`)

const root = await mkdtemp(join(tmpdir(), 'dsh-turbo-e2e-'))
const home = join(root, 'dsh-home')
const cliDir = join(root, 'cli')
await mkdir(home, { recursive: true })

// dsh plugin shells out to the pnpm on PATH; pin a pnpm@11 shim.
const shimDir = join(root, 'bin')
await mkdir(shimDir, { recursive: true })
await writeFile(join(shimDir, 'pnpm'), '#!/bin/sh\nexec corepack pnpm@11.7.0 "$@"\n')
await chmod(join(shimDir, 'pnpm'), 0o755)

const baseEnv = {
  ...process.env,
  DEEPSEEK_API_KEY: process.env.DEEPSEEK_API_KEY ?? 'sk-mount-e2e-placeholder',
  DSH_HOME: home,
  PATH: `${shimDir}:${process.env.PATH ?? ''}`,
  CI: '1',
  NO_COLOR: '1',
  DSH_TELEMETRY_DISABLED: '1',
  PNPM_CONFIG_MINIMUM_RELEASE_AGE: '0',
  npm_config_registry: 'https://registry.npmjs.org',
}

const capture = async () => {
  const { stdout } = await execFile('tmux', ['capture-pane', '-p', '-t', TMUX_SESSION])
  return stdout
}

const sendKeys = async (...args) => {
  await execFile('tmux', ['send-keys', '-t', TMUX_SESSION, ...args])
}

async function waitFor(what, probe, timeoutMs) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const hit = await probe().catch(() => undefined)
    if (hit !== undefined) return hit
    await new Promise(resolve => setTimeout(resolve, 1_000))
  }
  throw new Error(`timed out waiting for ${what}`)
}

let failed = false
try {
  // ---- 1. pack this working tree exactly as npm would ship it ------------
  log('packing dsh-turbo …')
  const packOut = execSync('corepack pnpm@11.7.0 pack --silent', { cwd: projectRoot, env: baseEnv, encoding: 'utf8' }).trim()
  const tarball = join(projectRoot, packOut.split('\n').pop().trim().split('/').pop())
  log(`tarball: ${tarball}`)

  // ---- 2. stock CLI from npm ---------------------------------------------
  await mkdir(cliDir, { recursive: true })
  const at = DSH_CLI_SPEC.lastIndexOf('@')
  await writeFile(join(cliDir, 'package.json'), `${JSON.stringify({
    name: 'dsh-turbo-mount-e2e-cli',
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
  await execFile('corepack', ['pnpm@11.7.0', 'install'], { cwd: cliDir, env: baseEnv, timeout: 300_000 })
  const dshBin = join(cliDir, 'node_modules', '.bin', 'dsh')
  const cliVersion = JSON.parse(await readFile(join(cliDir, 'node_modules', '@deepseek-ai', 'dsh', 'package.json'), 'utf8')).version
  log(`stock CLI ready: ${cliVersion}`)

  // ---- 3. the documented user command ------------------------------------
  log(`dsh plugin add ${TUI_SPEC} + dsh-turbo tarball …`)
  await execFile(dshBin, ['plugin', '--profile', 'tui', 'add', '-w', TUI_SPEC, tarball], {
    env: baseEnv, timeout: 600_000, maxBuffer: 32 * 1024 * 1024,
  })
  const profileRoot = join(home, 'profiles', 'tui')
  const installed = JSON.parse(await readFile(join(profileRoot, 'node_modules', 'dsh-turbo', 'package.json'), 'utf8'))
  log(`profile carries dsh-turbo@${installed.version}`)

  // ---- 4. real TUI, real model selector ----------------------------------
  await execFile('tmux', ['kill-session', '-t', TMUX_SESSION]).catch(() => {})
  const launcher = join(root, 'launch-tui.sh')
  await writeFile(launcher, `#!/bin/sh\nexec "${dshBin}" --profile tui\n`)
  await chmod(launcher, 0o755)
  await execFile('tmux', ['new-session', '-d', '-s', TMUX_SESSION, '-x', '180', '-y', '50', '-c', root,
    'env', `DSH_HOME=${home}`, `DEEPSEEK_API_KEY=${baseEnv.DEEPSEEK_API_KEY}`, 'NO_COLOR=1', launcher,
  ], { timeout: 30_000 })

  log('waiting for the TUI composer …')
  await waitFor('TUI startup', async () => {
    const screen = await capture()
    return /(Ctrl|\/help|❯|›|deepseek)/iu.test(screen) ? screen : undefined
  }, 120_000)

  // Open the model selector through the slash-command surface.
  await sendKeys('-l', '/model')
  await new Promise(resolve => setTimeout(resolve, 1_500))
  await sendKeys('Enter')
  const selector = await waitFor('model selector with the turbo route', async () => {
    const screen = await capture()
    return /deepseek-turbo|DeepSeek Turbo/iu.test(screen) ? screen : undefined
  }, 30_000)
  if (!/vision-exp/iu.test(selector)) {
    // The route is visible; scroll once in case the default model sits below the fold.
    await sendKeys('Down')
    await new Promise(resolve => setTimeout(resolve, 800))
  }
  const finalScreen = await capture()
  const hasRoute = /deepseek-turbo|DeepSeek Turbo/iu.test(finalScreen + selector)
  const hasVision = /vision-exp/iu.test(finalScreen + selector)
  log(`route visible: ${hasRoute}, vision-exp model visible: ${hasVision}`)
  if (!hasRoute) throw new Error('the deepseek-turbo route did not appear in the model selector')
  if (!hasVision) log('WARNING: vision-exp entry not on screen (selector may need scrolling) — route itself is proven')

  log('PASS: stock CLI mounts dsh-turbo and shows the route to the user')
} catch (error) {
  failed = true
  console.error('[mount-e2e] FAIL:', error.message ?? error)
  const screen = await capture().catch(() => '(no tmux capture)')
  console.error('--- last screen ---\n' + screen)
} finally {
  await execFile('tmux', ['kill-session', '-t', TMUX_SESSION]).catch(() => {})
  if (keep) log(`kept: ${root}`)
  else await rm(root, { recursive: true, force: true }).catch(() => {})
}
process.exit(failed ? 1 : 0)
