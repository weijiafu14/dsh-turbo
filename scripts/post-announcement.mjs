#!/usr/bin/env node
/**
 * Publish the posting kit's DSH community post as a GitHub Discussion under
 * "Show Your Plugins!". Relative asset paths are rewritten to raw.githubusercontent
 * URLs so the screenshots render, and the kit's header block (category/title/attach)
 * is stripped from the body.
 *
 * Usage: node scripts/post-announcement.mjs [--dry-run]
 */
import { execFile as execFileCb } from 'node:child_process'
import { readFile, writeFile } from 'node:fs/promises'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

const execFile = promisify(execFileCb)
const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const RAW = 'https://raw.githubusercontent.com/weijiafu14/dsh-turbo/main/docs/posting-kit/assets'
const REPO_ID = 'R_kgDOT3T1gw'                 // deepseek-ai/deepseek-harness
const CATEGORY_ID = 'DIC_kwDOT3T1g84DDSUe'     // Show Your Plugins!
const TITLE = 'dsh-turbo — best-of-N with an LLM-as-a-Verifier judge, as one model route'

const src = await readFile(join(root, 'docs/posting-kit/post-dsh-community.md'), 'utf8')
const body = src.split('---')[1].replace(/^\n+/, '').replaceAll('](assets/', `](${RAW}/`)
const bodyPath = join(root, '.artifacts', 'discussion-body.md')
await writeFile(bodyPath, body)

if (process.argv.includes('--dry-run')) {
  console.log(body)
  console.log(`\n--- dry run; body written to ${bodyPath}`)
  process.exit(0)
}

const query = `mutation($repoId:ID!,$categoryId:ID!,$title:String!,$body:String!){
  createDiscussion(input:{repositoryId:$repoId, categoryId:$categoryId, title:$title, body:$body}){
    discussion { url number }
  }
}`
const { stdout } = await execFile('gh', [
  'api', 'graphql',
  '-F', `body=@${bodyPath}`,
  '-f', `repoId=${REPO_ID}`,
  '-f', `categoryId=${CATEGORY_ID}`,
  '-f', `title=${TITLE}`,
  '-f', `query=${query}`,
], { maxBuffer: 8 * 1024 * 1024 })
const d = JSON.parse(stdout).data.createDiscussion.discussion
console.log(`published: ${d.url}`)
