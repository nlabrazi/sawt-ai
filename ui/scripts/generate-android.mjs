import { spawnSync } from 'node:child_process'
import { rmSync } from 'node:fs'
import { resolveMobileApiUrl } from '../config/mobile.mjs'

const apiBaseUrl = resolveMobileApiUrl(process.env.NUXT_PUBLIC_API_BASE_URL)
const prepared = spawnSync(process.execPath, ['scripts/prepare-types.mjs'], {
  cwd: new URL('..', import.meta.url),
  stdio: 'inherit',
  env: process.env,
})
if (prepared.error) throw prepared.error
if (prepared.status !== 0) process.exit(prepared.status ?? 1)
const result = spawnSync(process.execPath, ['node_modules/nuxt/bin/nuxt.mjs', 'generate'], {
  cwd: new URL('..', import.meta.url),
  stdio: 'inherit',
  env: {
    ...process.env,
    SAWT_BUILD_TARGET: 'android',
    NUXT_PUBLIC_API_BASE_URL: apiBaseUrl,
  },
})
if (result.error) throw result.error
if (result.status === 0) {
  // Release downloads belong to the website, never inside the next APK.
  rmSync(new URL('../.output-android/public/downloads', import.meta.url), {
    recursive: true,
    force: true,
  })
}
process.exitCode = result.status ?? 1
