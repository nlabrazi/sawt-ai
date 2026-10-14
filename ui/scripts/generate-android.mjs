import { spawnSync } from 'node:child_process'
import { resolveMobileApiUrl } from '../config/mobile.mjs'

const apiBaseUrl = resolveMobileApiUrl(process.env.NUXT_PUBLIC_API_BASE_URL)
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
process.exitCode = result.status ?? 1
