import { spawnSync } from 'node:child_process'
import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

export const uiRoot = fileURLToPath(new URL('..', import.meta.url))
export const privateDirectory = join(uiRoot, '.android-release')
export const signingFile = join(privateDirectory, 'signing.json')

export function run(command, args, options = {}) {
  const result = spawnSync(command, args, { cwd: uiRoot, stdio: 'inherit', ...options })
  if (result.error) throw result.error
  if (result.status !== 0) throw new Error(`${command} failed (exit ${result.status})`)
}

export function javaTool(name) {
  return process.env.JAVA_HOME ? join(process.env.JAVA_HOME, 'bin', name) : name
}

export function signingEnvironment() {
  const local = existsSync(signingFile) ? JSON.parse(readFileSync(signingFile, 'utf8')) : {}
  const env = { ...process.env, ...local }
  for (const name of [
    'SAWT_ANDROID_KEYSTORE',
    'SAWT_ANDROID_STORE_PASSWORD',
    'SAWT_ANDROID_KEY_ALIAS',
    'SAWT_ANDROID_KEY_PASSWORD',
  ]) {
    if (process.env[name]) env[name] = process.env[name]
    if (!env[name]) throw new Error(`Missing ${name}. Run npm run android:keygen first.`)
  }
  if (!existsSync(env.SAWT_ANDROID_KEYSTORE)) throw new Error('Release keystore does not exist.')
  return env
}

export function androidVersion() {
  const source = readFileSync(join(uiRoot, 'android/version.properties'), 'utf8')
  const versionName = source.match(/^versionName=(.+)$/m)?.[1]?.trim()
  const versionCode = Number(source.match(/^versionCode=(\d+)$/m)?.[1])
  if (
    !versionName ||
    !/^[\w.-]+$/.test(versionName) ||
    !Number.isInteger(versionCode) ||
    versionCode < 1 ||
    versionCode > 2100000000
  ) {
    throw new Error('Invalid Android version.properties')
  }
  return { versionName, versionCode }
}

export function sdkTool(name) {
  const sdk = process.env.ANDROID_HOME || process.env.ANDROID_SDK_ROOT
  if (!sdk) throw new Error('Set ANDROID_HOME to the installed Android SDK directory.')
  const directory = join(sdk, 'build-tools')
  const versions = readdirSync(directory)
    .filter((version) => /^\d+\.\d+\.\d+$/.test(version))
    .sort((a, b) => b.localeCompare(a, undefined, { numeric: true }))
  const executable = versions.map((version) => join(directory, version, name)).find(existsSync)
  if (!executable) throw new Error(`Android SDK tool ${name} is missing.`)
  return executable
}

export function artifactPath() {
  return join(uiRoot, 'artifacts/android', `sawt-ai-${androidVersion().versionName}.apk`)
}

export function artifactDirectory() {
  return dirname(artifactPath())
}
