import { createHash } from 'node:crypto'
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import {
  androidVersion,
  artifactDirectory,
  artifactPath,
  run,
  sdkTool,
  signingEnvironment,
  uiRoot,
} from './android-tools.mjs'

const mode = process.argv[2] || 'release'
if (!['debug', 'release', 'bundle'].includes(mode)) throw new Error('Use debug, release or bundle.')
const env = mode === 'debug' ? process.env : signingEnvironment()
const version = androidVersion()
run(process.execPath, ['scripts/generate-android.mjs'], { env })
run(process.execPath, ['node_modules/@capacitor/cli/bin/capacitor', 'sync', 'android'], { env })
run(
  './gradlew',
  [
    mode === 'bundle'
      ? ':app:bundleRelease'
      : mode === 'debug'
        ? ':app:assembleDebug'
        : ':app:assembleRelease',
    '--console=plain',
  ],
  { cwd: join(uiRoot, 'android'), env },
)
if (mode === 'release') {
  const source = join(uiRoot, 'android/app/build/outputs/apk/release/app-release.apk')
  run(sdkTool('apksigner'), ['verify', '--verbose', '--print-certs', source], { env })
  mkdirSync(artifactDirectory(), { recursive: true })
  copyFileSync(source, artifactPath())
  const sha256 = createHash('sha256').update(readFileSync(artifactPath())).digest('hex')
  writeFileSync(`${artifactPath()}.sha256`, `${sha256}  sawt-ai-${version.versionName}.apk\n`)
  console.log(`Signed APK: ${artifactPath()}`)
}
