import { createHash } from 'node:crypto'
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { basename, join } from 'node:path'
import { androidVersion, artifactPath, run, sdkTool, uiRoot } from './android-tools.mjs'

const apk = artifactPath()
run(sdkTool('apksigner'), ['verify', '--verbose', apk])
const sha256 = createHash('sha256').update(readFileSync(apk)).digest('hex')
const expected = readFileSync(`${apk}.sha256`, 'utf8').split(/\s/)[0]
if (sha256 !== expected) throw new Error('The APK checksum no longer matches the verified release.')
const directory = join(uiRoot, 'public/downloads')
mkdirSync(directory, { recursive: true })
copyFileSync(apk, join(directory, basename(apk)))
copyFileSync(apk, join(directory, 'sawt-ai-android.apk'))
writeFileSync(
  join(directory, 'android-release.json'),
  `${JSON.stringify({ ...androidVersion(), sha256, url: `/downloads/${basename(apk)}` }, null, 2)}\n`,
)
writeFileSync(join(directory, `${basename(apk)}.sha256`), `${sha256}  ${basename(apk)}\n`)
console.log('Verified Android downloads prepared in ui/public/downloads.')
