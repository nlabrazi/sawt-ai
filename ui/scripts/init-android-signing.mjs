import { randomBytes } from 'node:crypto'
import { chmodSync, existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { javaTool, privateDirectory, run, signingFile } from './android-tools.mjs'

const keystore = join(privateDirectory, 'release.jks')
if (existsSync(signingFile) || existsSync(keystore)) {
  throw new Error('Signing material already exists; it will not be overwritten.')
}
mkdirSync(privateDirectory, { recursive: true, mode: 0o700 })
chmodSync(privateDirectory, 0o700)
const password = randomBytes(32).toString('hex')
const signing = {
  SAWT_ANDROID_KEYSTORE: keystore,
  SAWT_ANDROID_STORE_PASSWORD: password,
  SAWT_ANDROID_KEY_ALIAS: 'sawt-ai',
  SAWT_ANDROID_KEY_PASSWORD: password,
}
run(
  javaTool('keytool'),
  [
    '-genkeypair',
    '-keystore',
    keystore,
    '-storetype',
    'PKCS12',
    '-storepass:env',
    'SAWT_ANDROID_STORE_PASSWORD',
    '-keypass:env',
    'SAWT_ANDROID_KEY_PASSWORD',
    '-alias',
    'sawt-ai',
    '-keyalg',
    'RSA',
    '-keysize',
    '4096',
    '-validity',
    '10000',
    '-dname',
    'CN=Sawt AI',
  ],
  { env: { ...process.env, ...signing } },
)
chmodSync(keystore, 0o600)
writeFileSync(signingFile, `${JSON.stringify(signing, null, 2)}\n`, { mode: 0o600, flag: 'wx' })
console.log(
  'Release signing created in ui/.android-release. Back up this private directory securely; it is never committed or published.',
)
