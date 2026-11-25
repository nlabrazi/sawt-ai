import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { buildNuxt, loadNuxt, writeTypes } from '@nuxt/kit'

// In CI, Playwright runs against the pre-built Nitro server (.output/server/index.mjs)
// produced by `npm run build`, so preparing dev types is unnecessary.
if (process.env.CI && existsSync(new URL('../.output/server/index.mjs', import.meta.url))) {
  process.exit(0)
}

// Vue's production transform reads the root tsconfig project references even
// when Vite uses another buildDir. Generate those types without clearing dist.
const nuxt = await loadNuxt({
  cwd: fileURLToPath(new URL('..', import.meta.url)),
  overrides: { _prepare: true, buildDir: '.nuxt', ssr: true },
})
try {
  await buildNuxt(nuxt)
  await writeTypes(nuxt)
} finally {
  await nuxt.close()
}
