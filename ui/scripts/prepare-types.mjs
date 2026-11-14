import { fileURLToPath } from 'node:url'
import { buildNuxt, loadNuxt, writeTypes } from '@nuxt/kit'

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
