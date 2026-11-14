import { fileURLToPath } from 'node:url'
import { defineSharedVitestConfig } from '@nlabrazi/config-vitest'

export default defineSharedVitestConfig({
  rootDir: import.meta.url,
  environment: 'happy-dom',
  setupFiles: ['./tests/setup.ts'],
  alias: {
    '#app': fileURLToPath(new URL('./tests/mocks/nuxt-app.ts', import.meta.url)),
  },
})
