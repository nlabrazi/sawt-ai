import { defineSharedPlaywrightConfig } from '@nlabrazi/config-playwright'

export default defineSharedPlaywrightConfig({
  useBuiltServerInCI: true,
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 4 : undefined,
  reporter: [['list'], ['html', { open: 'never', outputFolder: 'playwright-report' }]],
  use: {
    screenshot: 'only-on-failure',
    video: 'on-first-retry',
  },
  env: {
    NUXT_PUBLIC_WEB3FORMS_ACCESS_KEY: 'web3forms-e2e-test-key',
    NUXT_PUBLIC_ANDROID_DOWNLOAD_URL: '/downloads/sawt-ai-android.apk',
    SAWT_NUXT_BUILD_DIR: '.nuxt-e2e',
    SAWT_DISABLE_DEVTOOLS: 'true',
  },
})
