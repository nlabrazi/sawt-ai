import { defineSharedPlaywrightConfig } from '@nlabrazi/config-playwright'

export default defineSharedPlaywrightConfig({
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: [['list'], ['html', { open: 'never', outputFolder: 'playwright-report' }]],
  use: {
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  env: {
    NUXT_PUBLIC_WEB3FORMS_ACCESS_KEY: 'web3forms-e2e-test-key',
    NUXT_PUBLIC_ANDROID_DOWNLOAD_URL: '/downloads/sawt-ai-android.apk',
    SAWT_NUXT_BUILD_DIR: '.nuxt-e2e',
  },
})
