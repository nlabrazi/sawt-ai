import { expect, test } from '@playwright/test'

test('offers the Android beta download without overflowing a narrow screen', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 740 })
  await page.goto('/')
  const download = page.getByRole('link', { name: 'Télécharger l’application Android Bêta' })
  await expect(download).toBeVisible()
  await expect(download).toHaveAttribute('href', '/downloads/sawt-ai-android.apk')
  await page.getByText('Comment installer ?', { exact: true }).click()
  await expect(page.getByText('Téléchargez l’APK, ouvrez-le', { exact: false })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )
  await page.screenshot({ path: 'test-results/android-download-mobile.png', fullPage: true })
})
