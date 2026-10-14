import { expect, test } from '@playwright/test'

test('offers the Android beta download without overflowing a narrow screen', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 740 })
  await page.goto('/')
  const download = page.getByRole('link', {
    name: 'Télécharger l’application Android',
    exact: true,
  })
  await expect(download).toBeVisible()
  await expect(download).toHaveAttribute('href', '/downloads/sawt-ai-android.apk')
  await expect(page.getByText('Bêta', { exact: true })).toBeVisible()
  await expect(page.getByText('Comment installer ?', { exact: true })).toHaveCount(0)
  await expect(
    page.getByRole('button', { name: 'Identifier un verset', exact: true }),
  ).toBeEnabled()
  await page.evaluate(() => document.fonts.ready.then(() => undefined))
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )
  await page.screenshot({ path: 'test-results/android-download-mobile.png', fullPage: true })
})
