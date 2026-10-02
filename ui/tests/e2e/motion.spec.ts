import { expect, test } from '@playwright/test'
import { setupMockApi } from './helpers/mock-api'

test('stops decorative motion when the device preference changes', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  page.on('console', (message) => {
    if (/hydration/i.test(message.text())) errors.push(message.text())
  })
  await setupMockApi(page)
  await page.goto('/')
  await expect(page.getByRole('button', { name: 'Explorer le Coran', exact: true })).toBeEnabled()
  const orbit = page.locator('.signal-orbit')
  const initial = await orbit.evaluate((node) => getComputedStyle(node).transform)
  await expect
    .poll(() => orbit.evaluate((node) => getComputedStyle(node).transform))
    .not.toBe(initial)

  await page.emulateMedia({ reducedMotion: 'reduce' })
  await expect(orbit).toHaveCSS('transform', 'none')
  await expect
    .poll(() =>
      page.evaluate(() => document.getAnimations().filter((a) => a.playState === 'running').length),
    )
    .toBe(0)
  await page.getByRole('button', { name: 'Explorer le Coran', exact: true }).click()
  await expect(page.locator('#recognition-title')).toBeVisible()
  await expect(page.locator('.button-orbit')).toHaveCSS('transform', 'none')
  expect(errors).toEqual([])
})

for (const width of [320, 375, 1440]) {
  test(`keeps the animated landing within a ${width}px viewport`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 })
    await setupMockApi(page)
    await page.goto('/')
    await expect(page.getByRole('button', { name: 'Explorer le Coran', exact: true })).toBeEnabled()
    await expect(page.locator('#landing-title')).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false)
  })
}

test.describe('Server-rendered landing', () => {
  test.use({ javaScriptEnabled: false })

  test('keeps the introduction visible before hydration', async ({ page }) => {
    await setupMockApi(page)
    await page.goto('/')
    await expect(page.locator('#landing-title')).toBeVisible()
    await expect(page.locator('.landing-description')).toBeVisible()
    await expect(page.locator('.landing-actions')).toHaveCSS('opacity', '1')
  })
})
