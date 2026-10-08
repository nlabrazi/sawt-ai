import { expect, test } from '@playwright/test'
import { setupMockApi } from './helpers/mock-api'

for (const document of [
  { path: '/privacy-policy', title: 'Politique de confidentialité', section: '5. Conservation' },
  {
    path: '/terms-of-service',
    title: 'Conditions d’utilisation',
    section: '3. Traductions, tafsirs et sources',
  },
]) {
  test(`serves ${document.path} without JavaScript, with its own metadata and return link`, async ({
    browser,
    baseURL,
  }) => {
    const context = await browser.newContext({ baseURL, javaScriptEnabled: false })
    const page = await context.newPage()
    await setupMockApi(page)
    const response = await page.goto(document.path)
    expect(response?.status()).toBe(200)
    await expect(page.getByRole('heading', { name: document.title, exact: true })).toBeVisible()
    await expect(page.getByRole('heading', { name: document.section, exact: true })).toBeVisible()
    await expect(page.locator('#legal-content')).toContainText('OVHcloud')
    await expect(page.locator('#legal-content')).toContainText('Allemagne')
    await expect(page.locator('#legal-content')).not.toContainText(
      /À renseigner|à renseigner|Version à compléter/,
    )
    await expect(page).toHaveTitle(`${document.title} — Sawt AI`)
    await expect(page.locator('link[rel="canonical"]')).toHaveCount(1)
    await expect(page.locator('link[rel="canonical"]')).toHaveAttribute(
      'href',
      new RegExp(`${document.path}$`),
    )
    await expect(page.locator('meta[property="og:url"]')).toHaveCount(1)
    await expect(page.locator('meta[property="og:url"]')).toHaveAttribute(
      'content',
      new RegExp(`${document.path}$`),
    )
    await expect(
      page.getByRole('navigation', { name: 'Documents légaux' }).locator('[aria-current="page"]'),
    ).toHaveAttribute('href', document.path)
    await page.getByRole('link', { name: '← Retour à Sawt AI' }).click()
    await expect(page.locator('#landing-title')).toBeVisible()
    await context.close()
  })
}

test('opens both documents from the footer and fits a mobile viewport', async ({ page }) => {
  await setupMockApi(page)
  await page.setViewportSize({ width: 320, height: 568 })
  await page.goto('/')
  await page.locator('.app-footer').getByRole('link', { name: 'Confidentialité' }).click()
  await expect(page).toHaveURL(/\/privacy-policy$/)
  await expect(
    page.getByRole('heading', { name: 'Politique de confidentialité', exact: true }),
  ).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false)
  await page.locator('.app-footer').getByRole('link', { name: 'Conditions d’utilisation' }).click()
  await expect(page).toHaveURL(/\/terms-of-service$/)
  await expect(
    page.getByRole('heading', { name: 'Conditions d’utilisation', exact: true }),
  ).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false)
})
