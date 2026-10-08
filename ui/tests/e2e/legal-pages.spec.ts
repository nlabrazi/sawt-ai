import { expect, test } from '@playwright/test'
import { setupMockApi } from './helpers/mock-api'

for (const document of [
  { path: '/legal-notice', title: 'Mentions légales', section: '1. Présentation du site' },
  {
    path: '/terms-of-service',
    title: 'Conditions générales d’utilisation',
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

test('opens the public documents from the footer and fits a mobile viewport', async ({ page }) => {
  await setupMockApi(page)
  await page.setViewportSize({ width: 320, height: 568 })
  await page.goto('/')
  await page.locator('.app-footer').getByRole('link', { name: 'Mentions légales' }).click()
  await expect(page).toHaveURL(/\/legal-notice$/)
  await expect(
    page.getByRole('heading', {
      name: '4. Confidentialité et protection des données',
      exact: true,
    }),
  ).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false)
  await page
    .locator('.app-footer')
    .getByRole('link', { name: 'Conditions générales d’utilisation' })
    .click()
  await expect(page).toHaveURL(/\/terms-of-service$/)
  await expect(
    page.getByRole('heading', { name: 'Conditions générales d’utilisation', exact: true }),
  ).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false)
  await page.locator('.app-footer').getByRole('link', { name: 'Mentions légales' }).click()
  await expect(page).toHaveURL(/\/legal-notice$/)
  await expect(page.locator('#legal-content')).toContainText('Nabil Labrazi')
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false)
})

for (const path of ['/privacy-policy', '/privacy-policy/']) {
  test(`redirects ${path} to the privacy section in the legal notice`, async ({ request }) => {
    const response = await request.get(path, { maxRedirects: 0 })
    expect(response.status()).toBe(301)
    expect(response.headers().location).toBe('/legal-notice#privacy')
  })
}

for (const width of [320, 375]) {
  test(`centres every footer row at ${width}px`, async ({ page }) => {
    await setupMockApi(page)
    await page.setViewportSize({ width, height: 667 })
    await page.goto('/')
    const footer = page.locator('.app-footer')
    await expect(footer.getByRole('navigation')).toBeVisible()
    const rows = await footer.locator('.footer-links a').evaluateAll((links) => {
      const grouped = new Map<number, { left: number; right: number }>()
      for (const link of links) {
        const box = link.getBoundingClientRect()
        const y = Math.round(box.top)
        const row = grouped.get(y)
        grouped.set(y, {
          left: Math.min(row?.left ?? box.left, box.left),
          right: Math.max(row?.right ?? box.right, box.right),
        })
      }
      return [...grouped.values()]
    })
    for (const row of rows) {
      expect(Math.abs((row.left + row.right) / 2 - width / 2)).toBeLessThan(2)
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false)
  })
}
