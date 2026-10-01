import { expect, test } from '@playwright/test'
import { hadithFixture } from '../fixtures/hadith'
import { corsHeaders, setupMockApi } from './helpers/mock-api'

test.beforeEach(async ({ page }) => {
  await setupMockApi(page)
  await page.goto('/')
  await page.getByRole('button', { name: 'Hadiths', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Retrouvez un hadith' })).toBeVisible()
})

test('searches, reads the source text and restores focus after Escape', async ({ page }) => {
  const query = page.getByLabel('Que recherchez-vous ?')
  await query.fill('la colère')
  const request = page.waitForRequest(
    (req) => req.url().endsWith('/hadith/search') && req.method() === 'POST',
  )
  await query.press('Enter')
  expect((await request).postDataJSON()).toEqual({ query: 'la colère', limit: 3 })
  await expect(page.getByRole('heading', { name: 'Hadiths proposés' })).toBeVisible()
  const read = page.getByRole('button', { name: /Lire le hadith/ })
  await read.click()
  const dialog = page.getByRole('dialog')
  await expect(dialog).toBeVisible()
  await expect(dialog.locator('[lang="ar"][dir="rtl"]')).toHaveText(hadithFixture.arabic)
  await expect(dialog.getByRole('heading', { name: 'Traduction française' })).toBeVisible()
  await expect(dialog.getByRole('heading', { name: 'Explication' })).toBeVisible()
  await expect(dialog.getByRole('link', { name: /Consulter HadeethEnc/ })).toHaveAttribute(
    'href',
    hadithFixture.source_url,
  )
  // Native modal focus stays within the reading panel.
  await page.keyboard.press('Tab')
  expect(await dialog.evaluate((node) => node.contains(document.activeElement))).toBe(true)
  await page.keyboard.press('Escape')
  await expect(dialog).not.toBeVisible()
  await expect(read).toBeFocused()
})

test('keeps the query and proposals after returning from Quran mode', async ({ page }) => {
  const query = page.getByLabel('Que recherchez-vous ?')
  await query.fill('les intentions')
  await page.getByRole('button', { name: 'Rechercher', exact: true }).click()
  await expect(page.locator('.hadith-card')).toBeVisible()
  await page.getByRole('button', { name: 'Coran', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Récitez un passage du Coran' })).toBeVisible()
  await page.getByRole('button', { name: 'Hadiths', exact: true }).click()
  await expect(query).toHaveValue('les intentions')
  await expect(page.locator('.hadith-card')).toBeVisible()
  await query.fill('la miséricorde')
  await query.press('Enter')
  await expect(page.locator('.results-heading')).toContainText('la miséricorde')
})

test('fills examples without submitting and validates a short query', async ({ page }) => {
  let requests = 0
  page.on('request', (request) => {
    if (request.url().endsWith('/hadith/search') && request.method() === 'POST') requests += 1
  })
  await page.getByRole('button', { name: 'Ne pas se mettre en colère', exact: true }).click()
  const query = page.getByLabel('Que recherchez-vous ?')
  await expect(query).toHaveValue('Ne pas se mettre en colère')
  await expect(query).toBeFocused()
  await query.fill('ab')
  await query.press('Enter')
  await expect(page.getByRole('alert')).toContainText('3 caractères')
  expect(requests).toBe(0)
})

test('recovers from service unavailability and distinguishes an empty list', async ({ page }) => {
  await setupMockApi(page, { hadithStatus: 503 })
  await page.getByLabel('Que recherchez-vous ?').fill('la colère')
  await page.getByRole('button', { name: 'Rechercher', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('temporairement indisponible')
  await setupMockApi(page, {
    hadithResponse: { query: 'la colère', results: [], search_mode: 'semantic', search_terms: [] },
  })
  await page.getByRole('button', { name: 'Réessayer', exact: true }).click()
  await expect(page.locator('.empty-state')).toContainText('Aucun résultat exploitable')
  await expect(page.getByRole('alert')).not.toBeVisible()
})

for (const query of ['couronne', 'hadith couronne']) {
  test(`explains the absence of keyword results for ${query}`, async ({ page }) => {
    await setupMockApi(page, {
      hadithResponse: { query, results: [], search_mode: 'keywords', search_terms: ['couronne'] },
    })
    await page.getByLabel('Que recherchez-vous ?').fill(query)
    await page.getByRole('button', { name: 'Rechercher', exact: true }).click()
    await expect(page.locator('.search-method')).toContainText(
      'Recherche par mots-clés : « couronne »',
    )
    await expect(page.locator('.empty-state')).toContainText('Aucun résultat pour ces mots-clés')
    await expect(page.locator('.hadith-card')).toHaveCount(0)
    await expect(page.getByRole('alert')).not.toBeVisible()
    await expect(page.getByRole('heading', { name: 'Hadiths proposés' })).toBeFocused()
    await expect(
      page.getByRole('link', { name: /Consulter la collection HadeethEnc/ }),
    ).toHaveAttribute('href', 'https://hadeethenc.com/fr')
  })
}

test('cancels a pending request when changing mode and ignores its late response', async ({
  page,
}) => {
  let release!: () => void
  let sent!: () => void
  const pending = new Promise<void>((resolve) => {
    release = resolve
  })
  const fulfilled = new Promise<void>((resolve) => {
    sent = resolve
  })
  await page.route('**/hadith/search', async (route) => {
    if (route.request().method() === 'OPTIONS') {
      await route.fulfill({ status: 204, headers: corsHeaders })
      return
    }
    await pending
    await route.fulfill({
      status: 200,
      headers: corsHeaders,
      contentType: 'application/json',
      body: JSON.stringify({ query: 'la colère', results: [hadithFixture] }),
    })
    sent()
  })
  await page.getByLabel('Que recherchez-vous ?').fill('la colère')
  const started = page.waitForRequest(
    (request) => request.url().endsWith('/hadith/search') && request.method() === 'POST',
  )
  await page.getByRole('button', { name: 'Rechercher', exact: true }).click()
  await started
  await expect(page.getByRole('button', { name: 'Annuler', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Coran', exact: true }).click()
  release()
  await fulfilled
  await expect(page.getByRole('heading', { name: 'Récitez un passage du Coran' })).toBeVisible()
  await page.getByRole('button', { name: 'Hadiths', exact: true }).click()
  await expect(page.getByLabel('Que recherchez-vous ?')).toHaveValue('la colère')
  await expect(page.locator('.hadith-card')).not.toBeVisible()
  await expect(page.locator('.loading-panel')).not.toBeVisible()
})

test('supports reading on mobile without horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 })
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.getByLabel('Que recherchez-vous ?').fill('la colère')
  await page.getByRole('button', { name: 'Rechercher', exact: true }).click()
  await page.getByRole('button', { name: /Lire le hadith/ }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
    ),
  ).toBe(false)
  await page.getByRole('button', { name: 'Fermer la lecture du hadith' }).click()
  await expect(page.getByRole('dialog')).not.toBeVisible()
})
