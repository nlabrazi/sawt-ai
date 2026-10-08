import { expect, type Page, test } from '@playwright/test'
import { quranContentFixture } from '../fixtures/quran-content'
import { tafsirFixture } from '../fixtures/tafsir'
import { corsHeaders, createSampleAudioBuffer, setupMockApi } from './helpers/mock-api'

const password = 'fictitious-dedicated-review-password'

async function mockReview(
  page: Page,
  conflict = false,
  rows = [tafsirFixture(), tafsirFixture('as_saadi')],
) {
  await setupMockApi(page)
  let revision = 0
  const calls: { path: string; method: string; body: Record<string, unknown> | null }[] = []
  await page.route(/\/internal\/tafsir(?:\/|\?|$)/, async (route) => {
    if (route.request().resourceType() === 'document') return route.fallback()
    const request = route.request()
    if (request.method() === 'OPTIONS') {
      return route.fulfill({
        status: 204,
        headers: { ...corsHeaders, 'Access-Control-Allow-Methods': 'GET, POST, PATCH, OPTIONS' },
      })
    }
    const url = new URL(request.url())
    calls.push({
      path: url.pathname,
      method: request.method(),
      body: request.postData() ? request.postDataJSON() : null,
    })
    const reply = (body: unknown, status = 200) =>
      route.fulfill({
        status,
        headers: corsHeaders,
        contentType: 'application/json',
        body: JSON.stringify(body),
      })
    if (request.headers().authorization !== `Bearer ${password}`)
      return reply({ detail: 'Mot de passe incorrect.' }, 401)
    if (url.pathname.endsWith('/access'))
      return route.fulfill({ status: 204, headers: corsHeaders })
    if (request.method() === 'GET') {
      return reply(
        rows.filter(
          (row) =>
            (!url.searchParams.get('source') || row.source === url.searchParams.get('source')) &&
            row.status === url.searchParams.get('status'),
        ),
      )
    }
    if (conflict) return reply({ detail: 'Version obsolète.' }, 409)
    const row = rows.find((entry) =>
      url.pathname.includes(`/${entry.surah_id}/${entry.ayah}/${entry.source}`),
    )
    if (!row) return reply({ detail: 'Unknown test reference' }, 404)
    const body = request.postDataJSON()
    if (body.expected_updated_at !== row.updated_at)
      return reply({ detail: 'Version obsolète.' }, 409)
    row.updated_at = new Date(Date.parse('2026-10-07T10:00:00Z') + ++revision * 60_000)
      .toISOString()
      .replace('.000Z', 'Z')
    if (request.method() === 'PATCH') {
      row.text_fr = body.text_fr
      row.status = 'need_review'
      row.reviewed_at = null
    } else {
      row.status = 'verified'
      row.reviewed_at = row.updated_at
    }
    return reply(row)
  })
  return calls
}

async function login(page: Page) {
  await page.getByLabel('Mot de passe interne').fill(password)
  await page.getByRole('button', { name: 'Se connecter', exact: true }).click()
  await expect(page.locator('.review-entry')).toHaveCount(2)
}

test('protects the review screen, edits then validates the exact source, and clears access on reload', async ({
  page,
}) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  const calls = await mockReview(page)
  await page.goto('/internal/tafsir')
  await expect(page.getByRole('heading', { name: 'Review des tafsirs français' })).toBeVisible()
  await expect(page.getByLabel('Mot de passe interne')).toBeVisible()
  await expect(page.locator('.review-entry')).toHaveCount(0)
  await expect(page.locator('script[src*="umami"]')).toHaveCount(0)
  await expect(page.locator('meta[name="robots"]')).toHaveAttribute('content', 'noindex, nofollow')
  expect(calls).toEqual([])
  await page.getByLabel('Mot de passe interne').fill('wrong')
  await page.getByRole('button', { name: 'Se connecter', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('Mot de passe incorrect')
  await login(page)
  const kathir = page
    .locator('.review-entry')
    .filter({ has: page.getByRole('heading', { name: /Ibn Kathir/ }) })
  const saadi = page
    .locator('.review-entry')
    .filter({ has: page.getByRole('heading', { name: /As-Sa‘di/ }) })
  await kathir.getByText('Référence et passage source', { exact: true }).click()
  await expect(kathir.locator('pre')).toContainText('<img')
  await expect(kathir.locator('img')).toHaveCount(0)
  await kathir.getByLabel('Texte français').fill('Correction fictive de test.')
  await expect(kathir.getByRole('button', { name: 'Valider', exact: true })).toBeDisabled()
  await kathir.getByRole('button', { name: 'Enregistrer', exact: true }).click()
  await expect(kathir.getByRole('button', { name: 'Valider', exact: true })).toBeEnabled()
  await kathir.getByRole('button', { name: 'Valider', exact: true }).click()
  await expect(kathir).toHaveCount(0)
  await expect(saadi).toBeVisible()
  const writes = calls.filter((call) => call.method !== 'GET')
  expect(writes.map((call) => call.path)).toEqual([
    '/internal/tafsir/2/255/ibn_kathir',
    '/internal/tafsir/2/255/ibn_kathir/verify',
  ])
  expect(writes[1]?.body).toEqual({ expected_updated_at: '2026-10-07T10:01:00Z' })
  await page.getByLabel('Statut', { exact: true }).selectOption('verified')
  await page.getByLabel('Source', { exact: true }).selectOption('ibn_kathir')
  await page.getByLabel('Sourate', { exact: true }).selectOption('2')
  await page.getByRole('button', { name: 'Filtrer', exact: true }).click()
  await expect(kathir.getByLabel('Texte français')).toHaveValue('Correction fictive de test.')
  await expect(kathir.getByRole('button', { name: 'Valider', exact: true })).toBeDisabled()
  await kathir.getByLabel('Texte français').fill('Nouvelle correction fictive.')
  await kathir.getByRole('button', { name: 'Enregistrer', exact: true }).click()
  await expect(kathir).toHaveCount(0)
  expect(
    await page.evaluate(() => JSON.stringify([localStorage, sessionStorage, document.cookie])),
  ).not.toContain(password)
  await page.reload()
  await expect(page.getByLabel('Mot de passe interne')).toBeVisible()
  await expect(page.locator('.review-entry')).toHaveCount(0)
  expect(errors).toEqual([])
})

test('requires a fresh review after a conflicting write', async ({ page }) => {
  await mockReview(page, true)
  await page.goto('/internal/tafsir')
  await login(page)
  await page
    .locator('.review-entry')
    .first()
    .getByRole('button', { name: 'Valider', exact: true })
    .click()
  await expect(page.getByRole('alert')).toContainText('Rechargez la liste puis relisez')
  for (const button of await page.getByRole('button', { name: 'Valider', exact: true }).all()) {
    await expect(button).toBeDisabled()
  }
  await page.getByRole('button', { name: 'Recharger la liste', exact: true }).click()
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(
    page.locator('.review-entry').first().getByRole('button', { name: 'Valider', exact: true }),
  ).toBeEnabled()
  await page.getByRole('button', { name: 'Se déconnecter', exact: true }).click()
  await expect(page.getByLabel('Mot de passe interne')).toBeVisible()
  await expect(page.locator('.review-entry')).toHaveCount(0)
})

test('publishes each source only after its own review and withdraws a corrected tafsir from recognition details', async ({
  page,
  context,
}) => {
  const rows = [tafsirFixture(), tafsirFixture('as_saadi')]
  await mockReview(page, false, rows)
  await page.goto('/internal/tafsir')
  await login(page)
  const kathir = page
    .locator('.review-entry')
    .filter({ has: page.getByRole('heading', { name: /Ibn Kathir/ }) })
  const saadi = page
    .locator('.review-entry')
    .filter({ has: page.getByRole('heading', { name: /As-Sa‘di/ }) })

  const publicPage = await context.newPage()
  await setupMockApi(publicPage, {
    recognizeResponse: {
      verse: {
        sourate_id: 2,
        sourate_name: 'البقرة',
        transliteration: 'Al-Baqara',
        start_verse: 255,
        end_verse: 255,
        text: 'نص عربي للاختبار',
        similarity: 0.96,
      },
      imam_predictions: [],
      imam_status: 'unknown',
      imam_detection_enabled: false,
    },
  })
  const frenchContent = quranContentFixture(2, 255, 255)
  await publicPage.route(
    (url) => url.pathname === '/quran/content',
    async (route) => {
      // The public mock reads the very same rows edited by the internal screen.
      const tafsirs = rows
        .filter((row) => row.status === 'verified')
        .map(
          ({
            surah_id,
            ayah,
            source,
            text_fr,
            source_reference,
            version,
            status,
            reviewed_at,
          }) => ({
            surah_id,
            ayah,
            source,
            text_fr,
            source_reference,
            version,
            status,
            reviewed_at,
          }),
        )
      await route.fulfill({
        status: 200,
        headers: { ...corsHeaders, 'Cache-Control': 'no-store' },
        contentType: 'application/json',
        body: JSON.stringify({
          ...frenchContent,
          ayahs: frenchContent.ayahs.map((entry) => ({ ...entry, tafsirs })),
        }),
      })
    },
  )
  await publicPage.goto('/')
  await publicPage.getByRole('button', { name: 'Coran', exact: true }).click()
  await publicPage.locator('input[type="file"]').setInputFiles({
    name: 'pilot.wav',
    mimeType: 'audio/wav',
    buffer: createSampleAudioBuffer(),
  })
  await expect(publicPage.locator('#result-title')).toBeVisible()
  const sheet = publicPage.getByRole('dialog')
  const openDetails = publicPage.getByRole('button', { name: 'Voir le verset', exact: true })
  async function readPublicDetails() {
    if (await sheet.isVisible()) {
      await sheet.getByRole('button', { name: 'Retour au résultat' }).click()
    }
    await openDetails.click()
    await expect(sheet.locator('[data-ayah="255"] .translation-text')).toHaveText(
      'Traduction fictive 2:255.',
    )
  }
  await readPublicDetails()
  await expect(sheet.locator('.tafsir-section')).toHaveCount(0)
  await expect(sheet).not.toContainText('Brouillon fictif')

  const correction = 'Correction fictive Ibn Kathir, relue pour le test.'
  await kathir.getByLabel('Texte français').fill(correction)
  await kathir.getByRole('button', { name: 'Enregistrer', exact: true }).click()
  await expect(kathir.getByRole('button', { name: 'Valider', exact: true })).toBeEnabled()
  await readPublicDetails()
  await expect(sheet.locator('.tafsir-section')).toHaveCount(0)
  await expect(sheet).not.toContainText(correction)

  await kathir.getByRole('button', { name: 'Valider', exact: true }).click()
  await expect(kathir).toHaveCount(0)
  await expect(saadi).toBeVisible()
  await readPublicDetails()
  await expect(sheet.locator('.tafsir-text')).toHaveText(correction)
  await expect(sheet.getByRole('button', { name: 'As-Sa‘di', exact: true })).toBeDisabled()
  await expect(sheet).not.toContainText(rows[1]?.text_fr ?? '')

  await saadi.getByRole('button', { name: 'Valider', exact: true }).click()
  await expect(saadi).toHaveCount(0)
  await readPublicDetails()
  await sheet.getByRole('button', { name: 'As-Sa‘di', exact: true }).click()
  await expect(sheet.locator('.tafsir-text')).toHaveText(rows[1]?.text_fr ?? '')
  await expect(sheet).not.toContainText(correction)
  await sheet.getByRole('button', { name: 'Ibn Kathir', exact: true }).click()
  await expect(sheet.locator('.tafsir-text')).toHaveText(correction)

  await page.getByLabel('Statut', { exact: true }).selectOption('verified')
  await page.getByLabel('Source', { exact: true }).selectOption('ibn_kathir')
  await page.getByRole('button', { name: 'Filtrer', exact: true }).click()
  await expect(kathir.getByLabel('Texte français')).toHaveValue(correction)
  const newDraft = 'Nouvelle correction fictive privée, à relire.'
  await kathir.getByLabel('Texte français').fill(newDraft)
  await kathir.getByRole('button', { name: 'Enregistrer', exact: true }).click()
  await expect(kathir).toHaveCount(0)
  await readPublicDetails()
  await expect(sheet.locator('.tafsir-text')).toHaveText(rows[1]?.text_fr ?? '')
  await expect(sheet.getByRole('button', { name: 'Ibn Kathir', exact: true })).toBeDisabled()
  await expect(sheet).not.toContainText(correction)
  await expect(sheet).not.toContainText(newDraft)
  await expect(sheet.locator('img')).toHaveCount(0)
  await expect(sheet).not.toContainText('Passage fictif')
  await publicPage.close()
})
