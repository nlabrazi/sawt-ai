import { expect, type Page, test } from '@playwright/test'
import { quranContentFixture } from '../fixtures/quran-content'
import {
  corsHeaders,
  createSampleAudioBuffer,
  defaultMockVerse,
  setupMockApi,
} from './helpers/mock-api'

const recognizedPassage = {
  verse: { ...defaultMockVerse, end_verse: 2 },
  transcription_text: 'بسم الله الرحمن الرحيم',
  imam_predictions: [],
  imam_status: 'unknown',
  imam_detection_enabled: true,
}

async function recognizePassage(page: Page) {
  await page.goto('/')
  await page.getByRole('button', { name: 'Coran', exact: true }).click()
  await page.locator('input[type="file"]').setInputFiles({
    name: 'passage.wav',
    mimeType: 'audio/wav',
    buffer: createSampleAudioBuffer(),
  })
  await expect(page.locator('#result-title')).toBeVisible()
}

test('loads French content on opening, separates sources, and rechecks validation on reopening', async ({
  page,
}) => {
  const response = quranContentFixture()
  const verified = response.ayahs[1]?.tafsirs[0]
  const withDraft = {
    ...response,
    ayahs: response.ayahs.map((entry) =>
      entry.ayah === 2
        ? {
            ...entry,
            tafsirs: [
              ...entry.tafsirs,
              {
                ...verified,
                source: 'ibn_kathir',
                status: 'need_review',
                reviewed_at: null,
                text_fr: 'Brouillon fictif privé',
              },
            ],
          }
        : entry,
    ),
  }
  await setupMockApi(page, {
    recognizeResponse: recognizedPassage,
    quranContentResponse: withDraft,
  })
  const contentRequests: URL[] = []
  page.on('request', (request) => {
    const url = new URL(request.url())
    if (url.pathname === '/quran/content' && request.method() === 'GET') contentRequests.push(url)
  })
  await recognizePassage(page)
  const open = page.getByRole('button', { name: 'Voir le verset', exact: true })
  await open.hover()
  expect(contentRequests).toHaveLength(0)
  await open.click()
  const sheet = page.getByRole('dialog')
  await expect(sheet.locator('.arabic-verse-text')).toHaveText(defaultMockVerse.text)
  const first = sheet.locator('[data-ayah="1"]')
  const second = sheet.locator('[data-ayah="2"]')
  await expect(first.locator('.translation-text')).toHaveText('Traduction fictive 1:1.')
  await expect(sheet.getByRole('button', { name: 'Retour au résultat' })).toBeFocused()
  await expect(second.locator('.translation-text')).toHaveText('Traduction fictive 1:2.')
  expect(contentRequests).toHaveLength(1)
  expect(Object.fromEntries(contentRequests[0]?.searchParams ?? [])).toEqual({
    surah_id: '1',
    start_verse: '1',
    end_verse: '2',
  })
  await expect(first.locator('.tafsir-text')).toHaveText('Commentaire fictif ibn_kathir 1:1.')
  await first.getByRole('button', { name: 'As-Sa‘di', exact: true }).click()
  await expect(first.locator('.tafsir-text')).toHaveText('Commentaire fictif as_saadi 1:1.')
  await expect(first.getByRole('button', { name: 'As-Sa‘di' })).toHaveAttribute(
    'aria-pressed',
    'true',
  )
  await expect(first).not.toContainText('Commentaire fictif ibn_kathir')
  await expect(second.locator('.tafsir-text')).toHaveText('Commentaire fictif as_saadi 1:2.')
  await expect(second.getByRole('button', { name: 'Ibn Kathir' })).toBeDisabled()
  await expect(sheet).not.toContainText('Brouillon fictif privé')
  await first.locator('summary').click()
  await expect(first.locator('.translation-notes p')).toHaveText(
    'Note fictive <script>test</script>.',
  )
  await expect(sheet.locator('script')).toHaveCount(0)

  await sheet.getByRole('button', { name: 'Retour au résultat' }).click()
  await page.route(
    (url) => url.pathname === '/quran/content',
    (route) =>
      route.fulfill({
        status: 200,
        headers: { ...corsHeaders, 'Cache-Control': 'no-store' },
        contentType: 'application/json',
        body: JSON.stringify({
          ...response,
          ayahs: response.ayahs.map((entry) => ({ ...entry, tafsirs: [] })),
        }),
      }),
  )
  await open.click()
  await expect(sheet.locator('[data-ayah="1"] .translation-text')).toHaveText(
    'Traduction fictive 1:1.',
  )
  expect(contentRequests).toHaveLength(2)
  await expect(sheet.locator('.tafsir-section')).toHaveCount(0)
})

test('keeps recognition, Arabic and tajwid usable when French content fails, then allows retry', async ({
  page,
}) => {
  await setupMockApi(page, { recognizeResponse: recognizedPassage, quranContentStatus: 503 })
  await recognizePassage(page)
  await page.getByRole('button', { name: 'Voir le verset', exact: true }).click()
  const sheet = page.getByRole('dialog')
  await expect(sheet.locator('.arabic-verse-text')).toHaveText(defaultMockVerse.text)
  await expect(sheet).toContainText('Impossible de charger le contenu français.')
  await sheet.getByRole('button', { name: 'Afficher le tajwid', exact: true }).click()
  await expect(sheet.locator('.tajwid-reading-card')).toBeVisible()
  await page.route(
    (url) => url.pathname === '/quran/content',
    (route) =>
      route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: 'application/json',
        body: JSON.stringify(quranContentFixture()),
      }),
  )
  await sheet.getByRole('button', { name: 'Réessayer le chargement du contenu français' }).click()
  await expect(sheet.locator('[data-ayah="1"] .translation-text')).toHaveText(
    'Traduction fictive 1:1.',
  )
  await expect(sheet.locator('.tajwid-reading-card')).toBeVisible()
  await sheet.getByRole('button', { name: 'Retour au résultat' }).click()
  await expect(page.locator('#result-title')).toBeVisible()
})

test('shows translations without a tafsir section when its storage is unavailable', async ({
  page,
}) => {
  const response = quranContentFixture()
  await setupMockApi(page, {
    recognizeResponse: recognizedPassage,
    quranContentResponse: {
      ...response,
      tafsir_status: 'unavailable',
      ayahs: response.ayahs.map((entry) => ({ ...entry, tafsirs: [] })),
    },
  })
  await recognizePassage(page)
  await page.getByRole('button', { name: 'Voir le verset', exact: true }).click()
  const sheet = page.getByRole('dialog')
  await expect(sheet.locator('[data-ayah="1"] .translation-text')).toHaveText(
    'Traduction fictive 1:1.',
  )
  await expect(sheet).toContainText('Les tafsirs sont temporairement indisponibles.')
  await expect(sheet.locator('.tafsir-section')).toHaveCount(0)
})
