import { expect, test } from '@playwright/test'
import { createSampleAudioBuffer, setupMockApi } from './helpers/mock-api'

test.describe('Recognition Flow', () => {
  test('completes full recognition flow with audio upload, result card, and reset', async ({
    page,
  }) => {
    await setupMockApi(page)
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const audioBuffer = createSampleAudioBuffer()
    const fileInput = page.locator('input[type="file"]')

    await fileInput.setInputFiles({
      name: 'al-fatiha.wav',
      mimeType: 'audio/wav',
      buffer: audioBuffer,
    })

    // Wait for the result screen
    const resultHeading = page.locator('#result-title')
    await expect(resultHeading).toBeVisible({ timeout: 15_000 })
    await expect(resultHeading).toContainText('Passage proposé')

    // Result card details
    const surahName = page.locator('.surah-arabic')
    await expect(surahName).toBeVisible()
    await expect(surahName).toContainText('Al-Fatiha')

    const verseRange = page.locator('.verse-range')
    await expect(verseRange).toBeVisible()
    await expect(verseRange).toContainText('Versets 1 à 7')

    // Action button to view details
    const viewVerseButton = page.getByRole('button', { name: /Voir le verset/i })
    await expect(viewVerseButton).toBeVisible()

    // Reset button returns to idle screen
    const resetButton = page.locator('button.reset-action')
    await expect(resetButton).toBeVisible()
    await resetButton.click()

    await expect(page.locator('#recognition-title')).toBeVisible()
    await expect(page.locator('#recognition-title')).toContainText('Récitez un passage du Coran')
  })

  test('displays appropriate guidance when recitation is rejected as insufficient', async ({
    page,
  }) => {
    const rejectionResponse = {
      transcription_text: '',
      verse: null,
      detection: {
        status: 'insufficient',
        score: null,
        score_margin: null,
        matched_word_count: 0,
        analyzed_duration_seconds: 2,
        analysis_attempts: 1,
        rejection_reason: 'insufficient_speech',
      },
      imam_predictions: [],
      imam_status: 'disabled',
      imam_detection_enabled: false,
    }

    await setupMockApi(page, { recognizeResponse: rejectionResponse })
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const audioBuffer = createSampleAudioBuffer()
    const fileInput = page.locator('input[type="file"]')

    await fileInput.setInputFiles({
      name: 'short.wav',
      mimeType: 'audio/wav',
      buffer: audioBuffer,
    })

    const resultHeading = page.locator('#result-title')
    await expect(resultHeading).toBeVisible({ timeout: 15_000 })
    await expect(resultHeading).toContainText('Récitation trop courte')

    const subtitle = page.locator('.main-subtitle')
    await expect(subtitle).toContainText('Récitez distinctement pendant quelques secondes')

    // Reset back to idle screen
    await page.locator('button.reset-action').click()
    await expect(page.locator('#recognition-title')).toContainText('Récitez un passage du Coran')
  })

  test('displays server error detail when recognition API returns error response', async ({
    page,
  }) => {
    const errorDetail = 'Format audio invalide ou non pris en charge.'
    await setupMockApi(page, {
      recognizeStatus: 400,
      recognizeErrorDetail: errorDetail,
    })
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const audioBuffer = createSampleAudioBuffer()
    const fileInput = page.locator('input[type="file"]')

    await fileInput.setInputFiles({
      name: 'corrupted.wav',
      mimeType: 'audio/wav',
      buffer: audioBuffer,
    })

    const resultHeading = page.locator('#result-title')
    await expect(resultHeading).toBeVisible({ timeout: 15_000 })
    await expect(resultHeading).toContainText('Analyse interrompue')

    const statusPanel = page.locator('.status-panel')
    await expect(statusPanel).toBeVisible()
    await expect(statusPanel).toContainText(errorDetail)

    // Reset back to idle screen
    await page.locator('button.reset-action').click()
    await expect(page.locator('#recognition-title')).toContainText('Récitez un passage du Coran')
  })

  test('rejects unsupported file type directly in the client without API call', async ({
    page,
  }) => {
    let apiCalled = false
    await setupMockApi(page)
    await page.route('**/recognize', async (route) => {
      apiCalled = true
      await route.abort()
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles({
      name: 'test.pdf',
      mimeType: 'application/pdf',
      buffer: Buffer.from('%PDF-1.4 dummy'),
    })

    // Should display client validation error
    const uploadError = page.locator('.status-message.is-error, .status-title')
    await expect(uploadError.first()).toBeVisible()
    await expect(uploadError.first()).toContainText('Format audio non pris en charge')
    expect(apiCalled).toBe(false)
  })
})
