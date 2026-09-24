import { expect, test } from '@playwright/test'
import { createSampleAudioBuffer, setupMockApi } from './helpers/mock-api'

test.describe('Verse Details and Feedback', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockApi(page)
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles({
      name: 'al-fatiha.wav',
      mimeType: 'audio/wav',
      buffer: createSampleAudioBuffer(),
    })

    // Wait for the result screen to appear
    await expect(page.locator('#result-title')).toBeVisible({ timeout: 15_000 })
  })

  test('opens verse details sheet, toggles tajwid view, and closes sheet', async ({ page }) => {
    const viewVerseButton = page.getByRole('button', { name: /Voir le verset/i })
    await expect(viewVerseButton).toBeVisible()
    await viewVerseButton.click()

    // Sheet should be opened
    const sheet = page.getByRole('dialog')
    await expect(sheet).toBeVisible({ timeout: 15_000 })
    await expect(sheet.locator('.sheet-title')).toContainText('Al-Fatiha')

    // Click toggle tajwid
    const toggleTajwidBtn = sheet.getByRole('button', { name: /tajwid/i })
    await expect(toggleTajwidBtn).toBeVisible()
    await toggleTajwidBtn.click()

    // Tajwid reading panel should appear
    const tajwidSection = sheet.locator('.tajwid-reading-card')
    await expect(tajwidSection).toBeVisible({ timeout: 10_000 })
    await expect(tajwidSection).toContainText('Affichage tajwid')

    // Close using close button
    const closeBtn = sheet.getByRole('button', { name: /Retour au résultat/i })
    await expect(closeBtn).toBeVisible()
    await closeBtn.click()
    await expect(sheet).not.toBeVisible()
  })

  test('submits positive feedback and shows confirmation toast', async ({ page }) => {
    let feedbackSubmitted = false
    await page.route(
      (url) => url.pathname === '/feedback',
      async (route) => {
        feedbackSubmitted = true
        await route.fulfill({
          status: 200,
          headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': '*',
          },
          contentType: 'application/json',
          body: JSON.stringify({ status: 'ok' }),
        })
      },
    )

    const positiveButton = page.locator('button.feedback-action-primary')
    await expect(positiveButton).toBeVisible()
    await positiveButton.click()

    // Toast message should appear
    const toast = page.locator('.mini-toast')
    await expect(toast).toBeVisible()
    await expect(toast).toContainText('Retour envoyé, merci de votre contribution !')
    expect(feedbackSubmitted).toBe(true)
  })

  test('opens correction form on negative feedback and allows submitting report', async ({
    page,
  }) => {
    let feedbackSubmitted = false
    await page.route(
      (url) => url.pathname === '/feedback',
      async (route) => {
        feedbackSubmitted = true
        await route.fulfill({
          status: 200,
          headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': '*',
          },
          contentType: 'application/json',
          body: JSON.stringify({ status: 'ok' }),
        })
      },
    )

    const negativeButton = page.locator('button.feedback-action-secondary')
    await expect(negativeButton).toBeVisible()
    await negativeButton.click()

    // Correction panel should open
    const correctionPanel = page.locator('.correction-panel')
    await expect(correctionPanel).toBeVisible()
    await expect(correctionPanel.locator('.panel-title')).toContainText('Correction')

    // Select surah
    const surahSelect = correctionPanel.locator('select#sourate')
    await expect(surahSelect).toBeVisible()
    await surahSelect.selectOption('1')

    // Fill in correction comments
    const commentField = correctionPanel.locator('textarea#comment')
    await expect(commentField).toBeVisible()
    await commentField.fill('Le passage correspond à la sourate Al-Fatiha.')

    // Submit correction
    const submitBtn = correctionPanel.locator('button.submit-btn')
    await expect(submitBtn).toBeEnabled()
    await submitBtn.click()

    // Toast confirmation
    const toast = page.locator('.mini-toast')
    await expect(toast).toBeVisible()
    await expect(toast).toContainText('Retour envoyé, merci de votre contribution !')
    expect(feedbackSubmitted).toBe(true)
  })
})
