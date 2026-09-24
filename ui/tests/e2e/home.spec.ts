import { expect, test } from '@playwright/test'
import { defaultMockHealth, setupMockApi } from './helpers/mock-api'

test.describe('Home / Idle Screen', () => {
  test('renders the landing page with branding, titles, guidance and actions', async ({ page }) => {
    await setupMockApi(page)
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Brand header
    const brand = page.locator('header.brand')
    await expect(brand).toBeVisible()
    await expect(brand).toContainText('Sawt')
    await expect(brand).toContainText('AI')

    // State label and title
    await expect(page.locator('.state-label')).toContainText('Reconnaissance coranique')
    const title = page.locator('#recognition-title')
    await expect(title).toBeVisible()
    await expect(title).toContainText('Récitez un passage du Coran')

    const subtitle = page.locator('#recognition-guidance')
    await expect(subtitle).toBeVisible()
    await expect(subtitle).toContainText(
      'Sawt AI vous propose la sourate et les versets correspondants.',
    )

    // Action buttons
    const micButton = page.getByRole('button', { name: /Commencer/i })
    await expect(micButton).toBeVisible()
    await expect(micButton).toHaveAttribute('type', 'button')

    const uploadButton = page.getByRole('button', { name: 'Importer un fichier audio' })
    await expect(uploadButton).toBeVisible()

    const fileInput = page.locator('input[type="file"]')
    await expect(fileInput).toBeAttached()

    // Detect imam toggle inside options details
    const optionsSummary = page.locator('summary', { hasText: 'Options de reconnaissance' })
    await expect(optionsSummary).toBeVisible()
    await optionsSummary.click()

    const imamToggle = page.locator('input[type="checkbox"]')
    await expect(imamToggle).toBeVisible()
    await expect(imamToggle).not.toBeChecked()

    // Footer
    const footer = page.locator('footer.app-footer')
    await expect(footer).toBeVisible()
    await expect(footer).toContainText('Sawt AI Tous droits réservés')
    await expect(footer.getByRole('link', { name: 'GitHub' })).toBeVisible()
    await expect(footer.getByRole('link', { name: 'Portfolio' })).toBeVisible()
    await expect(footer.getByRole('link', { name: 'Contact' })).toBeVisible()
  })

  test('displays dynamic upload policy hints from API health endpoint', async ({ page }) => {
    const customHealth = {
      ...defaultMockHealth,
      services: {
        ...defaultMockHealth.services,
        upload_policy: {
          max_file_size_bytes: 5 * 1024 * 1024,
          max_audio_duration_seconds: 15,
          accepted_mime_types: ['audio/wav', 'audio/mpeg'],
          accepted_file_extensions: ['.wav', '.mp3'],
        },
      },
    }

    await setupMockApi(page, { health: customHealth })
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const uploadHint = page.locator('.upload-hint')
    await expect(uploadHint).toBeVisible()
    await expect(uploadHint).toContainText('max 5 Mo')
    await expect(uploadHint).toContainText('max 15 sec')
    await expect(uploadHint).toContainText('wav, mp3')
  })

  test('disables imam detection option when backend marks it unavailable', async ({ page }) => {
    const customHealth = {
      ...defaultMockHealth,
      services: {
        ...defaultMockHealth.services,
        imam_detection: {
          available: false,
          status: 'unavailable' as const,
          message: 'La reconnaissance de l’imam est temporairement indisponible.',
        },
      },
    }

    await setupMockApi(page, { health: customHealth })
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Open options details
    const optionsSummary = page.locator('summary', { hasText: 'Options de reconnaissance' })
    await expect(optionsSummary).toBeVisible()
    await optionsSummary.click()

    const imamCheckbox = page.locator('input[type="checkbox"]')
    await expect(imamCheckbox).toBeDisabled()

    const unavailableNotice = page.locator('.imam-toggle-hint.is-unavailable')
    await expect(unavailableNotice).toBeVisible()
    await expect(unavailableNotice).toContainText('temporairement indisponible')
  })

  test('renders properly on mobile viewport without horizontal overflow', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    await setupMockApi(page)
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const title = page.locator('#recognition-title')
    await expect(title).toBeVisible()

    const micButton = page.getByRole('button', { name: /Commencer/i })
    await expect(micButton).toBeVisible()

    const uploadButton = page.getByRole('button', { name: 'Importer un fichier audio' })
    await expect(uploadButton).toBeVisible()

    // Verify horizontal scroll width equals client width (no overflow)
    const hasHorizontalOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth
    })
    expect(hasHorizontalOverflow).toBe(false)
  })
})
