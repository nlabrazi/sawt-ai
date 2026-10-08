import { expect, test } from '@playwright/test'
import { createSampleAudioBuffer, setupMockApi } from './helpers/mock-api'

test.beforeEach(async ({ page }) => {
  await setupMockApi(page)
  await page.goto('/')
})

test('presents the product and opens both tools from the landing CTAs', async ({ page }) => {
  await expect(
    page.getByRole('heading', { name: 'Identifier un verset. Rechercher un hadith.' }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Identifier un verset', exact: true }).click()
  await expect(page.locator('#recognition-title')).toBeVisible()
  await page.getByRole('button', { name: 'Sawt AI — Accueil et réinitialisation' }).click()
  await expect(page.locator('#landing-title')).toBeVisible()
  await page.getByRole('button', { name: 'Rechercher un hadith', exact: true }).click()
  await expect(page.locator('#hadith-title')).toBeVisible()
})

test('opens the FAQ and expands answers with the keyboard', async ({ page }) => {
  await page.getByRole('button', { name: 'FAQ', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Questions fréquentes' })).toBeVisible()
  const question = page.locator('summary', { hasText: 'D’où viennent les hadiths ?' })
  await question.focus()
  await page.keyboard.press('Enter')
  await expect(page.locator('.question[open] p')).toContainText('collection HadeethEnc')
  await page.keyboard.press('Enter')
  await expect(page.locator('.question[open]')).toHaveCount(0)
})

test('resets a completed Hadith search from the logo', async ({ page }) => {
  await page.getByRole('button', { name: 'Hadiths', exact: true }).click()
  await page.getByLabel('Que recherchez-vous ?').fill('la colère')
  await page.getByRole('button', { name: 'Rechercher', exact: true }).click()
  await expect(page.locator('.hadith-card')).toBeVisible()
  await page.getByRole('button', { name: 'Sawt AI — Accueil et réinitialisation' }).click()
  await expect(page.locator('#landing-title')).toBeVisible()
  await page.getByRole('button', { name: 'Hadiths', exact: true }).click()
  await expect(page.getByLabel('Que recherchez-vous ?')).toHaveValue('')
  await expect(page.locator('.hadith-card')).toHaveCount(0)
})

test('resets Quran results after switching tools or clicking the logo', async ({ page }) => {
  await page.getByRole('button', { name: 'Coran', exact: true }).click()
  await page.locator('input[type="file"]').setInputFiles({
    name: 'al-fatiha.wav',
    mimeType: 'audio/wav',
    buffer: createSampleAudioBuffer(),
  })
  await expect(page.locator('#result-title')).toBeVisible({ timeout: 15_000 })
  await page.getByRole('button', { name: 'Hadiths', exact: true }).click()
  await expect(page.locator('#hadith-title')).toBeVisible()
  await page.getByRole('button', { name: 'Coran', exact: true }).click()
  await expect(page.locator('#recognition-title')).toBeVisible()
  await expect(page.locator('#result-title')).toHaveCount(0)
  await page.getByRole('button', { name: 'Sawt AI — Accueil et réinitialisation' }).click()
  await expect(page.locator('#landing-title')).toBeVisible()
})

for (const viewport of [
  { width: 320, height: 568 },
  { width: 375, height: 667 },
  { width: 1440, height: 900 },
]) {
  test(`centres the header and empty Hadith search at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport)
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await page.getByRole('button', { name: 'Hadiths', exact: true }).click()
    await expect(page.locator('.hadith-screen.is-idle')).toBeVisible()
    await expect(page.locator('.search-controls')).toHaveCSS('display', 'flex')
    const brand = await page.locator('.brand').boundingBox()
    const nav = await page.locator('.mode-selector').boundingBox()
    const experience = await page.locator('.experience').boundingBox()
    const shell = await page.locator('.search-shell').boundingBox()
    if (!brand || !nav || !experience || !shell) throw new Error('Expected visible layout elements')
    expect(Math.abs(brand.x + brand.width / 2 - viewport.width / 2)).toBeLessThan(2)
    expect(nav.y).toBeGreaterThanOrEqual(brand.y + brand.height)
    const centre = shell.y + shell.height / 2
    expect(Math.abs(centre - (experience.y + experience.height / 2))).toBeLessThan(20)
    expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false)
  })
}
