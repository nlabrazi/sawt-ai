import { expect, test } from '@playwright/test'
import { corsHeaders, setupMockApi } from './helpers/mock-api'

// Contrat du SDK simulé : aucun challenge réel et aucun e-mail envoyé.
const captchaScript = `
let widget, options;
window.hcaptcha = {
  render(element, config) {
    widget = element;
    options = config;
    const verify = document.createElement('button');
    verify.type = 'button';
    verify.textContent = 'Verify test captcha';
    verify.onclick = () => config.callback('test-captcha-token');
    const expire = document.createElement('button');
    expire.type = 'button';
    expire.textContent = 'Expire test captcha';
    expire.onclick = () => config['expired-callback']();
    element.append(verify, expire);
    return 'test-widget';
  },
  reset() { options['expired-callback'](); },
  remove() { widget.replaceChildren(); },
};
window.sawtCaptchaReady();
`

test('serves the portfolio contact layout without JavaScript and with its own metadata', async ({
  browser,
  baseURL,
}) => {
  const context = await browser.newContext({ baseURL, javaScriptEnabled: false })
  const page = await context.newPage()
  await setupMockApi(page)
  await page.setViewportSize({ width: 320, height: 667 })
  const response = await page.goto('/contact')
  expect(response?.status()).toBe(200)
  await expect(page.getByRole('heading', { name: 'Contact', exact: true })).toBeVisible()
  await expect(page.locator('.contact-link[href^="mailto:"]')).toHaveAttribute('href', /^mailto:/)
  await expect(page.getByRole('heading', { name: 'Coordonnées' })).toBeVisible()
  await expect(page.locator('#contact-name')).toBeDisabled()
  await expect(page).toHaveTitle('Contact — Sawt AI')
  await expect(page.locator('link[rel="canonical"]')).toHaveAttribute('href', /\/contact$/)
  await expect(page.locator('meta[property="og:url"]')).toHaveAttribute('content', /\/contact$/)
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false)
  await page.getByRole('link', { name: '← Retour à Sawt AI' }).click()
  await expect(page.locator('#landing-title')).toBeVisible()
  await context.close()
})

test('requires a fresh captcha after a failed send and preserves the message until confirmed', async ({
  page,
}) => {
  const submissions: Record<string, unknown>[] = []
  await setupMockApi(page)
  await page.route('https://js.hcaptcha.com/1/api.js?*', (route) =>
    route.fulfill({ contentType: 'application/javascript', body: captchaScript }),
  )
  await page.route('https://api.web3forms.com/**', async (route) => {
    if (route.request().method() === 'OPTIONS') {
      await route.fulfill({ status: 204, headers: corsHeaders })
      return
    }
    submissions.push(route.request().postDataJSON())
    await route.fulfill({
      status: 200,
      headers: corsHeaders,
      contentType: 'application/json',
      body: JSON.stringify({ success: submissions.length > 1 }),
    })
  })
  await page.setViewportSize({ width: 320, height: 667 })
  await page.goto('/contact')
  await expect(page.getByRole('heading', { name: 'Contact', exact: true })).toBeVisible()
  test.skip(
    (await page.locator('.form-unavailable').count()) > 0,
    'Le serveur réutilisé doit avoir une clé Web3Forms configurée ; utiliser un serveur de test dédié.',
  )

  const submitButton = page.getByRole('button', { name: 'Envoyer le message' })
  await expect(submitButton).toBeEnabled()
  expect(await page.locator('script[src*="js.hcaptcha.com"]').count()).toBe(0)
  await submitButton.click()
  await expect(page.locator('#contact-name')).toBeFocused()
  expect(submissions).toHaveLength(0)
  await page.locator('#contact-name').fill('Prénom Nom')
  await page.locator('#contact-email').fill('person@example.com')
  await page.locator('#contact-subject').fill('Signalement')
  await page.locator('#contact-message').fill('Une remarque sur la référence 2:255.')
  await submitButton.click()
  await expect(page.getByRole('alert')).toContainText('vérification antispam')
  expect(submissions).toHaveLength(0)
  await page.getByRole('button', { name: 'Verify test captcha' }).click()
  await page.getByRole('button', { name: 'Expire test captcha' }).click()
  await submitButton.click()
  expect(submissions).toHaveLength(0)
  await page.getByRole('button', { name: 'Verify test captcha' }).click()
  await submitButton.click()
  await expect(page.getByRole('alert')).toContainText('L’envoi n’a pas pu être confirmé.')
  await expect(page.locator('#contact-message')).toHaveValue('Une remarque sur la référence 2:255.')
  await submitButton.click()
  await expect(page.getByRole('alert')).toContainText('vérification antispam')
  expect(submissions).toHaveLength(1)
  await page.getByRole('button', { name: 'Verify test captcha' }).click()
  await submitButton.click()
  await expect(page.getByRole('status')).toContainText('Votre message a été transmis.')
  await expect(page.locator('#contact-message')).toHaveValue('')
  expect(submissions).toHaveLength(2)
  expect(submissions[0]).toMatchObject({
    name: 'Prénom Nom',
    email: 'person@example.com',
    subject: 'Sawt AI — Signalement',
    message: 'Une remarque sur la référence 2:255.',
    from_name: 'Sawt AI',
    'h-captcha-response': 'test-captcha-token',
  })
  expect(submissions[0]).not.toHaveProperty('redirect')
  await expect(page).toHaveURL(/\/contact$/)
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false)
})

test('offers a retry when the captcha script is blocked', async ({ page }) => {
  await setupMockApi(page)
  await page.route('https://js.hcaptcha.com/1/api.js?*', (route) => route.abort())
  await page.goto('/contact')
  test.skip(
    (await page.locator('.form-unavailable').count()) > 0,
    'Clé Web3Forms requise sur le serveur de test.',
  )
  await expect(page.locator('#contact-name')).toBeEnabled()
  await page.locator('#contact-name').focus()
  await expect(page.locator('.captcha-error')).toContainText(
    'vérification antispam est indisponible',
  )
  await page.route('https://js.hcaptcha.com/1/api.js?*', (route) =>
    route.fulfill({ contentType: 'application/javascript', body: captchaScript }),
  )
  await page.getByRole('button', { name: 'Réessayer', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Verify test captcha' })).toBeVisible()
})

for (const width of [375, 1440]) {
  test(`lays out the portfolio contact cards and identity fields at ${width}px`, async ({
    page,
  }) => {
    await setupMockApi(page)
    await page.setViewportSize({ width, height: 900 })
    await page.goto('/contact')
    const cards = await page.locator('.contact-card').evaluateAll((nodes) =>
      nodes.map((node) => {
        const box = node.getBoundingClientRect()
        return { x: box.x, y: box.y, width: box.width }
      }),
    )
    expect(cards).toHaveLength(2)
    const [left, right] = cards
    if (!left || !right) throw new Error('Expected both contact cards')
    expect(Math.abs(left.width - right.width)).toBeLessThan(2)
    if (width >= 1024) {
      expect(Math.abs(left.y - right.y)).toBeLessThan(2)
      const name = await page.locator('#contact-name').boundingBox()
      const email = await page.locator('#contact-email').boundingBox()
      if (!name || !email) throw new Error('Expected visible identity fields')
      expect(Math.abs(name.y - email.y)).toBeLessThan(2)
    } else {
      expect(right.y).toBeGreaterThan(left.y)
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false)
  })
}
