import { productionApiUrl, resolveMobileApiUrl } from '../../config/mobile.mjs'

describe('Android API configuration', () => {
  it('uses the verified public API by default', () => {
    expect(resolveMobileApiUrl()).toBe(productionApiUrl)
  })

  it('accepts a custom HTTPS API path and removes a trailing slash', () => {
    expect(resolveMobileApiUrl('https://example.com/api/')).toBe('https://example.com/api')
  })

  it.each([
    'http://api.example.com',
    'https://localhost:8000',
    'https://127.0.0.1',
    'https://[::1]',
    'https://user:secret@example.com',
    'https://example.com?secret=value',
    'https://example.com#fragment',
    'invalid-url',
  ])('rejects an unsuitable API URL: %s', (url) => {
    expect(() => resolveMobileApiUrl(url)).toThrow()
  })
})
