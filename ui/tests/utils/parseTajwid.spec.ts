import { parseTajwidToHtml } from '~/utils/parseTajwid'

describe('parseTajwidToHtml', () => {
  it('wraps tagged segments with a sanitized CSS class', () => {
    const html = parseTajwidToHtml('[ghunnah[الرَّحْمَٰنِ]]')

    expect(html).toContain('tajwid-rule-ghunnah')
    expect(html).toContain('data-rule="ghunnah"')
    expect(html).toContain('الرَّحْمَٰنِ')
  })

  it('escapes raw HTML outside tajwid tags', () => {
    const html = parseTajwidToHtml('<script>alert(1)</script>')

    expect(html).toBe('&lt;script&gt;alert(1)&lt;/script&gt;')
  })

  it('supports nested tajwid tags', () => {
    const html = parseTajwidToHtml('[outer[abc [inner[def]] ghi]]')

    expect(html).toContain('tajwid-rule-outer')
    expect(html).toContain('tajwid-rule-inner')
  })
})
