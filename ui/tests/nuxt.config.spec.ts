import config from '../nuxt.config'

type HeadLink = {
  rel?: string
  href?: string
  crossorigin?: string
}

describe('Nuxt font configuration', () => {
  it('loads application fonts from one stylesheet in the document head', () => {
    const head = config.app?.head as { link?: HeadLink[] } | undefined
    const fontStylesheets = (head?.link ?? []).filter(
      (link) => link.rel === 'stylesheet' && link.href?.startsWith('https://fonts.googleapis.com/'),
    )

    expect(fontStylesheets).toHaveLength(1)
    expect(fontStylesheets[0]?.href).toContain('family=Amiri:wght@400;700')
    expect(fontStylesheets[0]?.href).toContain('family=Inter:wght@400;500;600;700;800')
    expect(fontStylesheets[0]?.href).toContain('display=swap')
  })

  it('preconnects to both Google Fonts origins', () => {
    const head = config.app?.head as { link?: HeadLink[] } | undefined
    const links = head?.link ?? []

    expect(links).toContainEqual({
      rel: 'preconnect',
      href: 'https://fonts.googleapis.com',
    })
    expect(links).toContainEqual({
      rel: 'preconnect',
      href: 'https://fonts.gstatic.com',
      crossorigin: '',
    })
  })
})
