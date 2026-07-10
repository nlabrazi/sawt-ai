import config from '../nuxt.config'

type HeadLink = {
  rel?: string
  href?: string
  crossorigin?: string
}

type HeadMeta = {
  name?: string
  property?: string
  content?: string
}

type HeadScript = {
  type?: string
  innerHTML?: string
}

describe('Nuxt font configuration', () => {
  it('loads application fonts from one stylesheet in the document head', () => {
    const head = config.app?.head as { link?: HeadLink[] } | undefined
    const fontStylesheets = (head?.link ?? []).filter(
      (link) => link.rel === 'stylesheet' && link.href?.startsWith('https://fonts.googleapis.com/'),
    )

    expect(fontStylesheets).toHaveLength(1)
    expect(fontStylesheets[0]?.href).toContain('family=Amiri+Quran')
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

describe('Nuxt social sharing metadata', () => {
  const head = config.app?.head as { meta?: HeadMeta[]; link?: HeadLink[] } | undefined
  const meta = head?.meta ?? []

  it('provides Open Graph metadata with an absolute public image URL', () => {
    expect(meta).toContainEqual({ property: 'og:type', content: 'website' })
    expect(meta).toContainEqual({ property: 'og:site_name', content: 'Sawt AI' })
    expect(meta).toContainEqual({
      property: 'og:image',
      content: 'https://sawt-ai.nabster.dev/assets/images/screenshot.png',
    })
    expect(meta).toContainEqual({ property: 'og:image:width', content: '1024' })
    expect(meta).toContainEqual({ property: 'og:image:height', content: '1024' })
  })

  it('provides a large Twitter card and a canonical URL', () => {
    expect(meta).toContainEqual({ name: 'twitter:card', content: 'summary_large_image' })
    expect(head?.link).toContainEqual({
      rel: 'canonical',
      href: 'https://sawt-ai.nabster.dev',
    })
  })
})

describe('Nuxt search engine metadata', () => {
  const head = config.app?.head as { meta?: HeadMeta[]; script?: HeadScript[] } | undefined

  it('allows indexing and large search result previews', () => {
    expect(head?.meta).toContainEqual({
      name: 'robots',
      content: 'index, follow, max-image-preview:large, max-snippet:-1',
    })
  })

  it('describes the site as a web application using JSON-LD', () => {
    const jsonLd = head?.script?.find((script) => script.type === 'application/ld+json')
    const data = JSON.parse(jsonLd?.innerHTML ?? '{}')

    expect(data).toMatchObject({
      '@context': 'https://schema.org',
      '@type': 'WebApplication',
      name: 'Sawt AI',
      url: 'https://sawt-ai.nabster.dev',
      applicationCategory: 'EducationalApplication',
    })
  })
})
