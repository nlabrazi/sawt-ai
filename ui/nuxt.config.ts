import { defineNuxtConfig } from 'nuxt/config'

// https://nuxt.com/docs/api/configuration/nuxt-config
const hmrClientPort = process.env.NUXT_HMR_CLIENT_PORT
const hmrHost = process.env.NUXT_HMR_HOST
const isDevelopment = process.env.NODE_ENV === 'development'
const usePolling = process.env.CHOKIDAR_USEPOLLING === 'true'
const siteUrl = (process.env.NUXT_PUBLIC_SITE_URL || 'https://sawt-ai.nabster.dev').replace(
  /\/$/,
  '',
)
const contactEmail = process.env.NUXT_PUBLIC_CONTACT_EMAIL?.trim() || ''
const siteName = 'Sawt AI'
const siteDescription =
  'Identifiez un verset du Coran à partir d’un enregistrement audio grâce à l’intelligence artificielle.'
const socialImage = `${siteUrl}/assets/images/screenshot.png`
const structuredData = {
  '@context': 'https://schema.org',
  '@type': 'WebApplication',
  name: siteName,
  url: siteUrl,
  description: siteDescription,
  applicationCategory: 'EducationalApplication',
  operatingSystem: 'Web',
  inLanguage: 'fr-FR',
  image: socialImage,
  offers: {
    '@type': 'Offer',
    price: '0',
    priceCurrency: 'EUR',
  },
}

export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: isDevelopment },
  devServer: {
    host: '0.0.0.0',
    port: 3000,
  },
  runtimeConfig: {
    public: {
      apiBaseUrl: process.env.NUXT_PUBLIC_API_BASE_URL || 'http://localhost:8000',
      contactEmail,
      siteUrl,
    },
  },
  vite: {
    server: {
      hmr: {
        ...(hmrClientPort ? { clientPort: Number(hmrClientPort) } : {}),
        ...(hmrHost ? { host: hmrHost } : {}),
        protocol: 'ws',
      },
      watch: {
        interval: 100,
        usePolling,
      },
    },
  },
  app: {
    head: {
      htmlAttrs: {
        lang: 'fr',
      },
      title: `${siteName} — Reconnaissance de versets coraniques`,
      meta: [
        { name: 'description', content: siteDescription },
        { name: 'robots', content: 'index, follow, max-image-preview:large, max-snippet:-1' },
        { property: 'og:type', content: 'website' },
        { property: 'og:site_name', content: siteName },
        { property: 'og:title', content: `${siteName} — Reconnaissance de versets coraniques` },
        { property: 'og:description', content: siteDescription },
        { property: 'og:url', content: siteUrl },
        { property: 'og:image', content: socialImage },
        { property: 'og:image:secure_url', content: socialImage },
        { property: 'og:image:type', content: 'image/png' },
        { property: 'og:image:width', content: '1024' },
        { property: 'og:image:height', content: '1024' },
        {
          property: 'og:image:alt',
          content: 'Sawt AI, reconnaissance audio de versets coraniques',
        },
        { property: 'og:locale', content: 'fr_FR' },
        { name: 'twitter:card', content: 'summary_large_image' },
        { name: 'twitter:title', content: `${siteName} — Reconnaissance de versets coraniques` },
        { name: 'twitter:description', content: siteDescription },
        { name: 'twitter:image', content: socialImage },
        {
          name: 'twitter:image:alt',
          content: 'Sawt AI, reconnaissance audio de versets coraniques',
        },
      ],
      link: [
        {
          rel: 'canonical',
          href: siteUrl,
        },
        {
          rel: 'preconnect',
          href: 'https://fonts.googleapis.com',
        },
        {
          rel: 'preconnect',
          href: 'https://fonts.gstatic.com',
          crossorigin: '',
        },
        {
          rel: 'stylesheet',
          href: 'https://fonts.googleapis.com/css2?family=Amiri+Quran&family=Amiri:wght@400;700&family=Inter:wght@400;500;600;700;800&display=swap',
        },
      ],
      script: [
        {
          type: 'application/ld+json',
          innerHTML: JSON.stringify(structuredData),
        },
        {
          src: 'https://umami.nabster.dev/script.js',
          defer: true,
          'data-website-id': '6c8e5246-8c6c-4964-a20d-f8e66169aed6',
        },
      ],
    },
  },
})
