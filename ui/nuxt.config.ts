// https://nuxt.com/docs/api/configuration/nuxt-config
const hmrClientPort = process.env.NUXT_HMR_CLIENT_PORT
const hmrHost = process.env.NUXT_HMR_HOST
const isDevelopment = process.env.NODE_ENV === 'development'
const usePolling = process.env.CHOKIDAR_USEPOLLING === 'true'

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
      link: [
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
          href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap',
        },
      ],
      script: [
        {
          src: 'https://umami.nabster.dev/script.js',
          defer: true,
          'data-website-id': '6c8e5246-8c6c-4964-a20d-f8e66169aed6',
        },
      ],
    },
  },
})
