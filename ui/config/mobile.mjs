export const productionApiUrl = 'https://api.sawt-ai.nabster.dev'

export function resolveMobileApiUrl(value = productionApiUrl) {
  const url = new URL(value)
  if (
    url.protocol !== 'https:' ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    /^(localhost|127\.|\[::1\])/i.test(url.hostname)
  ) {
    throw new Error('La compilation Android exige une URL API HTTPS publique, sans identifiants.')
  }
  return url.href.replace(/\/$/, '')
}
