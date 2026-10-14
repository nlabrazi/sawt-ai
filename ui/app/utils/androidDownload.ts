export function resolveAndroidDownloadUrl(value: unknown): string | null {
  if (typeof value !== 'string' || !value.trim()) return null
  const candidate = value.trim()
  if (/^\/(?!\/)[^?#]+\.apk$/.test(candidate)) return candidate
  try {
    const url = new URL(candidate)
    if (
      url.protocol !== 'https:' ||
      url.username ||
      url.password ||
      url.hash ||
      !url.pathname.endsWith('.apk')
    )
      return null
    return url.href
  } catch {
    return null
  }
}
