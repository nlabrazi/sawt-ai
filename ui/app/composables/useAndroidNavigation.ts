import { App } from '@capacitor/app'
import { Browser } from '@capacitor/browser'
import { Capacitor, type PluginListenerHandle } from '@capacitor/core'
import { onMounted, onScopeDispose } from 'vue'
import { dispatchAndroidBack } from './useAndroidBackHandler'

export function useAndroidNavigation(options: {
  siteUrl: string
  getPath: () => string
  setPath: (path: string) => void
}) {
  let disposed = false
  let handlingBack = false
  let listener: PluginListenerHandle | undefined

  const onPopState = () => options.setPath(window.location.pathname.replace(/\/$/, '') || '/')
  const onClick = (event: MouseEvent) => {
    if (
      event.defaultPrevented ||
      event.button !== 0 ||
      event.ctrlKey ||
      event.metaKey ||
      event.shiftKey ||
      event.altKey
    )
      return
    const anchor =
      event.target instanceof Element ? event.target.closest<HTMLAnchorElement>('a[href]') : null
    if (!anchor || anchor.hasAttribute('download')) return
    const url = new URL(anchor.href, window.location.href)
    if (url.protocol === 'mailto:' || url.protocol === 'tel:') return
    event.preventDefault()
    if (!['https:', 'http:'].includes(url.protocol)) return

    // Captcha/contact run on their public web origin, in an Android Custom Tab.
    if (url.origin !== window.location.origin || url.pathname === '/contact') {
      const target =
        url.origin === window.location.origin
          ? new URL(`${url.pathname}${url.search}${url.hash}`, options.siteUrl).href
          : url.href
      void Browser.open({ url: target }).catch((error) =>
        console.error('Android link failed', error),
      )
      return
    }
    if (url.pathname.replace(/\/$/, '') === '/privacy-policy') {
      url.pathname = '/legal-notice'
      url.hash = '#privacy'
    }
    window.history.pushState({}, '', `${url.pathname}${url.search}${url.hash}`)
    options.setPath(url.pathname.replace(/\/$/, '') || '/')
    if (url.hash) {
      requestAnimationFrame(() => document.getElementById(url.hash.slice(1))?.scrollIntoView())
    } else window.scrollTo(0, 0)
  }

  onMounted(() => {
    if (Capacitor.getPlatform() !== 'android') return
    document.addEventListener('click', onClick)
    window.addEventListener('popstate', onPopState)
    void App.addListener('backButton', async ({ canGoBack }) => {
      if (handlingBack) return
      handlingBack = true
      try {
        if (await dispatchAndroidBack()) return
        if (options.getPath() !== '/') {
          if (canGoBack) window.history.back()
          else {
            window.history.replaceState({}, '', '/')
            options.setPath('/')
          }
        } else await App.exitApp()
      } finally {
        handlingBack = false
      }
    })
      .then((handle) => {
        if (disposed) void handle.remove()
        else listener = handle
      })
      .catch((error) => console.error('Android navigation listener failed', error))
  })

  onScopeDispose(() => {
    disposed = true
    if (typeof document === 'undefined') return
    document.removeEventListener('click', onClick)
    window.removeEventListener('popstate', onPopState)
    void listener?.remove()
  })
}
