import { App } from '@capacitor/app'
import { Browser } from '@capacitor/browser'
import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, effectScope, ref } from 'vue'
import { dispatchAndroidBack, useAndroidBackHandler } from '~/composables/useAndroidBackHandler'
import { useAndroidNavigation } from '~/composables/useAndroidNavigation'

const native = vi.hoisted(() => ({ platform: 'android', remove: vi.fn() }))
vi.mock('@capacitor/core', () => ({ Capacitor: { getPlatform: () => native.platform } }))
vi.mock('@capacitor/app', () => ({
  App: { addListener: vi.fn(async () => ({ remove: native.remove })), exitApp: vi.fn() },
}))
vi.mock('@capacitor/browser', () => ({ Browser: { open: vi.fn(async () => undefined) } }))

function mountNavigation() {
  const path = ref('/')
  const wrapper = mount(
    defineComponent({
      setup() {
        useAndroidNavigation({
          siteUrl: 'https://sawt-ai.nabster.dev',
          getPath: () => path.value,
          setPath: (value) => {
            path.value = value
          },
        })
        return () => null
      },
    }),
  )
  return { wrapper, path }
}

describe('Android navigation', () => {
  beforeEach(() => {
    native.platform = 'android'
    vi.mocked(App.addListener).mockClear()
    window.history.replaceState({}, '', '/')
  })

  it('runs the most recently mounted back handler first and removes it on disposal', async () => {
    const parent = effectScope()
    const child = effectScope()
    const fallback = vi.fn(() => true)
    const close = vi.fn(() => true)
    parent.run(() => useAndroidBackHandler(fallback))
    child.run(() => useAndroidBackHandler(close))
    expect(await dispatchAndroidBack()).toBe(true)
    expect(close).toHaveBeenCalledOnce()
    expect(fallback).not.toHaveBeenCalled()
    child.stop()
    expect(await dispatchAndroidBack()).toBe(true)
    expect(fallback).toHaveBeenCalledOnce()
    parent.stop()
    expect(await dispatchAndroidBack()).toBe(false)
  })

  it('does not install Android listeners on the web', async () => {
    native.platform = 'web'
    const { wrapper } = mountNavigation()
    await flushPromises()
    expect(App.addListener).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('navigates internal legal links locally and opens contact on its public origin', async () => {
    const { wrapper, path } = mountNavigation()
    await flushPromises()
    const anchor = document.createElement('a')
    document.body.append(anchor)
    anchor.href = '/legal-notice'
    anchor.click()
    expect(path.value).toBe('/legal-notice')
    anchor.href = '/contact'
    anchor.click()
    expect(Browser.open).toHaveBeenCalledWith({ url: 'https://sawt-ai.nabster.dev/contact' })
    anchor.href = 'https://hadeethenc.com/fr'
    anchor.click()
    expect(Browser.open).toHaveBeenCalledWith({ url: 'https://hadeethenc.com/fr' })
    anchor.remove()
    wrapper.unmount()
    expect(native.remove).toHaveBeenCalled()
  })

  it('exits only from home and returns a standalone legal launch to home', async () => {
    const { wrapper, path } = mountNavigation()
    await flushPromises()
    const back = vi.mocked(App.addListener).mock.calls[0]?.[1] as (event: {
      canGoBack: boolean
    }) => Promise<void>
    path.value = '/legal-notice'
    await back({ canGoBack: false })
    expect(path.value).toBe('/')
    expect(App.exitApp).not.toHaveBeenCalled()
    await back({ canGoBack: false })
    expect(App.exitApp).toHaveBeenCalledOnce()
    wrapper.unmount()
  })
})
