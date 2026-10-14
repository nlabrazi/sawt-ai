import { mount } from '@vue/test-utils'
import AndroidDownloadBanner from '~/components/AndroidDownloadBanner.vue'
import { resolveAndroidDownloadUrl } from '~/utils/androidDownload'
import { setRuntimeConfig } from '../mocks/nuxt-app'

const native = vi.hoisted(() => ({ platform: 'web' }))
vi.mock('@capacitor/core', () => ({ Capacitor: { getPlatform: () => native.platform } }))

describe('Android download', () => {
  beforeEach(() => {
    native.platform = 'web'
    setRuntimeConfig({
      public: {
        apiBaseUrl: 'http://localhost:8000',
        androidDownloadUrl: '/downloads/sawt-ai-android.apk',
      },
    })
  })

  it('offers a direct APK download and installation instructions', () => {
    const wrapper = mount(AndroidDownloadBanner)
    expect(wrapper.get('a').attributes('href')).toBe('/downloads/sawt-ai-android.apk')
    expect(wrapper.get('a').attributes('download')).toBe('sawt-ai-android.apk')
    expect(wrapper.text()).toContain('Télécharger pour Android')
    expect(wrapper.text()).toContain('autorisez l’installation')
  })

  it('hides the download inside the Android app', () => {
    native.platform = 'android'
    expect(mount(AndroidDownloadBanner).find('a').exists()).toBe(false)
  })

  it('hides the download until an APK URL is configured', () => {
    setRuntimeConfig({ public: { apiBaseUrl: 'http://localhost:8000' } })
    expect(mount(AndroidDownloadBanner).find('a').exists()).toBe(false)
  })

  it.each([
    'javascript:alert(1)',
    'http://example.com/app.apk',
    '//example.com/app.apk',
    'https://user:secret@example.com/app.apk',
    'https://example.com/page',
    null,
  ])('rejects an unsuitable download URL: %s', (url) => {
    expect(resolveAndroidDownloadUrl(url)).toBeNull()
  })

  it('accepts HTTPS hosting on another domain', () => {
    expect(resolveAndroidDownloadUrl('https://downloads.example.com/app.apk')).toBe(
      'https://downloads.example.com/app.apk',
    )
  })
})
