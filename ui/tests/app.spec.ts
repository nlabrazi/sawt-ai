import { flushPromises, mount } from '@vue/test-utils'
import { $fetch } from 'ofetch'
import { defineComponent, nextTick } from 'vue'
import App from '~/app.vue'
import { hadithFixture } from './fixtures/hadith'

vi.mock('ofetch', () => ({ $fetch: vi.fn() }))
const QuranStub = defineComponent({
  name: 'QuranRecognitionScreen',
  emits: ['navigation-lock'],
  template: '<section>Parcours Coran</section>',
})
async function mountApp() {
  const wrapper = mount(App, {
    global: { stubs: { QuranRecognitionScreen: QuranStub, AppFooter: true } },
  })
  await nextTick()
  return wrapper
}
async function settleScreen() {
  await vi.dynamicImportSettled()
  await flushPromises()
}

describe('App', () => {
  beforeEach(() => vi.mocked($fetch).mockReset())

  it('starts in Quran mode and renders the shared footer', async () => {
    const wrapper = await mountApp()
    expect(wrapper.findComponent({ name: 'QuranRecognitionScreen' }).exists()).toBe(true)
    expect(wrapper.get('.mode-selector button').attributes('aria-pressed')).toBe('true')
    expect(wrapper.findComponent({ name: 'AppFooter' }).exists()).toBe(true)
    expect($fetch).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('retains the Hadith draft and results across mode changes', async () => {
    vi.mocked($fetch).mockResolvedValueOnce({ query: 'la colère', results: [hadithFixture] })
    const wrapper = await mountApp()
    await wrapper.get('.mode-selector button:last-child').trigger('click')
    await settleScreen()
    await wrapper.get('input').setValue('la colère')
    await wrapper.get('form').trigger('submit')
    await flushPromises()
    expect(wrapper.find('.hadith-card').exists()).toBe(true)
    await wrapper.get('.mode-selector button:first-child').trigger('click')
    await settleScreen()
    expect(wrapper.text()).toContain('Parcours Coran')
    await wrapper.get('.mode-selector button:last-child').trigger('click')
    await settleScreen()
    expect((wrapper.get('input').element as HTMLInputElement).value).toBe('la colère')
    expect(wrapper.find('.hadith-card').exists()).toBe(true)
    expect($fetch).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })

  it('prevents changing mode while the Quran microphone is locked', async () => {
    const wrapper = await mountApp()
    wrapper.getComponent(QuranStub).vm.$emit('navigation-lock', true)
    await flushPromises()
    expect(wrapper.get('.mode-selector button:last-child').attributes('disabled')).toBeDefined()
    await wrapper.get('.mode-selector button:last-child').trigger('click')
    expect(wrapper.find('input').exists()).toBe(false)
    wrapper.getComponent(QuranStub).vm.$emit('navigation-lock', false)
    await flushPromises()
    expect(wrapper.get('.mode-selector button:last-child').attributes('disabled')).toBeUndefined()
    wrapper.unmount()
  })
})
