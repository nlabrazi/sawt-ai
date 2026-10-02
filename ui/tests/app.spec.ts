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

  it('starts on the landing screen with three navigation buttons and a shared footer', async () => {
    const wrapper = await mountApp()
    expect(wrapper.get('#landing-title').text()).toContain('Retrouvez les mots')
    expect(wrapper.findComponent({ name: 'QuranRecognitionScreen' }).exists()).toBe(false)
    expect(wrapper.findAll('.mode-selector button').map((button) => button.text())).toEqual([
      'Coran',
      'Hadiths',
      'FAQ',
    ])
    expect(wrapper.findComponent({ name: 'AppFooter' }).exists()).toBe(true)
    expect($fetch).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('opens Quran from a landing CTA and returns home from the logo', async () => {
    const wrapper = await mountApp()
    await wrapper.get('.primary-action').trigger('click')
    await settleScreen()
    expect(wrapper.findComponent(QuranStub).exists()).toBe(true)
    expect(wrapper.get('.mode-selector button').attributes('aria-pressed')).toBe('true')
    await wrapper.get('.brand').trigger('click')
    await settleScreen()
    expect(wrapper.findComponent(QuranStub).exists()).toBe(false)
    expect(wrapper.find('#landing-title').exists()).toBe(true)
    wrapper.unmount()
  })

  it('clears the Hadith draft and results when changing screens', async () => {
    vi.mocked($fetch).mockResolvedValueOnce({ query: 'la colère', results: [hadithFixture] })
    const wrapper = await mountApp()
    await wrapper.get('.secondary-action').trigger('click')
    await settleScreen()
    await wrapper.get('input').setValue('la colère')
    await wrapper.get('form').trigger('submit')
    await flushPromises()
    expect(wrapper.find('.hadith-card').exists()).toBe(true)
    await wrapper.get('.mode-selector button:first-of-type').trigger('click')
    await settleScreen()
    expect(wrapper.text()).toContain('Parcours Coran')
    await wrapper.get('.mode-selector button:nth-of-type(2)').trigger('click')
    await settleScreen()
    expect((wrapper.get('input').element as HTMLInputElement).value).toBe('')
    expect(wrapper.find('.hadith-card').exists()).toBe(false)
    expect($fetch).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })

  it('aborts a Hadith search from the logo and ignores its late result', async () => {
    let resolve!: (value: unknown) => void
    vi.mocked($fetch).mockReturnValueOnce(
      new Promise((done) => {
        resolve = done
      }),
    )
    const wrapper = await mountApp()
    await wrapper.get('.secondary-action').trigger('click')
    await settleScreen()
    await wrapper.get('input').setValue('la colère')
    await wrapper.get('form').trigger('submit')
    const signal = vi.mocked($fetch).mock.calls[0]?.[1]?.signal as AbortSignal
    await wrapper.get('.brand').trigger('click')
    await settleScreen()
    expect(signal.aborted).toBe(true)
    expect(wrapper.find('#landing-title').exists()).toBe(true)
    resolve({ query: 'la colère', results: [hadithFixture] })
    await settleScreen()
    await wrapper.get('.secondary-action').trigger('click')
    await settleScreen()
    expect((wrapper.get('input').element as HTMLInputElement).value).toBe('')
    expect(wrapper.find('.hadith-card').exists()).toBe(false)
    expect(wrapper.find('.loading-panel').exists()).toBe(false)
    wrapper.unmount()
  })

  it('prevents leaving Quran or resetting from the logo while recording', async () => {
    const wrapper = await mountApp()
    await wrapper.get('.primary-action').trigger('click')
    await settleScreen()
    wrapper.getComponent(QuranStub).vm.$emit('navigation-lock', true)
    await flushPromises()
    for (const selector of [
      '.brand',
      '.mode-selector button:nth-of-type(2)',
      '.mode-selector button:last-of-type',
    ]) {
      expect(wrapper.get(selector).attributes('disabled')).toBeDefined()
      await wrapper.get(selector).trigger('click')
    }
    expect(wrapper.findComponent(QuranStub).exists()).toBe(true)
    wrapper.getComponent(QuranStub).vm.$emit('navigation-lock', false)
    await flushPromises()
    expect(wrapper.get('.brand').attributes('disabled')).toBeUndefined()
    expect(wrapper.get('.mode-selector button:last-of-type').attributes('disabled')).toBeUndefined()
    await wrapper.get('.mode-selector button:last-of-type').trigger('click')
    await settleScreen()
    expect(wrapper.find('#faq-title').exists()).toBe(true)
    wrapper.unmount()
  })
})
