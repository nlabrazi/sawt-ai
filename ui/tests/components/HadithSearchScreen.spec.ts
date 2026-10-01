import { flushPromises, mount } from '@vue/test-utils'
import { $fetch } from 'ofetch'
import { defineComponent, h, ref } from 'vue'
import HadithSearchScreen from '~/components/HadithSearchScreen.vue'
import { useHadithSearch } from '~/composables/useHadithSearch'
import { hadithFixture } from '../fixtures/hadith'

vi.mock('ofetch', () => ({ $fetch: vi.fn() }))

function mountScreen() {
  return mount(HadithSearchScreen, {
    props: { searchState: useHadithSearch() },
    global: { stubs: { HadithDetailsDialog: true } },
  })
}

describe('HadithSearchScreen', () => {
  beforeEach(() => vi.mocked($fetch).mockReset())

  it('fills an example without making a request', async () => {
    const wrapper = mountScreen()
    await wrapper.get('.examples button').trigger('click')
    expect((wrapper.get('input').element as HTMLInputElement).value).toBe(
      'Ne pas se mettre en colère',
    )
    expect($fetch).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('announces validation errors without calling the API', async () => {
    const wrapper = mountScreen()
    await wrapper.get('form').trigger('submit')
    expect(wrapper.get('[role="alert"]').text()).toContain('3 caractères')
    expect(wrapper.get('input').attributes('aria-invalid')).toBe('true')
    expect($fetch).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('renders proposals, labels their query and opens the selected reading', async () => {
    vi.mocked($fetch).mockResolvedValueOnce({ query: 'la colère', results: [hadithFixture] })
    const wrapper = mountScreen()
    await wrapper.get('input').setValue('la colère')
    await wrapper.get('form').trigger('submit')
    await flushPromises()
    expect(wrapper.get('h2').text()).toBe('Hadiths proposés')
    expect(wrapper.text()).toContain('Pour « la colère »')
    expect(wrapper.get('[role="status"]').text()).toContain('1 proposition disponible')
    await wrapper.get('.hadith-card button').trigger('click')
    await vi.dynamicImportSettled()
    await flushPromises()
    expect(wrapper.findComponent({ name: 'HadithDetailsDialog' }).exists()).toBe(true)
    wrapper.unmount()
  })

  it('distinguishes empty results from a service error and supports retrying', async () => {
    vi.mocked($fetch).mockRejectedValueOnce({ statusCode: 503 })
    const wrapper = mountScreen()
    await wrapper.get('input').setValue('la colère')
    await wrapper.get('form').trigger('submit')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('temporairement indisponible')
    vi.mocked($fetch).mockResolvedValueOnce({ query: 'la colère', results: [] })
    await wrapper.get('.status-panel button').trigger('click')
    await flushPromises()
    expect(wrapper.find('[role="alert"]').exists()).toBe(false)
    expect(wrapper.get('.empty-state').text()).toContain('Aucun résultat exploitable')
    wrapper.unmount()
  })

  it('offers cancellation while the request is pending', async () => {
    let resolve!: (value: unknown) => void
    vi.mocked($fetch).mockReturnValueOnce(
      new Promise((yes) => {
        resolve = yes
      }),
    )
    const wrapper = mountScreen()
    await wrapper.get('input').setValue('la colère')
    await wrapper.get('form').trigger('submit')
    expect(wrapper.find('.loading-panel').exists()).toBe(true)
    expect(wrapper.get('.search-button').attributes('disabled')).toBeDefined()
    await wrapper.get('.loading-copy button').trigger('click')
    expect(wrapper.find('.loading-panel').exists()).toBe(false)
    resolve({ query: 'la colère', results: [hadithFixture] })
    await flushPromises()
    expect(wrapper.find('.hadith-card').exists()).toBe(false)
    wrapper.unmount()
  })

  it('cancels on removal and retains the draft on remount', async () => {
    let resolve!: (value: unknown) => void
    vi.mocked($fetch).mockReturnValueOnce(
      new Promise((yes) => {
        resolve = yes
      }),
    )
    const searchState = useHadithSearch()
    const visible = ref(true)
    const Host = defineComponent({
      setup: () => () => (visible.value ? h(HadithSearchScreen, { searchState }) : null),
    })
    const wrapper = mount(Host)
    await wrapper.get('input').setValue('la colère')
    await wrapper.get('form').trigger('submit')
    const signal = vi.mocked($fetch).mock.calls[0]?.[1]?.signal as AbortSignal
    visible.value = false
    await flushPromises()
    expect(signal.aborted).toBe(true)
    visible.value = true
    await flushPromises()
    expect((wrapper.get('input').element as HTMLInputElement).value).toBe('la colère')
    resolve({ query: 'la colère', results: [] })
    await flushPromises()
    expect(wrapper.find('.results').exists()).toBe(false)
    wrapper.unmount()
  })
})
