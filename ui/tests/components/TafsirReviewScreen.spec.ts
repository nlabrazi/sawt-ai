import { flushPromises, mount } from '@vue/test-utils'
import { $fetch } from 'ofetch'
import TafsirReviewScreen from '~/components/TafsirReviewScreen.vue'
import { clearSurahOptionsCache } from '~/composables/useSurahOptions'
import { tafsirFixture } from '../fixtures/tafsir'

vi.mock('ofetch', () => ({ $fetch: vi.fn() }))

beforeEach(() => {
  vi.mocked($fetch).mockReset()
  clearSurahOptionsCache()
})

it('loads drafts only after login and refuses to validate an unsaved correction', async () => {
  vi.mocked($fetch)
    .mockResolvedValueOnce(undefined)
    .mockResolvedValueOnce([tafsirFixture()])
    .mockResolvedValueOnce([])
  const wrapper = mount(TafsirReviewScreen)
  await flushPromises()
  expect($fetch).not.toHaveBeenCalled()
  await wrapper.get('input').setValue('fictitious-password')
  await wrapper.get('form').trigger('submit')
  await flushPromises()
  expect(wrapper.findAll('.review-entry')).toHaveLength(1)
  expect($fetch).toHaveBeenNthCalledWith(
    1,
    'http://localhost:8000/internal/tafsir/access',
    expect.anything(),
  )
  await wrapper.get('textarea').setValue('Correction fictive.')
  const buttons = wrapper.findAll('.entry-actions button')
  expect(buttons[0]?.attributes('disabled')).toBeUndefined()
  expect(buttons[2]?.attributes('disabled')).toBeDefined()
  wrapper.unmount()
})
