import { flushPromises, mount } from '@vue/test-utils'
import { $fetch } from 'ofetch'

import FeedbackForm from '~/components/FeedbackForm.vue'
import { clearSurahOptionsCache } from '~/composables/useSurahOptions'

vi.mock('ofetch', () => ({
  $fetch: vi.fn(),
}))

const result = {
  transcription_text: 'قل هو الله احد',
  verse: {
    sourate_id: 112,
    sourate_name: 'الإخلاص',
    transliteration: 'Al-Ikhlas',
    start_verse: 1,
    end_verse: 4,
    text: 'قل هو الله احد',
    similarity: 0.92,
  },
  imam_predictions: [],
  imam_status: 'unknown' as const,
  imam_detection_enabled: true,
}

describe('FeedbackForm', () => {
  beforeEach(() => {
    clearSurahOptionsCache()
  })

  it('shows compact validation actions before correction', () => {
    const wrapper = mount(FeedbackForm, {
      props: {
        result,
      },
    })

    expect(wrapper.get('.feedback-label').text()).toBe('Validation')
    expect(wrapper.get('.feedback-action-primary').attributes('aria-label')).toBe(
      'Le résultat est correct',
    )
    expect(wrapper.get('.feedback-action-secondary').attributes('aria-label')).toBe(
      'Le résultat est incorrect',
    )
    expect(wrapper.find('.lucide-thumbs-up').exists()).toBe(true)
    expect(wrapper.find('.lucide-thumbs-down').exists()).toBe(true)
  })

  it('replaces the form with a toast after positive feedback is sent', async () => {
    vi.mocked($fetch).mockResolvedValueOnce({ success: true })

    const wrapper = mount(FeedbackForm, {
      props: {
        result,
      },
      global: {
        stubs: {
          teleport: true,
        },
      },
    })

    await wrapper.get('.feedback-action-primary').trigger('click')
    await flushPromises()

    expect(wrapper.find('.feedback').exists()).toBe(false)
    expect(wrapper.get('.mini-toast').text()).toBe('Retour envoyé, merci de votre contribution !')
    expect(wrapper.get('.mini-toast').attributes('role')).toBe('status')
  })

  it('loads canonical surah options and sends a bounded correction payload', async () => {
    vi.mocked($fetch)
      .mockResolvedValueOnce([
        {
          id: 112,
          name: 'الإخلاص',
          transliteration: 'Al-Ikhlas',
          total_verses: 4,
        },
        {
          id: 114,
          name: 'الناس',
          transliteration: 'An-Nas',
          total_verses: 6,
        },
      ])
      .mockResolvedValueOnce({ success: true })

    const wrapper = mount(FeedbackForm, {
      props: {
        result,
      },
      global: {
        stubs: {
          teleport: true,
        },
      },
    })

    await wrapper.get('.feedback-action-secondary').trigger('click')
    await flushPromises()

    const surahSelect = wrapper.get('#sourate')
    const surahOptions = surahSelect.findAll('option')

    expect(surahOptions).toHaveLength(2)
    expect(surahOptions[1]?.text()).toBe('114 · الناس · An-Nas')

    await surahSelect.setValue('114')
    await flushPromises()

    const startVerseOptions = wrapper.get('#start-verse').findAll('option')
    const endVerseOptions = wrapper.get('#end-verse').findAll('option')

    expect(startVerseOptions).toHaveLength(6)
    expect(endVerseOptions).toHaveLength(6)

    await wrapper.get('#start-verse').setValue('2')
    await wrapper.get('#end-verse').setValue('6')
    await wrapper.get('#comment').setValue('Correction guidée')
    await wrapper.get('.submit-btn').trigger('click')
    await flushPromises()

    expect($fetch).toHaveBeenNthCalledWith(1, 'http://localhost:8000/surahs')
    expect($fetch).toHaveBeenNthCalledWith(2, 'http://localhost:8000/feedback', {
      method: 'POST',
      body: {
        is_correct: false,
        transcription_text: 'قل هو الله احد',
        detected_verse: result.verse,
        correction: {
          sourate_id: 114,
          sourate_name: 'الناس',
          transliteration: 'An-Nas',
          start_verse: 2,
          end_verse: 6,
        },
        comment: 'Correction guidée',
      },
    })
    expect(wrapper.find('.feedback').exists()).toBe(false)
    expect(wrapper.get('.mini-toast').text()).toBe('Retour envoyé, merci de votre contribution !')
  })
})
