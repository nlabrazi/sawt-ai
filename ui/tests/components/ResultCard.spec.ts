import { flushPromises, mount } from '@vue/test-utils'

import ResultCard from '~/components/ResultCard.vue'

describe('ResultCard', () => {
  it('shows the detected verse text in the result card', () => {
    const wrapper = mount(ResultCard, {
      props: {
        result: {
          transcription_text: 'قل هو الله احد',
          verse: {
            sourate_id: 112,
            sourate_name: 'الإخلاص',
            transliteration: 'Al-Ikhlas',
            start_verse: 1,
            end_verse: 4,
            text: 'قل هو الله احد',
            similarity: 0.93,
          },
          imam_predictions: [],
          imam_status: 'unknown',
          imam_detection_enabled: true,
        },
      },
    })

    expect(wrapper.get('.detected-verse-text').text()).toBe('قل هو الله احد')
  })

  it('shows a compact confidence score in the result card', () => {
    const wrapper = mount(ResultCard, {
      props: {
        result: {
          transcription_text: 'قل هو الله احد',
          verse: {
            sourate_id: 112,
            sourate_name: 'الإخلاص',
            transliteration: 'Al-Ikhlas',
            start_verse: 1,
            end_verse: 4,
            text: 'قل هو الله احد',
            similarity: 0.74,
          },
          imam_predictions: [],
          imam_status: 'unknown',
          imam_detection_enabled: true,
        },
      },
    })

    const confidenceDetail = wrapper.get('.confidence-detail')

    expect(confidenceDetail.text()).toContain('74%')
    expect(confidenceDetail.text()).toContain('Correspondance probable')
    expect(confidenceDetail.text()).not.toContain('Une correspondance a été trouvée')
  })

  it('keeps copy as an icon action next to the verse', () => {
    const wrapper = mount(ResultCard, {
      props: {
        result: {
          transcription_text: 'قل هو الله احد',
          verse: {
            sourate_id: 112,
            sourate_name: 'الإخلاص',
            transliteration: 'Al-Ikhlas',
            start_verse: 1,
            end_verse: 4,
            text: 'قل هو الله احد',
            similarity: 0.93,
          },
          imam_predictions: [],
          imam_status: 'unknown',
          imam_detection_enabled: true,
        },
      },
    })

    expect(wrapper.get('.copy-verse-btn').attributes('aria-label')).toBe('Copier le verset')
    expect(wrapper.find('.lucide-copy').exists()).toBe(true)
    expect(wrapper.find('.lucide-eye').exists()).toBe(true)
    expect(wrapper.find('.secondary-btn').exists()).toBe(false)
  })

  it('shows a toast after copying the detected verse', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    })

    const wrapper = mount(ResultCard, {
      props: {
        result: {
          transcription_text: 'قل هو الله احد',
          verse: {
            sourate_id: 112,
            sourate_name: 'الإخلاص',
            transliteration: 'Al-Ikhlas',
            start_verse: 1,
            end_verse: 4,
            text: 'قل هو الله احد',
            similarity: 0.93,
          },
          imam_predictions: [],
          imam_status: 'unknown',
          imam_detection_enabled: true,
        },
      },
      global: {
        stubs: {
          teleport: true,
        },
      },
    })

    await wrapper.get('.copy-verse-btn').trigger('click')
    await flushPromises()

    expect(writeText).toHaveBeenCalledWith('الإخلاص — Al-Ikhlas\nVersets 1 à 4\nقل هو الله احد')
    expect(wrapper.get('.mini-toast').text()).toBe('Verset copié')
    expect(wrapper.get('.copy-verse-btn').attributes('aria-label')).toBe('Verset copié')
  })

  it('mounts verse details only after the user opens them', async () => {
    const wrapper = mount(ResultCard, {
      props: {
        result: {
          transcription_text: 'قل هو الله احد',
          verse: {
            sourate_id: 112,
            sourate_name: 'الإخلاص',
            transliteration: 'Al-Ikhlas',
            start_verse: 1,
            end_verse: 4,
            text: 'قل هو الله احد',
            similarity: 0.93,
          },
          imam_predictions: [],
          imam_status: 'unknown',
          imam_detection_enabled: true,
        },
      },
      global: {
        stubs: {
          teleport: true,
        },
      },
    })

    expect(wrapper.find('.sheet-overlay').exists()).toBe(false)

    await wrapper.get('.primary-btn').trigger('click')
    await vi.dynamicImportSettled()
    await flushPromises()

    expect(wrapper.find('.sheet-overlay').exists()).toBe(true)
  })

  it('distinguishes an unavailable imam service from an unknown imam result', () => {
    const wrapper = mount(ResultCard, {
      props: {
        result: {
          transcription_text: 'قل هو الله احد',
          verse: {
            sourate_id: 112,
            sourate_name: 'الإخلاص',
            transliteration: 'Al-Ikhlas',
            start_verse: 1,
            end_verse: 4,
            text: 'قل هو الله احد',
            similarity: 0.93,
          },
          imam_predictions: [],
          imam_status: 'unavailable',
          imam_detection_enabled: true,
        },
      },
    })

    expect(wrapper.text()).toContain('Imam indisponible')
    expect(wrapper.text()).toContain('Identification de l’imam temporairement indisponible.')
    expect(wrapper.text()).not.toContain('Imam non reconnu pour cet extrait.')
  })
})
