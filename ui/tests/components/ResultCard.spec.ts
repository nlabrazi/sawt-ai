import { flushPromises, mount } from '@vue/test-utils'

import ResultCard from '~/components/ResultCard.vue'

describe('ResultCard', () => {
  it('shows a compact passage reference without the full Arabic verse', () => {
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

    expect(wrapper.get('.surah-arabic').text()).toBe('الإخلاص')
    expect(wrapper.get('.surah-transliteration').text()).toBe('Sourate Al-Ikhlas')
    expect(wrapper.get('.verse-range').text()).toBe('Versets 1 à 4')
    expect(wrapper.find('.detected-verse-text').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('قل هو الله احد')
  })

  it('shows qualitative confidence without presenting similarity as reliability', () => {
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

    expect(confidenceDetail.text()).toContain('Correspondance probable')
    expect(confidenceDetail.text()).not.toContain('74%')
    expect(wrapper.find('.confidence-score').exists()).toBe(false)
    expect(confidenceDetail.text()).not.toContain('Une correspondance a été trouvée')
  })

  it('does not present an ambiguous high score as a reliable result', () => {
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
            similarity: 0.92,
          },
          detection: {
            status: 'ambiguous',
            score: 0.92,
            score_margin: 0.04,
            matched_word_count: 4,
            analyzed_duration_seconds: 10,
            analysis_attempts: 2,
            rejection_reason: 'ambiguous_match',
          },
          imam_predictions: [],
          imam_status: 'unknown',
          imam_detection_enabled: true,
        },
      },
    })

    const confidenceDetail = wrapper.get('.confidence-detail')

    expect(wrapper.get('.hero-kicker').text()).toBe('Hypothèse à vérifier')
    expect(confidenceDetail.text()).toContain('Correspondance à confirmer')
    expect(confidenceDetail.text()).not.toContain('92%')
    expect(confidenceDetail.text()).not.toContain('Résultat fiable')
    expect(wrapper.get('.primary-btn').text()).toBe('Vérifier le passage')
  })

  it('keeps verse details as the only passage action', () => {
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

    expect(wrapper.find('.copy-verse-btn').exists()).toBe(false)
    expect(wrapper.find('.lucide-copy').exists()).toBe(false)
    expect(wrapper.find('.lucide-eye').exists()).toBe(true)
    expect(wrapper.get('.primary-btn').text()).toBe('Voir le verset')
    expect(wrapper.find('.secondary-btn').exists()).toBe(false)
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

  it('hides imam metadata when the optional detection was disabled', () => {
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
          imam_status: 'disabled',
          imam_detection_enabled: false,
        },
      },
    })

    expect(wrapper.find('.imam-chip').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('Détection imam désactivée')
  })
})
