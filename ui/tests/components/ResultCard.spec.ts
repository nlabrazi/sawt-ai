import { mount } from '@vue/test-utils'

import ResultCard from '~/components/ResultCard.vue'

describe('ResultCard', () => {
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

    expect(wrapper.text()).toContain('Indisponible')
    expect(wrapper.text()).toContain('Identification de l’imam temporairement indisponible.')
    expect(wrapper.text()).not.toContain('Imam non reconnu pour cet extrait.')
  })
})
