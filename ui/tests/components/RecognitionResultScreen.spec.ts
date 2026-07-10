import { mount } from '@vue/test-utils'

import RecognitionResultScreen from '~/components/RecognitionResultScreen.vue'

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

describe('RecognitionResultScreen', () => {
  it('renders result, validation, then reset without a confidence banner', () => {
    const wrapper = mount(RecognitionResultScreen, {
      props: {
        result,
        error: null,
      },
      global: {
        stubs: {
          ResultCard: {
            template: '<article class="result-card-stub" />',
          },
          FeedbackForm: {
            template: '<section class="feedback-stub" />',
          },
        },
      },
    })

    const orderedSections = Array.from(
      wrapper.element.querySelectorAll('.result-card-stub, .feedback-stub, .reset-action'),
    ).map((element) => element.className)

    expect(wrapper.find('.banner').exists()).toBe(false)
    expect(orderedSections).toEqual(['result-card-stub', 'feedback-stub', 'reset-action'])
    expect(wrapper.get('.reset-action').text()).toBe('Recommencer')
    expect(wrapper.find('.lucide-rotate-ccw').exists()).toBe(true)
  })

  it('emits reset from the bottom action', async () => {
    const wrapper = mount(RecognitionResultScreen, {
      props: {
        result,
        error: null,
      },
      global: {
        stubs: {
          ResultCard: true,
          FeedbackForm: true,
        },
      },
    })

    await wrapper.get('.reset-action').trigger('click')

    expect(wrapper.emitted('reset')).toHaveLength(1)
  })
})
