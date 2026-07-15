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
    expect(wrapper.text()).toContain('Passage proposé')
    expect(wrapper.text()).toContain('Vérifiez la sourate et les versets')
    expect(wrapper.get('.reset-action').text()).toBe('Nouvelle récitation')
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

  it.each([
    [
      'insufficient_speech',
      'Récitation trop courte',
      'Récitez distinctement pendant quelques secondes',
    ],
    [
      'non_arabic_speech',
      'Récitation en arabe non détectée',
      'Cet audio semble contenir une autre langue',
    ],
    [
      'low_transcription_confidence',
      'Audio trop difficile à analyser',
      'réduisez le bruit ambiant',
    ],
  ] as const)('explains the %s rejection without showing a result card', (reason, title, hint) => {
    const wrapper = mount(RecognitionResultScreen, {
      props: {
        result: {
          transcription_text: '',
          verse: null,
          detection: {
            status: 'insufficient',
            score: null,
            score_margin: null,
            matched_word_count: 0,
            analyzed_duration_seconds: 5,
            analysis_attempts: 1,
            rejection_reason: reason,
          },
          imam_predictions: [],
          imam_status: 'disabled',
          imam_detection_enabled: false,
        },
        error: null,
      },
      global: {
        stubs: {
          ResultCard: {
            template: '<article class="result-card-stub" />',
          },
          FeedbackForm: true,
        },
      },
    })

    expect(wrapper.text()).toContain(title)
    expect(wrapper.text()).toContain(hint)
    expect(wrapper.find('.result-card-stub').exists()).toBe(false)
  })
})
