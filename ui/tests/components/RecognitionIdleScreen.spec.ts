import { mount } from '@vue/test-utils'

import RecognitionIdleScreen from '~/components/RecognitionIdleScreen.vue'

describe('RecognitionIdleScreen', () => {
  it('allows webm uploads and documents the supported format', () => {
    const wrapper = mount(RecognitionIdleScreen, {
      props: {
        detectImam: true,
        uploadAccept: '.wav,.mp3,.m4a,.ogg,.webm',
        uploadHint: 'Formats : wav, mp3, m4a, ogg, webm · max 12 Mo · max 90 sec',
      },
    })

    const fileInput = wrapper.get('input[type="file"]')

    expect(fileInput.attributes('accept')).toContain('.webm')
    expect(wrapper.text()).toContain('Formats : wav, mp3, m4a, ogg, webm')
  })

  it('disables imam detection and shows a warning when the feature is unavailable', () => {
    const wrapper = mount(RecognitionIdleScreen, {
      props: {
        detectImam: false,
        imamDetectionAvailable: false,
        imamDetectionMessage: 'La reconnaissance de l’imam est temporairement indisponible.',
      },
    })

    const checkbox = wrapper.get('input[type="checkbox"]')

    expect(checkbox.attributes('disabled')).toBeDefined()
    expect(wrapper.get('.imam-toggle').classes()).toContain('is-disabled')
    expect(wrapper.get('.imam-toggle-hint').classes()).toContain('is-unavailable')
    expect(wrapper.text()).toContain('La reconnaissance de l’imam est temporairement indisponible.')
  })
})
