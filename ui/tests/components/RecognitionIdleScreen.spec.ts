import { mount } from '@vue/test-utils'

import RecognitionIdleScreen from '~/components/RecognitionIdleScreen.vue'

describe('RecognitionIdleScreen', () => {
  it('keeps the idle copy short and action-led', () => {
    const wrapper = mount(RecognitionIdleScreen, {
      props: {
        detectImam: true,
      },
    })

    expect(wrapper.text()).toContain('Touchez pour réciter')
    expect(wrapper.text()).toContain('Récitez quelques secondes. Sawt AI reconnaît le passage.')
    expect(wrapper.text()).toContain('Importer un audio')
    expect(wrapper.text()).not.toContain('Lancez le micro')
    expect(wrapper.find('.record-action').exists()).toBe(false)
  })

  it('emits a microphone action from the primary action button', async () => {
    const wrapper = mount(RecognitionIdleScreen, {
      props: {
        detectImam: true,
      },
    })

    await wrapper.get('.hero-action button').trigger('click')

    expect(wrapper.emitted('micro-click')).toHaveLength(1)
  })

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
