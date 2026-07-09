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

  it('shows recording progress while recording', () => {
    const wrapper = mount(RecognitionIdleScreen, {
      props: {
        detectImam: true,
        isRecording: true,
        recordingSeconds: 30,
        maxRecordingSeconds: 90,
      },
    })

    const progress = wrapper.get('[role="progressbar"]')
    const progressFill = wrapper.get('.recording-progress-fill')

    expect(wrapper.text()).toContain('30s / 90s')
    expect(progress.attributes('aria-valuenow')).toBe('30')
    expect(progress.attributes('aria-valuemax')).toBe('90')
    expect(progressFill.attributes('style')).toContain('width: 33.3333')
  })

  it('hides recording progress before recording starts', () => {
    const wrapper = mount(RecognitionIdleScreen, {
      props: {
        detectImam: true,
        isRecording: false,
        recordingSeconds: 0,
        maxRecordingSeconds: 90,
      },
    })

    expect(wrapper.find('[role="progressbar"]').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('/ 90s')
  })

  it('shows microphone error guidance near the primary action', () => {
    const wrapper = mount(RecognitionIdleScreen, {
      props: {
        detectImam: true,
        micError: 'Impossible d’accéder au microphone.',
      },
    })

    const alert = wrapper.get('.hero-action [role="alert"]')

    expect(alert.text()).toContain('Impossible d’accéder au microphone.')
    expect(alert.text()).toContain('Autorisez le micro dans votre navigateur.')
  })

  it('shows upload error guidance near the import action', () => {
    const wrapper = mount(RecognitionIdleScreen, {
      props: {
        detectImam: true,
        uploadError: 'Format audio non pris en charge.',
      },
    })

    const alert = wrapper.get('.upload-action [role="alert"]')

    expect(alert.text()).toContain('Format audio non pris en charge.')
    expect(alert.text()).toContain('Utilisez wav, mp3, m4a, ogg ou webm.')
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
