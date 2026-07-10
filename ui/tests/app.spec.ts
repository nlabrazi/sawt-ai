import { mount } from '@vue/test-utils'
import { ref } from 'vue'

import App from '~/app.vue'

vi.mock('~/composables/useRecognitionFlow', () => ({
  useRecognitionFlow: () => ({
    screenState: ref('idle'),
    onMicroClick: vi.fn(),
    submitAudio: vi.fn(),
    loading: ref(false),
    loadingStep: ref('transcribing'),
    resetApp: vi.fn(),
    error: ref(null),
    result: ref(null),
    uploadError: ref(null),
    micError: ref(null),
    isRecording: ref(false),
    recordingSeconds: ref(0),
    maxRecordingSeconds: ref(90),
    audioLevel: ref(0),
    uploadAccept: ref('audio/*'),
    uploadHint: ref(null),
    detectImam: ref(false),
    imamDetectionAvailable: ref(true),
    imamDetectionMessage: ref(null),
  }),
}))

describe('App', () => {
  it('renders recognition screens through the configured transition', () => {
    const wrapper = mount(App, {
      global: {
        stubs: {
          AppFooter: true,
          RecognitionIdleScreen: true,
        },
      },
    })

    const transition = wrapper.get('transition-stub')

    expect(transition.attributes('name')).toBe('screen-transition')
    expect(transition.attributes('mode')).toBe('out-in')
    expect(wrapper.getComponent({ name: 'RecognitionIdleScreen' }).vm.$.vnode.key).toBe('idle')
  })
})
