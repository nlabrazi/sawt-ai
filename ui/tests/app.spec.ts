import { flushPromises, mount } from '@vue/test-utils'
import { nextTick, ref } from 'vue'

import App from '~/app.vue'

const { useRecognitionFlowMock } = vi.hoisted(() => ({
  useRecognitionFlowMock: vi.fn(),
}))

vi.mock('~/composables/useRecognitionFlow', () => ({
  useRecognitionFlow: useRecognitionFlowMock,
}))

const screenState = ref<'idle' | 'loading' | 'result'>('idle')
const error = ref<string | null>(null)

describe('App', () => {
  beforeEach(() => {
    screenState.value = 'idle'
    error.value = null
    useRecognitionFlowMock.mockReturnValue({
      screenState,
      error,
      result: ref(null),
      loading: ref(false),
      loadingStep: ref('transcribing'),
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
      onMicroClick: vi.fn(),
      submitAudio: vi.fn(),
      resetApp: vi.fn(),
    })
  })

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
    wrapper.unmount()
  })

  it('renders the lazy result screen after analysis', async () => {
    const wrapper = mount(App, {
      global: {
        stubs: {
          AppFooter: true,
          RecognitionIdleScreen: true,
          RecognitionLoadingScreen: true,
        },
      },
    })

    screenState.value = 'loading'
    await nextTick()

    expect(wrapper.findComponent({ name: 'RecognitionLoadingScreen' }).exists()).toBe(true)

    error.value = 'Erreur de test'
    screenState.value = 'result'
    await vi.dynamicImportSettled()
    await flushPromises()

    expect(wrapper.text()).toContain('Erreur de test')
    wrapper.unmount()
  })
})
