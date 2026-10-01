import { flushPromises, mount } from '@vue/test-utils'
import { nextTick, ref } from 'vue'

import QuranRecognitionScreen from '~/components/QuranRecognitionScreen.vue'

const { useRecognitionFlowMock } = vi.hoisted(() => ({
  useRecognitionFlowMock: vi.fn(),
}))

vi.mock('~/composables/useRecognitionFlow', () => ({
  useRecognitionFlow: useRecognitionFlowMock,
}))

const screenState = ref<'idle' | 'loading' | 'result'>('idle')
const error = ref<string | null>(null)
const isRecording = ref(false)
const isFinalizingRecording = ref(false)
const resetApp = vi.fn()
const onMicroClick = vi.fn()

describe('QuranRecognitionScreen', () => {
  beforeEach(() => {
    screenState.value = 'idle'
    error.value = null
    isRecording.value = false
    isFinalizingRecording.value = false
    onMicroClick.mockReset()
    resetApp.mockClear()
    useRecognitionFlowMock.mockReturnValue({
      screenState,
      error,
      result: ref(null),
      loading: ref(false),
      loadingStep: ref('transcribing'),
      uploadError: ref(null),
      micError: ref(null),
      isRecording,
      isFinalizingRecording,
      recordingSeconds: ref(0),
      maxRecordingSeconds: ref(90),
      audioLevel: ref(0),
      uploadAccept: ref('audio/*'),
      uploadHint: ref(null),
      detectImam: ref(false),
      imamDetectionAvailable: ref(true),
      imamDetectionMessage: ref(null),
      onMicroClick,
      submitAudio: vi.fn(),
      resetApp,
    })
  })

  it('renders recognition screens through the configured transition', () => {
    const wrapper = mount(QuranRecognitionScreen, {
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
    const wrapper = mount(QuranRecognitionScreen, {
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
  it('locks mode changes during recording and finalization', async () => {
    const wrapper = mount(QuranRecognitionScreen, {
      global: { stubs: { RecognitionIdleScreen: true } },
    })
    isRecording.value = true
    await nextTick()
    expect(wrapper.emitted('navigation-lock')?.at(-1)).toEqual([true])
    isRecording.value = false
    isFinalizingRecording.value = true
    await nextTick()
    expect(wrapper.emitted('navigation-lock')?.at(-1)).toEqual([true])
    isFinalizingRecording.value = false
    await nextTick()
    expect(wrapper.emitted('navigation-lock')?.at(-1)).toEqual([false])
    wrapper.unmount()
  })

  it('locks while microphone permission is pending and cancels the flow on unmount', async () => {
    let resolve!: () => void
    onMicroClick.mockReturnValueOnce(
      new Promise<void>((yes) => {
        resolve = yes
      }),
    )
    const wrapper = mount(QuranRecognitionScreen, {
      global: { stubs: { RecognitionIdleScreen: true } },
    })
    wrapper.getComponent({ name: 'RecognitionIdleScreen' }).vm.$emit('micro-click')
    await nextTick()
    expect(wrapper.emitted('navigation-lock')?.at(-1)).toEqual([true])
    resolve()
    await flushPromises()
    expect(wrapper.emitted('navigation-lock')?.at(-1)).toEqual([false])
    wrapper.unmount()
    expect(resetApp).toHaveBeenCalledOnce()
  })
})
