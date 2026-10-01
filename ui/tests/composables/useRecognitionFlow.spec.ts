import { nextTick, ref } from 'vue'

async function setupRecognitionFlow(options: { deferStop?: boolean } = {}) {
  vi.resetModules()

  const isRecording = ref(false)
  const isFinalizingRecording = ref(false)
  const maxDurationReached = ref(false)
  let releaseStop = () => undefined
  const stopGate = new Promise<void>((resolve) => {
    releaseStop = resolve
  })
  const startRecording = vi.fn(async () => {
    isRecording.value = true
  })
  const stopRecording = vi.fn(async () => {
    isRecording.value = false
    isFinalizingRecording.value = true

    if (options.deferStop) {
      await stopGate
    }

    isFinalizingRecording.value = false
    return new File(['audio'], 'recording.wav', { type: 'audio/wav' })
  })
  const recognizeAudio = vi.fn()

  vi.doMock('~/composables/useApiHealth', () => ({
    useApiHealth: () => ({
      imamDetectionAvailable: ref(true),
      imamDetectionMessage: ref(null),
      uploadPolicy: ref(null),
      refreshHealth: vi.fn(),
      markImamDetectionUnavailable: vi.fn(),
    }),
  }))
  vi.doMock('~/composables/useRecognition', () => ({
    useRecognition: () => ({
      loading: ref(false),
      loadingStep: ref('transcribing'),
      error: ref(null),
      result: ref(null),
      recognizeAudio,
      reset: vi.fn(),
    }),
  }))
  vi.doMock('~/composables/useMicrophoneRecorder', () => ({
    useMicrophoneRecorder: () => ({
      isRecording,
      isFinalizingRecording,
      micError: ref(null),
      recordingSeconds: ref(0),
      maxDurationReached,
      maxRecordingSeconds: ref(90),
      audioLevel: ref(0),
      startRecording,
      stopRecording,
      cleanup: vi.fn(),
    }),
  }))
  vi.doMock('~/composables/useTajwid', () => ({
    clearTajwidCache: vi.fn(),
  }))

  const { useRecognitionFlow } = await import('~/composables/useRecognitionFlow')

  return {
    flow: useRecognitionFlow(),
    isRecording,
    isFinalizingRecording,
    maxDurationReached,
    startRecording,
    stopRecording,
    recognizeAudio,
    releaseStop,
  }
}

describe('useRecognitionFlow microphone recording', () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.doUnmock('~/composables/useApiHealth')
    vi.doUnmock('~/composables/useRecognition')
    vi.doUnmock('~/composables/useMicrophoneRecorder')
    vi.doUnmock('~/composables/useTajwid')
  })

  it('waits for a second click before stopping and analyzing the complete recording', async () => {
    vi.useFakeTimers()
    const { flow, isRecording, stopRecording, recognizeAudio } = await setupRecognitionFlow()

    await flow.onMicroClick()
    await vi.advanceTimersByTimeAsync(5_000)

    expect(stopRecording).not.toHaveBeenCalled()
    expect(recognizeAudio).not.toHaveBeenCalled()
    expect(isRecording.value).toBe(true)

    await flow.onMicroClick()

    expect(stopRecording).toHaveBeenCalledTimes(1)
    expect(recognizeAudio).toHaveBeenCalledWith(expect.any(File), false)
    expect(isRecording.value).toBe(false)
  })

  it('submits only once when the stop action is tapped twice during WAV preparation', async () => {
    const { flow, isFinalizingRecording, stopRecording, recognizeAudio, releaseStop } =
      await setupRecognitionFlow({ deferStop: true })

    await flow.onMicroClick()

    const firstStop = flow.onMicroClick()
    const secondStop = flow.onMicroClick()

    expect(stopRecording).toHaveBeenCalledTimes(1)
    expect(isFinalizingRecording.value).toBe(true)

    releaseStop()
    await Promise.all([firstStop, secondStop])

    expect(recognizeAudio).toHaveBeenCalledTimes(1)
    expect(isFinalizingRecording.value).toBe(false)
  })

  it('stops and analyzes automatically only when the maximum duration is reached', async () => {
    const { flow, maxDurationReached, stopRecording, recognizeAudio } = await setupRecognitionFlow()

    await flow.onMicroClick()
    maxDurationReached.value = true
    await nextTick()
    await nextTick()

    expect(stopRecording).toHaveBeenCalledTimes(1)
    expect(recognizeAudio).toHaveBeenCalledWith(expect.any(File), false)
  })

  it('sets an upload error when audio duration reading times out', async () => {
    vi.useFakeTimers()
    const { flow } = await setupRecognitionFlow()
    const file = new File(['corrupt-data'], 'corrupt.mp3', { type: 'audio/mpeg' })

    const submitPromise = flow.submitAudio(file, true)
    await vi.advanceTimersByTimeAsync(4500)
    await submitPromise

    expect(flow.uploadError.value).toBe('Impossible de lire ce fichier audio.')
  })
  it('does not submit audio whose metadata arrives after the flow was reset', async () => {
    const { flow, recognizeAudio } = await setupRecognitionFlow()
    const audio = document.createElement('audio')
    Object.defineProperty(audio, 'duration', { value: 1, configurable: true })
    const createElement = vi.spyOn(document, 'createElement').mockReturnValueOnce(audio)
    const submit = flow.submitAudio(new File(['audio'], 'sample.wav', { type: 'audio/wav' }))
    flow.resetApp()
    audio.dispatchEvent(new Event('loadedmetadata'))
    await submit
    expect(recognizeAudio).not.toHaveBeenCalled()
    expect(flow.uploadError.value).toBeNull()
    createElement.mockRestore()
  })

  it('does not expose a metadata timeout after the flow was reset', async () => {
    vi.useFakeTimers()
    const { flow, recognizeAudio } = await setupRecognitionFlow()
    const submit = flow.submitAudio(new File(['audio'], 'sample.wav', { type: 'audio/wav' }))
    flow.resetApp()
    await vi.advanceTimersByTimeAsync(4500)
    await submit
    expect(flow.uploadError.value).toBeNull()
    expect(recognizeAudio).not.toHaveBeenCalled()
  })
})
