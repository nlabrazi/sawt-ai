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
  const snapshotRecording = vi.fn(async () => {
    return new File(['snapshot'], 'snapshot.wav', { type: 'audio/wav' })
  })
  const probeAudio = vi.fn()
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
      probeAudio,
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
      snapshotRecording,
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
    snapshotRecording,
    probeAudio,
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
    const { flow, isRecording, stopRecording, snapshotRecording, probeAudio, recognizeAudio } =
      await setupRecognitionFlow()

    await flow.onMicroClick()
    await vi.advanceTimersByTimeAsync(5_000)

    expect(snapshotRecording).not.toHaveBeenCalled()
    expect(probeAudio).not.toHaveBeenCalled()
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
    const {
      flow,
      maxDurationReached,
      stopRecording,
      snapshotRecording,
      probeAudio,
      recognizeAudio,
    } = await setupRecognitionFlow()

    await flow.onMicroClick()
    maxDurationReached.value = true
    await nextTick()
    await nextTick()

    expect(snapshotRecording).not.toHaveBeenCalled()
    expect(probeAudio).not.toHaveBeenCalled()
    expect(stopRecording).toHaveBeenCalledTimes(1)
    expect(recognizeAudio).toHaveBeenCalledWith(expect.any(File), false)
  })
})
