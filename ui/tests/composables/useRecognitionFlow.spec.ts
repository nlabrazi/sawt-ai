import { mount } from '@vue/test-utils'
import { defineComponent, nextTick, ref } from 'vue'

interface SetupOptions {
  deferStart?: boolean
  deferStop?: boolean
  stopResult?: File | null
  uploadPolicy?: {
    max_audio_duration_seconds: number
    max_file_size_bytes: number
    accepted_mime_types: string[]
    accepted_file_extensions: string[]
  } | null
}

async function setupRecognitionFlow(options: SetupOptions = {}) {
  vi.resetModules()

  const imamDetectionAvailable = ref(true)
  const uploadPolicy = ref(options.uploadPolicy ?? null)
  const refreshHealth = vi.fn()
  const markImamDetectionUnavailable = vi.fn()

  const loading = ref(false)
  const error = ref<string | null>(null)
  const result = ref<{ imam_status?: string } | null>(null)
  const recognizeAudio = vi.fn()
  const reset = vi.fn()

  const isRecording = ref(false)
  const isFinalizingRecording = ref(false)
  const maxDurationReached = ref(false)
  let releaseStart = () => undefined
  const startGate = new Promise<void>((resolve) => {
    releaseStart = resolve
  })
  let releaseStop = () => undefined
  const stopGate = new Promise<void>((resolve) => {
    releaseStop = resolve
  })
  const startRecording = vi.fn(async () => {
    if (options.deferStart) {
      await startGate
    }
    isRecording.value = true
  })
  const stopRecording = vi.fn(async () => {
    isRecording.value = false
    isFinalizingRecording.value = true

    if (options.deferStop) {
      await stopGate
    }

    isFinalizingRecording.value = false
    if ('stopResult' in options) {
      return options.stopResult ?? null
    }
    return new File(['audio'], 'recording.wav', { type: 'audio/wav' })
  })
  const cleanup = vi.fn()
  const clearTajwidCache = vi.fn()

  vi.doMock('~/composables/useApiHealth', () => ({
    useApiHealth: () => ({
      imamDetectionAvailable,
      imamDetectionMessage: ref(null),
      uploadPolicy,
      refreshHealth,
      markImamDetectionUnavailable,
    }),
  }))
  vi.doMock('~/composables/useRecognition', () => ({
    useRecognition: () => ({
      loading,
      loadingStep: ref('transcribing'),
      error,
      result,
      recognizeAudio,
      reset,
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
      cleanup,
    }),
  }))
  vi.doMock('~/composables/useTajwid', () => ({
    clearTajwidCache,
  }))

  const { useRecognitionFlow } = await import('~/composables/useRecognitionFlow')

  let flow!: ReturnType<typeof useRecognitionFlow>
  const wrapper = mount(
    defineComponent({
      setup() {
        flow = useRecognitionFlow()
        return () => null
      },
    }),
  )

  return {
    flow,
    wrapper,
    imamDetectionAvailable,
    uploadPolicy,
    refreshHealth,
    markImamDetectionUnavailable,
    loading,
    error,
    result,
    isRecording,
    isFinalizingRecording,
    maxDurationReached,
    startRecording,
    stopRecording,
    recognizeAudio,
    reset,
    cleanup,
    clearTajwidCache,
    releaseStart,
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

  it('computes uploadAccept, uploadHint, screenState and refreshes health on mount', async () => {
    const { flow, refreshHealth, uploadPolicy, loading, error, result } =
      await setupRecognitionFlow()

    expect(refreshHealth).toHaveBeenCalledTimes(1)
    expect(flow.uploadAccept.value).toBe('audio/*')
    expect(flow.uploadHint.value).toBeNull()
    expect(flow.screenState.value).toBe('idle')

    uploadPolicy.value = {
      max_audio_duration_seconds: 60,
      max_file_size_bytes: 5 * 1024 * 1024,
      accepted_mime_types: ['audio/mpeg', 'audio/wav'],
      accepted_file_extensions: ['.mp3', '.wav'],
    }
    expect(flow.uploadAccept.value).toBe('.mp3,.wav')
    expect(flow.uploadHint.value).toBe('Formats : mp3, wav · max 5 Mo · max 60 sec')

    uploadPolicy.value = {
      ...uploadPolicy.value,
      max_file_size_bytes: Math.round(1.5 * 1024 * 1024),
    }
    expect(flow.uploadHint.value).toBe('Formats : mp3, wav · max 1.5 Mo · max 60 sec')

    loading.value = true
    expect(flow.screenState.value).toBe('loading')
    loading.value = false
    error.value = 'network error'
    expect(flow.screenState.value).toBe('result')
    error.value = null
    result.value = { imam_status: 'available' }
    expect(flow.screenState.value).toBe('result')
  })

  it('disables imam detection when health or recognition result marks it unavailable', async () => {
    const { flow, imamDetectionAvailable, result, markImamDetectionUnavailable } =
      await setupRecognitionFlow()

    flow.detectImam.value = true
    imamDetectionAvailable.value = false
    await nextTick()
    expect(flow.detectImam.value).toBe(false)

    imamDetectionAvailable.value = true
    flow.detectImam.value = true
    result.value = { imam_status: 'unavailable' }
    await nextTick()
    expect(markImamDetectionUnavailable).toHaveBeenCalledTimes(1)
    expect(flow.detectImam.value).toBe(false)
  })

  it('validates MIME type, file size, audio duration and handles audio metadata errors', async () => {
    const { flow, loading, recognizeAudio } = await setupRecognitionFlow({
      uploadPolicy: {
        max_audio_duration_seconds: 30,
        max_file_size_bytes: 1024 * 1024,
        accepted_mime_types: ['audio/wav', 'audio/webm'],
        accepted_file_extensions: ['.wav', '.webm'],
      },
    })

    // Ignored while loading
    loading.value = true
    await flow.submitAudio(new File(['a'], 'a.wav', { type: 'audio/wav' }), false)
    await flow.onMicroClick()
    expect(recognizeAudio).not.toHaveBeenCalled()
    loading.value = false

    // Unsupported MIME type
    await flow.submitAudio(new File(['a'], 'doc.txt', { type: 'text/plain' }))
    expect(flow.uploadError.value).toBe('Format audio non pris en charge.')

    // File too large
    const bigFile = new File([new Uint8Array(1024 * 1024 + 1)], 'big.wav', { type: 'audio/wav' })
    await flow.submitAudio(bigFile)
    expect(flow.uploadError.value).toBe('Fichier trop volumineux. Maximum 1 Mo.')

    // Duration <= 0 / NaN
    const zeroAudio = document.createElement('audio')
    Object.defineProperty(zeroAudio, 'duration', { value: 0, configurable: true })
    const createElSpy = vi.spyOn(document, 'createElement').mockReturnValueOnce(zeroAudio)
    const zeroPromise = flow.submitAudio(new File(['a'], 'zero.wav', { type: 'audio/wav' }))
    zeroAudio.dispatchEvent(new Event('loadedmetadata'))
    await zeroPromise
    expect(flow.uploadError.value).toBe('Impossible de lire la durée de ce fichier audio.')

    // Duration exceeds max_audio_duration_seconds
    const longAudio = document.createElement('audio')
    Object.defineProperty(longAudio, 'duration', { value: 45, configurable: true })
    createElSpy.mockReturnValueOnce(longAudio)
    const longPromise = flow.submitAudio(new File(['a'], 'long.wav', { type: 'audio/wav' }))
    longAudio.dispatchEvent(new Event('loadedmetadata'))
    await longPromise
    expect(flow.uploadError.value).toBe('Audio trop long. Maximum 30 secondes.')

    // Audio element error event
    const brokenAudio = document.createElement('audio')
    createElSpy.mockReturnValueOnce(brokenAudio)
    const errPromise = flow.submitAudio(new File(['a'], 'bad.wav', { type: 'audio/wav' }))
    brokenAudio.dispatchEvent(new Event('error'))
    await errPromise
    expect(flow.uploadError.value).toBe('Impossible de lire ce fichier audio.')

    // Valid MIME with codec parameter and valid duration submits cleanly
    const okAudio = document.createElement('audio')
    Object.defineProperty(okAudio, 'duration', { value: 12, configurable: true })
    createElSpy.mockReturnValueOnce(okAudio)
    const okFile = new File(['audio'], 'ok.webm', { type: 'audio/webm;codecs=opus' })
    const okPromise = flow.submitAudio(okFile)
    okAudio.dispatchEvent(new Event('loadedmetadata'))
    await okPromise
    expect(flow.uploadError.value).toBeNull()
    expect(recognizeAudio).toHaveBeenCalledWith(okFile, false)
    createElSpy.mockRestore()
  })

  it('deduplicates concurrent start clicks and reports an error when stopRecording yields null', async () => {
    const { flow, startRecording, releaseStart } = await setupRecognitionFlow({
      deferStart: true,
      stopResult: null,
    })

    const firstStart = flow.onMicroClick()
    const secondStart = flow.onMicroClick()
    expect(startRecording).toHaveBeenCalledTimes(1)

    releaseStart()
    await Promise.all([firstStart, secondStart])

    await flow.onMicroClick()
    expect(flow.uploadError.value).toBe('Erreur pendant l’enregistrement audio.')
  })
})
