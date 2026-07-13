import { ref } from 'vue'

describe('useRecognitionFlow microphone probing', () => {
  it('stops recording for a confident legacy response without metadata', async () => {
    vi.useFakeTimers()
    vi.resetModules()

    const isRecording = ref(false)
    const result = ref(null)
    const startRecording = vi.fn(async () => {
      isRecording.value = true
    })
    const stopRecording = vi.fn(async () => {
      isRecording.value = false
      return new File(['audio'], 'recording.wav', { type: 'audio/wav' })
    })
    const snapshotRecording = vi.fn(async () => {
      return new File(['snapshot'], 'snapshot.wav', { type: 'audio/wav' })
    })
    const confidentResponse = {
      transcription_text: 'قل هو الله احد',
      verse: {
        sourate_id: 112,
        sourate_name: 'الإخلاص',
        transliteration: 'Al-Ikhlas',
        start_verse: 1,
        end_verse: 1,
        text: 'قل هو الله احد',
        similarity: 0.94,
      },
      imam_predictions: [],
      imam_status: 'disabled',
      imam_detection_enabled: false,
    }
    const probeAudio = vi.fn().mockResolvedValue(confidentResponse)
    const acceptResult = vi.fn((response) => {
      result.value = response
    })

    vi.doMock('~/composables/useApiHealth', () => ({
      useApiHealth: () => ({
        imamDetectionAvailable: ref(true),
        imamDetectionMessage: ref(null),
        uploadPolicy: ref(null),
        detectionPolicy: ref({
          min_accepted_similarity: 0.8,
          min_probable_similarity: 0.6,
          min_matched_word_count: 3,
          min_score_margin: 0.08,
          progressive_analysis_step_seconds: 5,
        }),
        refreshHealth: vi.fn(),
        markImamDetectionUnavailable: vi.fn(),
      }),
    }))
    vi.doMock('~/composables/useRecognition', () => ({
      useRecognition: () => ({
        loading: ref(false),
        loadingStep: ref('transcribing'),
        error: ref(null),
        result,
        recognizeAudio: vi.fn(),
        probeAudio,
        acceptResult,
        reset: vi.fn(),
      }),
    }))
    vi.doMock('~/composables/useMicrophoneRecorder', () => ({
      useMicrophoneRecorder: () => ({
        isRecording,
        micError: ref(null),
        recordingSeconds: ref(0),
        maxDurationReached: ref(false),
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
    const flow = useRecognitionFlow()

    await flow.onMicroClick()
    await vi.advanceTimersByTimeAsync(5_000)

    expect(snapshotRecording).toHaveBeenCalledTimes(1)
    expect(probeAudio).toHaveBeenCalledWith(expect.any(File), false)
    expect(stopRecording).toHaveBeenCalledTimes(1)
    expect(acceptResult).toHaveBeenCalledWith(confidentResponse)
    expect(isRecording.value).toBe(false)

    vi.doUnmock('~/composables/useApiHealth')
    vi.doUnmock('~/composables/useRecognition')
    vi.doUnmock('~/composables/useMicrophoneRecorder')
    vi.doUnmock('~/composables/useTajwid')
  })
})
