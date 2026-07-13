import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useApiHealth } from '~/composables/useApiHealth'
import { useRecognition } from '~/composables/useRecognition'
import { useMicrophoneRecorder } from '~/composables/useMicrophoneRecorder'
import { clearTajwidCache } from '~/composables/useTajwid'

export type RecognitionScreenState = 'idle' | 'loading' | 'result'

function formatMegabytes(sizeInBytes: number) {
  const megabytes = sizeInBytes / (1024 * 1024)

  if (Number.isInteger(megabytes)) {
    return String(megabytes)
  }

  return megabytes.toFixed(1).replace(/\.0$/, '')
}

function isAllowedAudioType(file: File, allowedMimeTypes: string[]) {
  if (!allowedMimeTypes.length) {
    return true
  }

  return allowedMimeTypes.some((type) => file.type === type || file.type.startsWith(`${type};`))
}

function getAudioDuration(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const audio = document.createElement('audio')
    const objectUrl = URL.createObjectURL(file)

    audio.preload = 'metadata'
    audio.src = objectUrl

    audio.onloadedmetadata = () => {
      const duration = audio.duration
      URL.revokeObjectURL(objectUrl)
      resolve(duration)
    }

    audio.onerror = () => {
      URL.revokeObjectURL(objectUrl)
      reject(new Error('Impossible de lire la durée du fichier audio.'))
    }
  })
}

export function useRecognitionFlow() {
  const {
    imamDetectionAvailable,
    imamDetectionMessage,
    uploadPolicy,
    detectionPolicy,
    refreshHealth,
    markImamDetectionUnavailable,
  } = useApiHealth()

  const maxAudioDurationSeconds = computed(
    () => uploadPolicy.value?.max_audio_duration_seconds ?? null,
  )
  const maxFileSizeBytes = computed(() => uploadPolicy.value?.max_file_size_bytes ?? null)
  const acceptedMimeTypes = computed(() => uploadPolicy.value?.accepted_mime_types ?? [])
  const acceptedFileExtensions = computed(() => uploadPolicy.value?.accepted_file_extensions ?? [])
  const maxFileSizeLabel = computed(() => {
    if (maxFileSizeBytes.value === null) {
      return null
    }

    return formatMegabytes(maxFileSizeBytes.value)
  })
  const uploadAccept = computed(() => {
    if (!acceptedFileExtensions.value.length) {
      return 'audio/*'
    }

    return acceptedFileExtensions.value.join(',')
  })
  const uploadHint = computed(() => {
    if (
      !acceptedFileExtensions.value.length ||
      maxFileSizeLabel.value === null ||
      maxAudioDurationSeconds.value === null
    ) {
      return null
    }

    const formatsLabel = acceptedFileExtensions.value
      .map((extension) => extension.replace(/^\./, ''))
      .join(', ')

    return `Formats : ${formatsLabel} · max ${maxFileSizeLabel.value} Mo · max ${maxAudioDurationSeconds.value} sec`
  })

  const { loading, loadingStep, error, result, recognizeAudio, probeAudio, acceptResult, reset } =
    useRecognition()

  const {
    isRecording,
    micError,
    recordingSeconds,
    maxDurationReached,
    maxRecordingSeconds,
    audioLevel,
    startRecording,
    stopRecording,
    snapshotRecording,
    cleanup,
  } = useMicrophoneRecorder(maxAudioDurationSeconds)

  const uploadError = ref<string | null>(null)
  const detectImam = ref(false)
  let microphoneProbeTimerId: number | null = null
  let microphoneProbeInFlight = false
  let recordingSessionId = 0

  function stopMicrophoneProbeLoop() {
    if (microphoneProbeTimerId !== null) {
      window.clearInterval(microphoneProbeTimerId)
      microphoneProbeTimerId = null
    }
  }

  async function runMicrophoneProbe(sessionId: number) {
    if (microphoneProbeInFlight || sessionId !== recordingSessionId || !isRecording.value) {
      return
    }

    microphoneProbeInFlight = true

    try {
      const snapshot = await snapshotRecording()

      if (!snapshot || sessionId !== recordingSessionId || !isRecording.value) {
        return
      }

      const response = await probeAudio(snapshot, detectImam.value)

      if (
        !response ||
        response.detection?.status !== 'confident' ||
        sessionId !== recordingSessionId ||
        !isRecording.value
      ) {
        return
      }

      stopMicrophoneProbeLoop()
      recordingSessionId += 1
      await stopRecording()
      acceptResult(response)
    } finally {
      microphoneProbeInFlight = false
    }
  }

  function startMicrophoneProbeLoop(sessionId: number) {
    stopMicrophoneProbeLoop()
    const intervalMs = Math.max(1, detectionPolicy.value.progressive_analysis_step_seconds) * 1000

    microphoneProbeTimerId = window.setInterval(() => {
      void runMicrophoneProbe(sessionId)
    }, intervalMs)
  }

  const screenState = computed<RecognitionScreenState>(() => {
    if (loading.value) return 'loading'
    if (result.value || error.value) return 'result'
    return 'idle'
  })

  onMounted(() => {
    void refreshHealth()
  })

  watch(
    imamDetectionAvailable,
    (available) => {
      if (!available) {
        detectImam.value = false
      }
    },
    { immediate: true },
  )

  watch(result, (nextResult) => {
    if (nextResult?.imam_status === 'unavailable') {
      markImamDetectionUnavailable()
      detectImam.value = false
    }
  })

  async function submitAudio(file: File) {
    if (loading.value) return

    uploadError.value = null

    if (!isAllowedAudioType(file, acceptedMimeTypes.value)) {
      uploadError.value = 'Format audio non pris en charge.'
      return
    }

    if (maxFileSizeBytes.value !== null && file.size > maxFileSizeBytes.value) {
      uploadError.value = `Fichier trop volumineux. Maximum ${maxFileSizeLabel.value} Mo.`
      return
    }

    const isManualUpload = !isRecording.value && !maxDurationReached.value

    if (isManualUpload) {
      try {
        const duration = await getAudioDuration(file)

        if (!Number.isFinite(duration) || duration <= 0) {
          uploadError.value = 'Impossible de lire la durée de ce fichier audio.'
          return
        }

        if (maxAudioDurationSeconds.value !== null && duration > maxAudioDurationSeconds.value) {
          uploadError.value = `Audio trop long. Maximum ${maxAudioDurationSeconds.value} secondes.`
          return
        }
      } catch {
        uploadError.value = 'Impossible de lire ce fichier audio.'
        return
      }
    }

    await recognizeAudio(file, detectImam.value)
  }

  async function onMicroClick() {
    uploadError.value = null

    if (!isRecording.value) {
      recordingSessionId += 1
      const sessionId = recordingSessionId
      await startRecording()

      if (isRecording.value && sessionId === recordingSessionId) {
        startMicrophoneProbeLoop(sessionId)
      }
      return
    }

    stopMicrophoneProbeLoop()
    recordingSessionId += 1
    const recordedFile = await stopRecording()

    if (!recordedFile) {
      uploadError.value = 'Erreur pendant l’enregistrement audio.'
      return
    }

    await submitAudio(recordedFile)
  }

  watch(maxDurationReached, async (reached) => {
    if (!reached || loading.value) return

    stopMicrophoneProbeLoop()
    recordingSessionId += 1
    const recordedFile = await stopRecording()

    if (!recordedFile) {
      uploadError.value = 'Erreur pendant l’enregistrement audio.'
      return
    }

    await submitAudio(recordedFile)
  })

  function resetApp() {
    stopMicrophoneProbeLoop()
    recordingSessionId += 1
    uploadError.value = null
    clearTajwidCache()
    cleanup()
    reset()
  }

  onBeforeUnmount(() => {
    stopMicrophoneProbeLoop()
    recordingSessionId += 1
  })

  return {
    loading,
    loadingStep,
    error,
    result,
    uploadError,
    micError,
    isRecording,
    recordingSeconds,
    maxDurationReached,
    maxRecordingSeconds,
    screenState,
    uploadAccept,
    uploadHint,
    submitAudio,
    onMicroClick,
    resetApp,
    audioLevel,
    detectImam,
    imamDetectionAvailable,
    imamDetectionMessage,
  }
}
