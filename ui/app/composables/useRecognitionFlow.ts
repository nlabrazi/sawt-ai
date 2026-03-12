import { useRecognition } from '~/composables/useRecognition'
import { useMicrophoneRecorder } from '~/composables/useMicrophoneRecorder'

export type RecognitionScreenState = 'idle' | 'loading' | 'result'

const MAX_FILE_SIZE_MB = 12
const MAX_AUDIO_DURATION_SECONDS = 90
const ALLOWED_MIME_TYPES = [
  'audio/wav',
  'audio/x-wav',
  'audio/mpeg',
  'audio/mp3',
  'audio/mp4',
  'audio/x-m4a',
  'audio/ogg',
  'audio/webm',
]

function getFileSizeInMb(file: File) {
  return file.size / (1024 * 1024)
}

function isAllowedAudioType(file: File) {
  return ALLOWED_MIME_TYPES.some(type =>
    file.type === type || file.type.startsWith(`${type};`)
  )
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
    loading,
    loadingStep,
    error,
    result,
    recognizeAudio,
    reset,
  } = useRecognition()

  const {
    isRecording,
    micError,
    recordingSeconds,
    maxDurationReached,
    maxRecordingSeconds,
    audioLevel,
    startRecording,
    stopRecording,
    cleanup,
  } = useMicrophoneRecorder()

  const uploadError = ref<string | null>(null)

  const screenState = computed<RecognitionScreenState>(() => {
    if (loading.value) return 'loading'
    if (result.value || error.value) return 'result'
    return 'idle'
  })

  async function submitAudio(file: File) {
    if (loading.value) return

    uploadError.value = null

    if (!isAllowedAudioType(file)) {
      uploadError.value = 'Format audio non pris en charge.'
      return
    }

    if (getFileSizeInMb(file) > MAX_FILE_SIZE_MB) {
      uploadError.value = `Fichier trop volumineux. Maximum ${MAX_FILE_SIZE_MB} Mo.`
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

        if (duration > MAX_AUDIO_DURATION_SECONDS) {
          uploadError.value = `Audio trop long. Maximum ${MAX_AUDIO_DURATION_SECONDS} secondes.`
          return
        }
      } catch {
        uploadError.value = 'Impossible de lire ce fichier audio.'
        return
      }
    }

    await recognizeAudio(file)
  }

  async function onMicroClick() {
    uploadError.value = null

    if (!isRecording.value) {
      await startRecording()
      return
    }

    const recordedFile = await stopRecording()

    if (!recordedFile) {
      uploadError.value = 'Erreur pendant l’enregistrement audio.'
      return
    }

    await recognizeAudio(recordedFile)
  }

  watch(maxDurationReached, async (reached) => {
    if (!reached || loading.value) return

    const recordedFile = await stopRecording()

    if (!recordedFile) {
      uploadError.value = 'Erreur pendant l’enregistrement audio.'
      return
    }

    await recognizeAudio(recordedFile)
  })

  function resetApp() {
    uploadError.value = null
    cleanup()
    reset()
  }

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
    submitAudio,
    onMicroClick,
    resetApp,
    audioLevel,
  }
}
