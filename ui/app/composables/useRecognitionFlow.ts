import { useRecognition } from '~/composables/useRecognition'

export type RecognitionScreenState = 'idle' | 'loading' | 'result'

export function useRecognitionFlow() {
  const {
    loading,
    error,
    result,
    recognizeAudio,
    reset,
  } = useRecognition()

  const screenState = computed<RecognitionScreenState>(() => {
    if (loading.value) return 'loading'
    if (result.value || error.value) return 'result'
    return 'idle'
  })

  async function submitAudio(file: File) {
    if (loading.value) return
    await recognizeAudio(file)
  }

  function onMicroClick() {
    if (import.meta.client) {
      window.alert('Enregistrement micro bientôt disponible.')
    }
  }

  function resetApp() {
    reset()
  }

  return {
    loading,
    error,
    result,
    screenState,
    submitAudio,
    onMicroClick,
    resetApp,
  }
}
