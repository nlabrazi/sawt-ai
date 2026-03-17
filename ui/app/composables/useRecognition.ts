// ROLE
// ----
// Gère l'appel API vers FastAPI pour envoyer un audio,
// piloter le loader visuel et récupérer le résultat.

export type ImamPrediction = {
  name: string
  score: number
}

export type VerseMatch = {
  sourate_id: number
  sourate_name: string
  transliteration: string
  start_verse: number
  end_verse: number
  text: string
  similarity: number
}

export type RecognizeResponse = {
  transcription_text: string
  verse: VerseMatch | null
  imam_predictions: ImamPrediction[]
  imam_status: string
}

export type LoadingStep = 'detecting' | 'result-found' | 'retrying'

const MIN_LOADING_MS = 1800
const RESULT_FOUND_STEP_MS = 850
const RETRY_STEP_MS = 1100
const LOW_CONFIDENCE_THRESHOLD = 0.8

function wait(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

export function useRecognition() {
  const apiBaseUrl = useRuntimeConfig().public.apiBaseUrl.replace(/\/$/, '')
  const loading = ref(false)
  const error = ref<string | null>(null)
  const result = ref<RecognizeResponse | null>(null)
  const loadingStep = ref<LoadingStep>('detecting')

  async function recognizeAudio(file: File) {
    loading.value = true
    error.value = null
    result.value = null
    loadingStep.value = 'detecting'

    const startedAt = Date.now()

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await $fetch<RecognizeResponse>(`${apiBaseUrl}/recognize`, {
        method: 'POST',
        body: formData,
      })

      const elapsed = Date.now() - startedAt
      const remaining = Math.max(0, MIN_LOADING_MS - elapsed)

      if (remaining > 0) {
        await wait(remaining)
      }

      const hasVerse = !!response.verse
      const similarity = response.verse?.similarity ?? 0

      const isConfident = similarity <= 1
        ? similarity >= LOW_CONFIDENCE_THRESHOLD
        : similarity >= 80

      if (hasVerse && isConfident) {
        loadingStep.value = 'result-found'
        await wait(RESULT_FOUND_STEP_MS)
      } else {
        loadingStep.value = 'retrying'
        await wait(RETRY_STEP_MS)
      }

      result.value = response

      if (!hasVerse) {
        error.value = 'Aucun verset fiable trouvé pour cet audio.'
      }
    } catch (err) {
      error.value = 'Erreur pendant la reconnaissance audio.'
      console.error(err)
    } finally {
      loading.value = false
    }
  }

  function reset() {
    loading.value = false
    error.value = null
    result.value = null
    loadingStep.value = 'detecting'
  }

  return {
    loading,
    loadingStep,
    error,
    result,
    recognizeAudio,
    reset,
  }
}
