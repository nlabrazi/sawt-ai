// ROLE
// ----
// Gère l'appel API vers FastAPI pour envoyer un audio
// et récupérer le résultat de reconnaissance.

export type ImamPrediction = {
  name: string
  score: number
}

export type VerseMatch = {
  sourate_id: number
  sourate_name: string
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

export function useRecognition() {
  const loading = ref(false)
  const error = ref<string | null>(null)
  const result = ref<RecognizeResponse | null>(null)

  async function recognizeAudio(file: File) {
    loading.value = true
    error.value = null
    result.value = null

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await $fetch<RecognizeResponse>('http://localhost:8000/recognize', {
        method: 'POST',
        body: formData,
      })

      result.value = response
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
  }

  return {
    loading,
    error,
    result,
    recognizeAudio,
    reset,
  }
}
