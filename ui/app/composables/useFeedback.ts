// ROLE
// ----
// Gère l'envoi du feedback utilisateur vers le backend FastAPI.

import type { VerseMatch } from '~/composables/useRecognition'

export type FeedbackCorrection = {
  sourate_name: string
  start_verse: number
  end_verse: number
}

export type FeedbackPayload = {
  is_correct: boolean
  transcription_text: string
  detected_verse: VerseMatch | null
  correction: FeedbackCorrection | null
  comment: string | null
}

export function useFeedback() {
  const sending = ref(false)
  const error = ref<string | null>(null)

  async function sendFeedback(payload: FeedbackPayload) {
    sending.value = true
    error.value = null

    try {
      await $fetch('http://localhost:8000/feedback', {
        method: 'POST',
        body: payload,
      })
    } catch (err) {
      error.value = 'Erreur pendant l’envoi du feedback.'
      console.error(err)
      throw err
    } finally {
      sending.value = false
    }
  }

  return {
    sending,
    error,
    sendFeedback,
  }
}
