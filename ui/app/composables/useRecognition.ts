// ROLE
// ----
// Gère l'appel API vers FastAPI pour envoyer un audio,
// piloter le loader visuel et récupérer le résultat.

import { useRuntimeConfig } from '#app'
import { $fetch } from 'ofetch'
import { ref } from 'vue'
import { isVerseConfident } from '~/utils/verseConfidence'

export type ImamPrediction = {
  name: string
  score: number
}

export type ImamStatus = 'disabled' | 'unknown' | 'unavailable' | 'high' | 'medium' | 'low'

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
  imam_status: ImamStatus
  imam_detection_enabled: boolean
}

export type LoadingStep = 'transcribing' | 'matching' | 'done'

const MIN_LOADING_MS = 1800
const MATCHING_STEP_MS = 850
const LOW_CONFIDENCE_MATCHING_STEP_MS = 1100
const DONE_STEP_MS = 250

function createAbortError() {
  const error = new Error('The operation was aborted.')
  error.name = 'AbortError'
  return error
}

function isAbortError(error: unknown) {
  if (error instanceof Error && error.name === 'AbortError') {
    return true
  }

  if (error && typeof error === 'object' && 'cause' in error) {
    return isAbortError(error.cause)
  }

  return false
}

function getApiErrorDetail(error: unknown) {
  if (!error || typeof error !== 'object' || !('data' in error)) {
    return null
  }

  const data = (error as { data?: unknown }).data

  if (!data || typeof data !== 'object' || !('detail' in data)) {
    return null
  }

  return typeof (data as { detail?: unknown }).detail === 'string'
    ? (data as { detail: string }).detail
    : null
}

function wait(ms: number, signal?: AbortSignal) {
  if (signal?.aborted) {
    return Promise.reject(createAbortError())
  }

  return new Promise<void>((resolve, reject) => {
    const timeoutId = window.setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, ms)

    const onAbort = () => {
      window.clearTimeout(timeoutId)
      signal?.removeEventListener('abort', onAbort)
      reject(createAbortError())
    }

    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

export function useRecognition() {
  const apiBaseUrl = useRuntimeConfig().public.apiBaseUrl.replace(/\/$/, '')
  const loading = ref(false)
  const error = ref<string | null>(null)
  const result = ref<RecognizeResponse | null>(null)
  const loadingStep = ref<LoadingStep>('transcribing')
  let activeController: AbortController | null = null
  let activeRequestId = 0

  function cancelActiveRequest() {
    activeController?.abort()
    activeController = null
  }

  function isActiveRequest(requestId: number) {
    return requestId === activeRequestId
  }

  async function recognizeAudio(file: File, detectImam = true) {
    cancelActiveRequest()
    const controller = new AbortController()
    const requestId = activeRequestId + 1

    activeController = controller
    activeRequestId = requestId
    loading.value = true
    error.value = null
    result.value = null
    loadingStep.value = 'transcribing'

    const startedAt = Date.now()

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('detect_imam', String(detectImam))

      const response = await $fetch<RecognizeResponse>(`${apiBaseUrl}/recognize`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })

      const elapsed = Date.now() - startedAt
      const remaining = Math.max(0, MIN_LOADING_MS - elapsed)

      if (remaining > 0) {
        await wait(remaining, controller.signal)
      }

      const hasVerse = !!response.verse
      const isConfident = isVerseConfident(response.verse?.similarity ?? 0)

      loadingStep.value = 'matching'
      await wait(
        hasVerse && isConfident ? MATCHING_STEP_MS : LOW_CONFIDENCE_MATCHING_STEP_MS,
        controller.signal,
      )

      loadingStep.value = 'done'
      await wait(DONE_STEP_MS, controller.signal)

      if (!isActiveRequest(requestId)) {
        return
      }

      result.value = response

      if (!hasVerse) {
        error.value = 'Aucun verset fiable trouvé pour cet audio.'
      }
    } catch (err) {
      if (!isActiveRequest(requestId) || isAbortError(err)) {
        return
      }

      error.value = getApiErrorDetail(err) ?? 'Erreur pendant la reconnaissance audio.'
      console.error(err)
    } finally {
      if (isActiveRequest(requestId)) {
        loading.value = false
        activeController = null
      }
    }
  }

  function reset() {
    activeRequestId += 1
    cancelActiveRequest()
    loading.value = false
    error.value = null
    result.value = null
    loadingStep.value = 'transcribing'
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
