// ROLE
// ----
// Gère l'appel API vers FastAPI pour envoyer un audio,
// piloter le loader visuel et récupérer le résultat.

import { useRuntimeConfig } from '#app'
import { $fetch } from 'ofetch'
import { ref } from 'vue'
import { useApiHealth } from '~/composables/useApiHealth'
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

export type DetectionStatus = 'confident' | 'probable' | 'ambiguous' | 'insufficient'

export type VerseDetectionMetadata = {
  status: DetectionStatus
  score: number | null
  score_margin: number | null
  matched_word_count: number
  analyzed_duration_seconds: number | null
  analysis_attempts: number
  rejection_reason:
    | 'no_match'
    | 'score_too_low'
    | 'transcription_too_short'
    | 'ambiguous_match'
    | null
}

export type RecognizeResponse = {
  transcription_text: string
  verse: VerseMatch | null
  detection?: VerseDetectionMetadata
  imam_predictions: ImamPrediction[]
  imam_status: ImamStatus
  imam_detection_enabled: boolean
}

export type LoadingStep = 'transcribing' | 'matching' | 'done'

const MIN_TRANSCRIBING_STEP_MS = 350
const CONFIDENT_LOADING_TARGET_MS = 1200
const UNCERTAIN_LOADING_TARGET_MS = 1500
const COMPACT_MATCHING_STEP_MS = 160
const DONE_STEP_MS = 160

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

function getElapsedMs(startedAt: number) {
  return Math.max(0, Date.now() - startedAt)
}

function resolveLoadingTargetMs(hasConfidentVerse: boolean) {
  return hasConfidentVerse ? CONFIDENT_LOADING_TARGET_MS : UNCERTAIN_LOADING_TARGET_MS
}

function resolveTranscribingDelayMs(startedAt: number, targetMs: number) {
  const elapsed = getElapsedMs(startedAt)
  const minimumDelay = Math.max(0, MIN_TRANSCRIBING_STEP_MS - elapsed)
  const remainingBeforeCompactSteps = Math.max(
    0,
    targetMs - elapsed - COMPACT_MATCHING_STEP_MS - DONE_STEP_MS,
  )

  return Math.min(minimumDelay, remainingBeforeCompactSteps)
}

function resolveMatchingDelayMs(startedAt: number, targetMs: number) {
  const remainingBeforeDone = Math.max(0, targetMs - getElapsedMs(startedAt) - DONE_STEP_MS)

  return Math.max(COMPACT_MATCHING_STEP_MS, remainingBeforeDone)
}

export function useRecognition() {
  const apiBaseUrl = useRuntimeConfig().public.apiBaseUrl.replace(/\/$/, '')
  const { detectionPolicy } = useApiHealth()
  const loading = ref(false)
  const error = ref<string | null>(null)
  const result = ref<RecognizeResponse | null>(null)
  const loadingStep = ref<LoadingStep>('transcribing')
  let activeController: AbortController | null = null
  let probeController: AbortController | null = null
  let activeRequestId = 0

  function cancelActiveRequest() {
    activeController?.abort()
    activeController = null
  }

  function isActiveRequest(requestId: number) {
    return requestId === activeRequestId
  }

  function buildRecognitionFormData(file: File, detectImam: boolean) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('detect_imam', String(detectImam))
    return formData
  }

  async function probeAudio(file: File, detectImam = false) {
    probeController?.abort()
    const controller = new AbortController()
    probeController = controller

    try {
      return await $fetch<RecognizeResponse>(`${apiBaseUrl}/recognize`, {
        method: 'POST',
        body: buildRecognitionFormData(file, detectImam),
        signal: controller.signal,
      })
    } catch (err) {
      if (!isAbortError(err)) {
        console.error(err)
      }

      return null
    } finally {
      if (probeController === controller) {
        probeController = null
      }
    }
  }

  function acceptResult(response: RecognizeResponse) {
    probeController?.abort()
    probeController = null
    error.value = response.verse ? null : 'Aucun verset fiable trouvé pour cet audio.'
    result.value = response
  }

  async function recognizeAudio(file: File, detectImam = false) {
    cancelActiveRequest()
    probeController?.abort()
    probeController = null
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
      const response = await $fetch<RecognizeResponse>(`${apiBaseUrl}/recognize`, {
        method: 'POST',
        body: buildRecognitionFormData(file, detectImam),
        signal: controller.signal,
      })

      const hasVerse = !!response.verse
      const isConfident = response.detection
        ? response.detection.status === 'confident'
        : isVerseConfident(
            response.verse?.similarity ?? 0,
            detectionPolicy.value.min_accepted_similarity,
          )
      const loadingTargetMs = resolveLoadingTargetMs(hasVerse && isConfident)
      const transcribingDelayMs = resolveTranscribingDelayMs(startedAt, loadingTargetMs)

      if (transcribingDelayMs > 0) {
        await wait(transcribingDelayMs, controller.signal)
      }

      loadingStep.value = 'matching'
      await wait(resolveMatchingDelayMs(startedAt, loadingTargetMs), controller.signal)

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
    probeController?.abort()
    probeController = null
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
    probeAudio,
    acceptResult,
    reset,
  }
}
