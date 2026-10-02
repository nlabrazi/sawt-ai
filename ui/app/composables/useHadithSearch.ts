import { useRuntimeConfig } from '#app'
import { $fetch } from 'ofetch'
import { getCurrentScope, onScopeDispose, ref } from 'vue'
import type { HadithSearchResponse } from '~/types/hadith'

export function useHadithSearch() {
  const apiBaseUrl = useRuntimeConfig().public.apiBaseUrl.replace(/\/$/, '')
  const query = ref('')
  const response = ref<HadithSearchResponse | null>(null)
  const loading = ref(false)
  const transcribing = ref(false)
  const transcriptionError = ref<string | null>(null)
  const error = ref<string | null>(null)
  const validationError = ref<string | null>(null)
  let controller: AbortController | null = null
  let requestId = 0
  let pendingQuery = ''

  function cancel() {
    requestId += 1
    controller?.abort()
    controller = null
    loading.value = false
    transcribing.value = false
  }

  function reset() {
    cancel()
    query.value = ''
    response.value = null
    error.value = null
    validationError.value = null
    pendingQuery = ''
    transcriptionError.value = null
  }

  async function search() {
    const text = query.value.trim()
    validationError.value =
      text.length < 3
        ? 'Décrivez votre recherche en au moins 3 caractères.'
        : text.length > 300
          ? 'Votre recherche doit contenir au maximum 300 caractères.'
          : null
    if (validationError.value || (loading.value && text === pendingQuery)) return

    cancel()
    const activeId = requestId
    const activeController = new AbortController()
    controller = activeController
    pendingQuery = text
    loading.value = true
    error.value = null
    transcriptionError.value = null
    response.value = null

    try {
      const result = await $fetch<HadithSearchResponse>(`${apiBaseUrl}/hadith/search`, {
        method: 'POST',
        body: { query: text, limit: 3 },
        signal: activeController.signal,
        timeout: 60_000,
        retry: 0,
      })
      if (activeId === requestId) response.value = result
    } catch (err) {
      if (activeId !== requestId || activeController.signal.aborted) return
      const status = (err as { statusCode?: number })?.statusCode
      error.value =
        status === 503
          ? 'La recherche de hadiths est temporairement indisponible. Réessayez dans un instant.'
          : status === 422
            ? 'Cette recherche n’a pas pu être acceptée. Vérifiez votre formulation.'
            : 'Impossible de terminer la recherche. Vérifiez votre connexion et réessayez.'
    } finally {
      if (activeId === requestId) {
        loading.value = false
        controller = null
      }
    }
  }

  async function transcribeAndSearch(file: File) {
    if (loading.value) return
    cancel()
    const activeId = requestId
    const activeController = new AbortController()
    controller = activeController
    loading.value = true
    transcribing.value = true
    transcriptionError.value = null
    validationError.value = null
    error.value = null
    response.value = null

    const body = new FormData()
    body.append('file', file)
    try {
      const result = await $fetch<{ query: string }>(`${apiBaseUrl}/hadith/transcribe`, {
        method: 'POST',
        body,
        signal: activeController.signal,
        timeout: 60_000,
        retry: 0,
      })
      if (activeId !== requestId) return
      query.value = result.query
      loading.value = false
      transcribing.value = false
      await search()
    } catch (err) {
      if (activeId !== requestId || activeController.signal.aborted) return
      const status = (err as { statusCode?: number })?.statusCode
      transcriptionError.value =
        status === 422
          ? 'Aucune demande exploitable. Réessayez en parlant clairement en français avec une phrase courte.'
          : status === 413
            ? 'Enregistrement trop long ou trop volumineux. Limitez votre demande à 30 secondes.'
            : status === 400 || status === 415
              ? 'Impossible de lire cet enregistrement. Réessayez ou saisissez votre recherche.'
              : status === 503
                ? 'La recherche vocale est temporairement indisponible. Vous pouvez saisir votre recherche.'
                : 'Impossible de transcrire votre demande. Vérifiez votre connexion et réessayez.'
    } finally {
      if (activeId === requestId) {
        loading.value = false
        transcribing.value = false
        controller = null
      }
    }
  }

  if (getCurrentScope()) onScopeDispose(cancel)
  return {
    query,
    response,
    loading,
    transcribing,
    transcriptionError,
    error,
    validationError,
    search,
    transcribeAndSearch,
    cancel,
    reset,
  }
}
