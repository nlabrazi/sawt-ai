import { useRuntimeConfig } from '#app'
import { $fetch } from 'ofetch'
import { getCurrentScope, onScopeDispose, ref } from 'vue'
import type { HadithSearchResponse } from '~/types/hadith'

export function useHadithSearch() {
  const apiBaseUrl = useRuntimeConfig().public.apiBaseUrl.replace(/\/$/, '')
  const query = ref('')
  const response = ref<HadithSearchResponse | null>(null)
  const loading = ref(false)
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
  }

  function reset() {
    cancel()
    query.value = ''
    response.value = null
    error.value = null
    validationError.value = null
    pendingQuery = ''
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

  if (getCurrentScope()) onScopeDispose(cancel)
  return { query, response, loading, error, validationError, search, cancel, reset }
}
