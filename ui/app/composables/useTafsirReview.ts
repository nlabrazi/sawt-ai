import { useRuntimeConfig } from '#app'
import { $fetch, type FetchOptions } from 'ofetch'
import { getCurrentScope, onScopeDispose, ref } from 'vue'

export type TafsirSource = 'ibn_kathir' | 'as_saadi'
export type TafsirReviewEntry = {
  surah_id: number
  ayah: number
  source: TafsirSource
  text_fr: string
  source_reference: string
  version: string
  status: 'need_review' | 'verified'
  reviewed_at: string | null
  updated_at: string
  provenance: {
    source_text?: string
    source_edition?: string
    source_language?: string
    reuse_reference?: string
    source_start_ayah?: number
    source_end_ayah?: number
  }
}

export function useTafsirReview() {
  const apiBaseUrl = useRuntimeConfig().public.apiBaseUrl.replace(/\/$/, '')
  const authenticated = ref(false)
  const busy = ref(false)
  const error = ref<string | null>(null)
  const conflict = ref(false)
  // Component-scoped memory only: never put this in useState, storage or a URL.
  let password = ''
  let controller: AbortController | null = null

  function logout() {
    controller?.abort()
    controller = null
    password = ''
    authenticated.value = false
    busy.value = false
    error.value = null
    conflict.value = false
  }

  async function request<T>(path: string, options: FetchOptions<'json'> = {}) {
    if (!password || busy.value) return undefined
    const activeController = new AbortController()
    controller = activeController
    busy.value = true
    error.value = null
    try {
      const result = await $fetch<T>(`${apiBaseUrl}/internal/tafsir${path}`, {
        ...options,
        headers: { Authorization: `Bearer ${password}` },
        signal: activeController.signal,
        cache: 'no-store',
        retry: 0,
        timeout: 20000,
      })
      return activeController.signal.aborted ? undefined : result
    } catch (err) {
      if (activeController.signal.aborted) return undefined
      const status = (err as { statusCode?: number }).statusCode
      if (status === 401) {
        logout()
        error.value = 'Mot de passe incorrect ou accès révoqué. Reconnectez-vous.'
      } else if (status === 409) {
        conflict.value = true
        error.value =
          'Ce tafsir a changé. Rechargez la liste puis relisez le texte avant de continuer.'
      } else if (status === 503) {
        error.value = 'L’accès ou le stockage de la review n’est pas configuré.'
      } else {
        error.value = 'Impossible de terminer cette action. Réessayez.'
      }
      // Do not log fetch errors: they can contain the Authorization header.
      return undefined
    } finally {
      if (controller === activeController) {
        controller = null
        busy.value = false
      }
    }
  }

  async function login(value: string) {
    logout()
    password = value
    if (!password.trim()) {
      error.value = 'Saisissez le mot de passe interne.'
      password = ''
      return false
    }
    await request('/access')
    authenticated.value = Boolean(password) && !error.value
    if (!authenticated.value) password = ''
    return authenticated.value
  }

  async function list(query: {
    surah_id?: number
    source?: TafsirSource
    status: TafsirReviewEntry['status']
    limit: number
    offset: number
  }) {
    const entries = await request<TafsirReviewEntry[]>('', { query })
    if (entries) conflict.value = false
    return entries
  }

  function save(entry: TafsirReviewEntry, text: string) {
    if (conflict.value) return undefined
    return request<TafsirReviewEntry>(`/${entry.surah_id}/${entry.ayah}/${entry.source}`, {
      method: 'PATCH',
      body: { text_fr: text, expected_updated_at: entry.updated_at },
    })
  }

  function verify(entry: TafsirReviewEntry) {
    if (conflict.value || entry.status !== 'need_review') return undefined
    return request<TafsirReviewEntry>(`/${entry.surah_id}/${entry.ayah}/${entry.source}/verify`, {
      method: 'POST',
      body: { expected_updated_at: entry.updated_at },
    })
  }

  if (getCurrentScope()) onScopeDispose(logout)
  return { authenticated, busy, error, conflict, login, logout, list, save, verify }
}
