// ROLE
// ----
// Charge le texte tajwid à la demande depuis le backend
// et garde les résultats en cache mémoire.

import { useRuntimeConfig } from '#app'
import { $fetch } from 'ofetch'
import { ref } from 'vue'

export type TajwidAyah = {
  number: number
  tajwid_text: string
}

export type TajwidResponse = {
  surah_id: number
  start_verse: number
  end_verse: number
  text: string
  ayahs: TajwidAyah[]
}

const MAX_TAJWID_CACHE_ENTRIES = 32
const tajwidCache = new Map<string, TajwidResponse>()

function buildCacheKey(surahId: number, startVerse: number, endVerse: number) {
  return `${surahId}-${startVerse}-${endVerse}`
}

function touchCachedResponse(cacheKey: string, response: TajwidResponse) {
  tajwidCache.delete(cacheKey)
  tajwidCache.set(cacheKey, response)
}

function storeCachedResponse(cacheKey: string, response: TajwidResponse) {
  touchCachedResponse(cacheKey, response)

  // Garde un cache borné pour éviter qu'une session longue accumule indéfiniment.
  if (tajwidCache.size <= MAX_TAJWID_CACHE_ENTRIES) {
    return
  }

  const oldestCacheKey = tajwidCache.keys().next().value

  if (oldestCacheKey) {
    tajwidCache.delete(oldestCacheKey)
  }
}

export function clearTajwidCache() {
  tajwidCache.clear()
}

export function useTajwid() {
  const apiBaseUrl = useRuntimeConfig().public.apiBaseUrl.replace(/\/$/, '')
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function fetchTajwid(surahId: number, startVerse: number, endVerse: number) {
    const cacheKey = buildCacheKey(surahId, startVerse, endVerse)
    const cachedResponse = tajwidCache.get(cacheKey)

    if (cachedResponse) {
      touchCachedResponse(cacheKey, cachedResponse)
      return cachedResponse
    }

    loading.value = true
    error.value = null

    try {
      const response = await $fetch<TajwidResponse>(`${apiBaseUrl}/tajwid`, {
        method: 'GET',
        query: {
          surah_id: surahId,
          start_verse: startVerse,
          end_verse: endVerse,
        },
      })

      storeCachedResponse(cacheKey, response)
      return response
    } catch (err) {
      error.value = 'Impossible de charger le tajwid.'
      console.error(err)
      throw err
    } finally {
      loading.value = false
    }
  }

  function getCachedTajwid(surahId: number, startVerse: number, endVerse: number) {
    const cacheKey = buildCacheKey(surahId, startVerse, endVerse)
    return tajwidCache.get(cacheKey) ?? null
  }

  return {
    loading,
    error,
    fetchTajwid,
    getCachedTajwid,
  }
}
