// ROLE
// ----
// Charge le texte tajwid à la demande depuis le backend
// et garde les résultats en cache mémoire.

type TajwidResponse = {
  surah_id: number
  start_verse: number
  end_verse: number
  text: string
}

const tajwidCache = new Map<string, TajwidResponse>()

function buildCacheKey(surahId: number, startVerse: number, endVerse: number) {
  return `${surahId}-${startVerse}-${endVerse}`
}

export function useTajwid() {
  const apiBaseUrl = useRuntimeConfig().public.apiBaseUrl.replace(/\/$/, '')
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function fetchTajwid(surahId: number, startVerse: number, endVerse: number) {
    const cacheKey = buildCacheKey(surahId, startVerse, endVerse)

    if (tajwidCache.has(cacheKey)) {
      return tajwidCache.get(cacheKey)!
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

      tajwidCache.set(cacheKey, response)
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
