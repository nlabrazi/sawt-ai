import { useRuntimeConfig } from '#app'
import { $fetch } from 'ofetch'
import { ref } from 'vue'

export type SurahOption = {
  id: number
  name: string
  transliteration: string
  total_verses: number
}

let surahOptionsCache: SurahOption[] | null = null
let surahOptionsPromise: Promise<SurahOption[]> | null = null

export function clearSurahOptionsCache() {
  surahOptionsCache = null
  surahOptionsPromise = null
}

export function useSurahOptions() {
  const apiBaseUrl = useRuntimeConfig().public.apiBaseUrl.replace(/\/$/, '')
  const surahs = ref<SurahOption[]>(surahOptionsCache ?? [])
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function fetchSurahOptions() {
    if (surahOptionsCache) {
      surahs.value = surahOptionsCache
      return surahOptionsCache
    }

    loading.value = true
    error.value = null

    if (!surahOptionsPromise) {
      surahOptionsPromise = $fetch<SurahOption[]>(`${apiBaseUrl}/surahs`)
        .then((response) => {
          surahOptionsCache = [...response].sort((left, right) => left.id - right.id)
          return surahOptionsCache
        })
    }

    try {
      const response = await surahOptionsPromise
      surahs.value = response
      return response
    } catch (err) {
      error.value = 'Impossible de charger la liste des sourates.'
      throw err
    } finally {
      loading.value = false
      surahOptionsPromise = null
    }
  }

  return {
    surahs,
    loading,
    error,
    fetchSurahOptions,
  }
}
