import { useRuntimeConfig } from '#app'
import { $fetch } from 'ofetch'
import { getCurrentScope, onScopeDispose, ref } from 'vue'
import type { TafsirSource } from '~/composables/useTafsirReview'

export type QuranTranslation = {
  surah_id: number
  ayah: number
  text: string
  source: string
  translator: string
  version: string
  source_url: string
  footnotes: string
}

export type VerifiedTafsir = {
  surah_id: number
  ayah: number
  source: TafsirSource
  text_fr: string
  source_reference: string
  version: string
  status: 'verified'
  reviewed_at: string
}

export type QuranAyahContent = {
  ayah: number
  translation: QuranTranslation | null
  tafsirs: VerifiedTafsir[]
}

export type QuranContentResponse = {
  surah_id: number
  start_verse: number
  end_verse: number
  tafsir_status: 'available' | 'unavailable'
  ayahs: QuranAyahContent[]
}

function checkedContent(
  response: QuranContentResponse,
  surahId: number,
  startVerse: number,
  endVerse: number,
): QuranContentResponse {
  if (
    response.surah_id !== surahId ||
    response.start_verse !== startVerse ||
    response.end_verse !== endVerse ||
    !['available', 'unavailable'].includes(response.tafsir_status) ||
    !Array.isArray(response.ayahs) ||
    response.ayahs.length !== endVerse - startVerse + 1
  ) {
    throw new Error('Unexpected Quran content range')
  }

  const ayahs = response.ayahs.map((entry, index) => {
    if (entry.ayah !== startVerse + index) throw new Error('Unexpected ayah')
    const translation =
      entry.translation?.surah_id === surahId && entry.translation.ayah === entry.ayah
        ? entry.translation
        : null
    // Types alone cannot protect a public screen from an unexpected API response.
    const tafsirs =
      response.tafsir_status === 'available' && Array.isArray(entry.tafsirs)
        ? entry.tafsirs.filter(
            (tafsir) =>
              tafsir?.status === 'verified' &&
              tafsir.surah_id === surahId &&
              tafsir.ayah === entry.ayah &&
              ['ibn_kathir', 'as_saadi'].includes(tafsir.source) &&
              typeof tafsir.reviewed_at === 'string' &&
              Number.isFinite(Date.parse(tafsir.reviewed_at)) &&
              typeof tafsir.text_fr === 'string' &&
              tafsir.text_fr.trim().length > 0,
          )
        : []
    return { ayah: entry.ayah, translation, tafsirs }
  })
  return { ...response, ayahs }
}

export function useQuranContent() {
  const apiBaseUrl = useRuntimeConfig().public.apiBaseUrl.replace(/\/$/, '')
  const content = ref<QuranContentResponse | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  let controller: AbortController | null = null

  function clearContent() {
    controller?.abort()
    controller = null
    content.value = null
    loading.value = false
    error.value = null
  }

  async function fetchContent(surahId: number, startVerse: number, endVerse: number) {
    clearContent()
    const activeController = new AbortController()
    controller = activeController
    loading.value = true
    try {
      const response = await $fetch<QuranContentResponse>(`${apiBaseUrl}/quran/content`, {
        method: 'GET',
        query: { surah_id: surahId, start_verse: startVerse, end_verse: endVerse },
        signal: activeController.signal,
        cache: 'no-store',
        retry: 0,
        timeout: 20000,
      })
      if (activeController.signal.aborted || controller !== activeController) return
      content.value = checkedContent(response, surahId, startVerse, endVerse)
    } catch {
      if (!activeController.signal.aborted && controller === activeController) {
        error.value = 'Impossible de charger le contenu français. Réessayez.'
      }
    } finally {
      if (controller === activeController) {
        controller = null
        loading.value = false
      }
    }
  }

  if (getCurrentScope()) onScopeDispose(clearContent)
  return { content, loading, error, fetchContent, clearContent }
}
