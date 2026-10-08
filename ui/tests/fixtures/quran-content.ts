import type { QuranContentResponse } from '../../app/composables/useQuranContent'

export function emptyQuranContent(
  surahId: number,
  startVerse: number,
  endVerse: number,
): QuranContentResponse {
  return {
    surah_id: surahId,
    start_verse: startVerse,
    end_verse: endVerse,
    tafsir_status: 'available',
    ayahs: Array.from({ length: endVerse - startVerse + 1 }, (_, index) => ({
      ayah: startVerse + index,
      translation: null,
      tafsirs: [],
    })),
  }
}

// Explicitly fictitious texts for UI tests; never imported into the religious corpus.
export function quranContentFixture(
  surahId = 1,
  startVerse = 1,
  endVerse = 2,
): QuranContentResponse {
  const response = emptyQuranContent(surahId, startVerse, endVerse)
  for (const entry of response.ayahs) {
    entry.translation = {
      surah_id: surahId,
      ayah: entry.ayah,
      text: `Traduction fictive ${surahId}:${entry.ayah}.`,
      source: 'Source de test',
      translator: 'Traducteur de test',
      version: 'test-v1',
      source_url: `https://example.com/translation/${surahId}/${entry.ayah}`,
      footnotes: entry.ayah === startVerse ? 'Note fictive <script>test</script>.' : '',
    }
    const sources =
      entry.ayah === startVerse ? (['ibn_kathir', 'as_saadi'] as const) : (['as_saadi'] as const)
    entry.tafsirs = sources.map((source) => ({
      surah_id: surahId,
      ayah: entry.ayah,
      source,
      text_fr: `Commentaire fictif ${source} ${surahId}:${entry.ayah}.`,
      source_reference: `Référence fictive ${source}`,
      version: 'test-v1',
      status: 'verified',
      reviewed_at: '2026-10-07T10:00:00Z',
    }))
  }
  return response
}
