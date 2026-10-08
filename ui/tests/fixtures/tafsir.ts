import type { TafsirReviewEntry, TafsirSource } from '../../app/composables/useTafsirReview'

export function tafsirFixture(source: TafsirSource = 'ibn_kathir'): TafsirReviewEntry {
  return {
    surah_id: 2,
    ayah: 255,
    source,
    text_fr: `Brouillon fictif de test (${source}), sans contenu religieux.`,
    source_reference: 'Édition fictive de test, 2:255',
    version: 'test-1',
    status: 'need_review',
    reviewed_at: null,
    updated_at: '2026-10-07T10:00:00Z',
    provenance: {
      source_text: '<img src="x" onerror="window.privateSourceExecuted=true"> Passage fictif.',
      source_edition: 'Édition fictive de test',
      source_language: 'ar',
      reuse_reference: 'Autorisation fictive de test',
      source_start_ayah: 255,
      source_end_ayah: 255,
    },
  }
}
