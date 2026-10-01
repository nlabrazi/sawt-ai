import type { HadithResult } from '~/types/hadith'

// Controlled fixture: no network request or retrieval-quality assertion.
export const hadithFixture: HadithResult = {
  id: '4709',
  title: 'Ne te mets pas colère !',
  arabic: 'لَا تَغْضَبْ',
  translation: 'Texte français témoin.\nDeuxième ligne conservée.',
  explanation: 'Explication témoin.',
  grade: 'Authentique',
  attribution: 'Attribution témoin.',
  source_url: 'https://hadeethenc.com/fr/browse/hadith/4709',
  provider: 'HadeethEnc',
}
