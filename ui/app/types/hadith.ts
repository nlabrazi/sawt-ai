export type HadithResult = {
  id: string
  title: string
  arabic: string
  translation: string
  explanation: string | null
  grade: string | null
  attribution: string | null
  source_url: string
  provider: 'HadeethEnc'
}

export type HadithSearchResponse = {
  query: string
  results: HadithResult[]
  search_mode: 'keywords' | 'semantic'
  search_terms: string[]
}
