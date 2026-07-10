const ARABIC_INDIC_DIGITS = ['٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩'] as const

export function formatAyahNumber(ayahNumber: number): string {
  return String(ayahNumber).replace(/\d/g, (digit) => ARABIC_INDIC_DIGITS[Number(digit)] ?? digit)
}
