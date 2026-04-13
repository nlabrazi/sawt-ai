export const LOW_CONFIDENCE_THRESHOLD = 0.8
export const LOW_CONFIDENCE_PERCENT_THRESHOLD = 80

export function isVerseConfident(similarity: number) {
  return similarity <= 1
    ? similarity >= LOW_CONFIDENCE_THRESHOLD
    : similarity >= LOW_CONFIDENCE_PERCENT_THRESHOLD
}
