export const LOW_CONFIDENCE_THRESHOLD = 0.8
export const LOW_CONFIDENCE_PERCENT_THRESHOLD = 80
export const PROBABLE_CONFIDENCE_THRESHOLD = 0.6
export const PROBABLE_CONFIDENCE_PERCENT_THRESHOLD = 60

function normalizeSimilarity(similarity: number) {
  return similarity <= 1 ? similarity * 100 : similarity
}

export function isVerseConfident(similarity: number) {
  return normalizeSimilarity(similarity) >= LOW_CONFIDENCE_PERCENT_THRESHOLD
}

export function getVerseConfidenceUi(similarity: number) {
  const percent = normalizeSimilarity(similarity)

  if (percent >= LOW_CONFIDENCE_PERCENT_THRESHOLD) {
    return {
      label: 'Résultat fiable',
      description: 'Le passage détecté semble cohérent avec un bon niveau de certitude.',
      className: 'banner-success',
    }
  }

  if (percent >= PROBABLE_CONFIDENCE_PERCENT_THRESHOLD) {
    return {
      label: 'Correspondance probable',
      description: 'Une correspondance a été trouvée, mais elle nécessite confirmation.',
      className: 'banner-warning',
    }
  }

  return {
    label: 'Résultat à confirmer',
    description: 'Le passage détecté reste plausible, mais mérite une vérification.',
    className: 'banner-error',
  }
}
