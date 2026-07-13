function normalizeSimilarity(similarity: number) {
  return similarity <= 1 ? similarity : similarity / 100
}

export function isVerseConfident(similarity: number, minAcceptedSimilarity: number) {
  return normalizeSimilarity(similarity) >= minAcceptedSimilarity
}

export function getVerseConfidenceUi(
  similarity: number,
  minAcceptedSimilarity: number,
  minProbableSimilarity: number,
) {
  const normalizedSimilarity = normalizeSimilarity(similarity)

  if (normalizedSimilarity >= minAcceptedSimilarity) {
    return {
      label: 'Résultat fiable',
      description: 'Le passage détecté semble cohérent avec un bon niveau de certitude.',
      className: 'banner-success',
    }
  }

  if (normalizedSimilarity >= minProbableSimilarity) {
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
