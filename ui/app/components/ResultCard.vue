<script setup lang="ts">
import Eye from '@lucide/vue/dist/esm/icons/eye.mjs'
import { computed, defineAsyncComponent, ref } from 'vue'

import { useApiHealth } from '~/composables/useApiHealth'
import type { RecognizeResponse } from '~/composables/useRecognition'
import { getVerseConfidenceUi } from '~/utils/verseConfidence'

const loadVerseDetailsSheet = () => import('~/components/VerseDetailsSheet.vue')
const VerseDetailsSheet = defineAsyncComponent(loadVerseDetailsSheet)

const props = defineProps<{
  result: RecognizeResponse
}>()
const { detectionPolicy } = useApiHealth()

const isDetailsOpen = ref(false)

const verseLabel = computed(() => {
  if (!props.result.verse) return ''

  const { start_verse, end_verse } = props.result.verse
  return start_verse === end_verse
    ? `Verset ${start_verse}`
    : `Versets ${start_verse} à ${end_verse}`
})

const topImam = computed(() => props.result.imam_predictions?.[0] ?? null)
const confidenceUi = computed(() => {
  if (props.result.detection?.status === 'ambiguous') {
    return {
      label: 'Correspondance à confirmer',
      description: 'Le score est élevé, mais un passage voisin reste également plausible.',
      className: 'banner-warning',
    }
  }

  return getVerseConfidenceUi(
    props.result.verse?.similarity ?? 0,
    detectionPolicy.value.min_accepted_similarity,
    detectionPolicy.value.min_probable_similarity,
  )
})
const confidenceScoreLabel = computed(() => {
  return `${formatSimilarityPercent(props.result.verse?.similarity ?? 0)}%`
})

function formatSimilarityPercent(similarity: number) {
  const percent = similarity <= 1 ? similarity * 100 : similarity
  const safePercent = Math.max(0, Math.min(100, percent))

  return Math.round(safePercent)
}

const imamName = computed(() => {
  if (!topImam.value?.name) {
    switch (props.result.imam_status) {
      case 'disabled':
        return 'Détection imam désactivée'
      case 'unavailable':
        return 'Imam indisponible'
      case 'low':
        return 'Imam à confirmer'
      case 'unknown':
        return 'Imam non reconnu'
      default:
        return 'Imam indisponible'
    }
  }

  return topImam.value.name.replace(/_/g, ' ').trim()
})

const imamStatusText = computed(() => {
  if (topImam.value?.name) return ''

  switch (props.result.imam_status) {
    case 'disabled':
      return 'Identification de l’imam désactivée.'
    case 'unavailable':
      return 'Identification de l’imam temporairement indisponible.'
    case 'unknown':
      return 'Imam non reconnu pour cet extrait.'
    case 'low':
      return 'Identification de l’imam à confirmer.'
    default:
      return ''
  }
})
</script>

<template>
  <article class="result-card">
    <template v-if="result.verse">
      <section class="hero-panel">
        <p class="hero-kicker">Passage détecté</p>

        <div class="hero-main">
          <p class="surah-arabic">
            {{ result.verse.sourate_name }}
          </p>

          <p class="surah-transliteration">
            Sourate {{ result.verse.transliteration }}
          </p>

          <p class="verse-range">
            {{ verseLabel }}
          </p>
        </div>

        <div class="result-meta">
          <div class="confidence-detail" :class="confidenceUi.className">
            <span class="confidence-score">{{ confidenceScoreLabel }}</span>
            <span class="confidence-label">
              {{ confidenceUi.label }}
            </span>
          </div>

          <div class="imam-chip">
            {{ imamName }}
          </div>
        </div>

        <p v-if="imamStatusText" class="imam-status-text">
          {{ imamStatusText }}
        </p>

        <div class="action-row">
          <button
            class="primary-btn"
            type="button"
            @pointerenter="loadVerseDetailsSheet"
            @focus="loadVerseDetailsSheet"
            @touchstart.passive="loadVerseDetailsSheet"
            @click="isDetailsOpen = true"
          >
            <Eye class="action-icon" :stroke-width="2" aria-hidden="true" />
            Voir le verset
          </button>
        </div>
      </section>

      <VerseDetailsSheet
        v-if="isDetailsOpen"
        :open="isDetailsOpen"
        :result="result"
        @close="isDetailsOpen = false"
      />
    </template>

    <template v-else>
      <section class="hero-panel empty-panel">
        <p class="hero-kicker">Aucun résultat confirmé</p>
        <h2 class="empty-title">Essayez un nouvel extrait</h2>
        <p class="empty-text">
          Le passage n’a pas pu être validé avec assez de certitude sur cette tentative.
        </p>
      </section>
    </template>
  </article>
</template>

<style scoped>
.result-card {
  display: grid;
  gap: 18px;
}

.hero-panel {
  border-radius: 32px;
  border: 1px solid rgba(148, 163, 184, 0.14);
  background:
    linear-gradient(180deg, rgba(10, 20, 37, 0.76) 0%, rgba(7, 15, 30, 0.68) 100%);
  backdrop-filter: blur(14px);
  box-shadow: 0 24px 80px rgba(2, 6, 23, 0.22);
  padding: 28px;
}

.hero-kicker {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #93c5fd;
}

.hero-main {
  margin-top: 22px;
  text-align: center;
}

.surah-arabic {
  margin: 0;
  font-size: 56px;
  line-height: 1.1;
  font-family: 'Amiri', serif;
  direction: rtl;
  color: #f8fafc;
  text-shadow: 0 0 24px rgba(96, 165, 250, 0.12);
}

.surah-transliteration {
  margin: 12px 0 0;
  font-size: 28px;
  font-weight: 800;
  text-align: center;
  color: #eff6ff;
}

.verse-range {
  margin: 8px 0 0;
  color: #b8c6d8;
  font-size: 18px;
}

.action-icon {
  width: 19px;
  height: 19px;
}

.result-meta {
  margin-top: 22px;
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 10px;
}

.confidence-detail,
.imam-chip {
  display: inline-flex;
  align-items: center;
  min-height: 38px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background: rgba(255, 255, 255, 0.05);
  color: #e2e8f0;
}

.confidence-detail {
  gap: 8px;
  padding: 4px 14px 4px 6px;
}

.confidence-detail.banner-success {
  border-color: rgba(34, 197, 94, 0.16);
  background: rgba(34, 197, 94, 0.1);
}

.confidence-detail.banner-warning {
  border-color: rgba(245, 158, 11, 0.18);
  background: rgba(245, 158, 11, 0.1);
}

.confidence-detail.banner-error {
  border-color: rgba(239, 68, 68, 0.18);
  background: rgba(239, 68, 68, 0.1);
}

.confidence-score {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 48px;
  min-height: 30px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  font-size: 16px;
  line-height: 1;
  font-weight: 800;
  color: #eff6ff;
}

.confidence-label {
  font-size: 13px;
  font-weight: 800;
  color: #e2e8f0;
}

.imam-chip {
  padding: 0 16px;
  font-size: 13px;
  font-weight: 700;
  color: #dbeafe;
}

.imam-status-text {
  margin: 12px 0 0;
  text-align: center;
  color: #8ea1b9;
  font-size: 14px;
  line-height: 1.5;
}

.action-row {
  margin-top: 24px;
  display: flex;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
}

.primary-btn {
  border: none;
  border-radius: 999px;
  min-height: 52px;
  padding: 0 22px;
  font-weight: 700;
  cursor: pointer;
  transition:
    transform 0.2s ease,
    opacity 0.2s ease,
    background 0.2s ease;
}

.primary-btn:hover {
  transform: translateY(-1px);
}

.primary-btn {
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  color: #fff;
  box-shadow: 0 10px 30px rgba(37, 99, 235, 0.22);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.empty-panel {
  text-align: center;
}

.empty-title {
  margin: 12px 0 0;
  font-size: 30px;
  line-height: 1.15;
}

.empty-text {
  margin: 12px auto 0;
  max-width: 560px;
  color: #cbd5e1;
  line-height: 1.7;
}

@media (max-width: 768px) {
  .hero-panel {
    border-radius: 24px;
    padding: 18px 14px;
  }

  .hero-kicker {
    font-size: 11px;
    letter-spacing: 0.14em;
  }

  .hero-main {
    margin-top: 14px;
  }

  .surah-arabic {
    font-size: 38px;
  }

  .surah-transliteration {
    margin-top: 10px;
    font-size: 21px;
  }

  .verse-range {
    font-size: 15px;
  }

  .result-meta {
    margin-top: 16px;
    display: grid;
    gap: 8px;
  }

  .confidence-detail,
  .imam-chip {
    justify-content: center;
    min-height: 40px;
  }

  .action-row {
    margin-top: 18px;
  }

  .primary-btn {
    min-height: 48px;
    padding: 0 16px;
  }
}

@media (max-width: 360px) {
  .primary-btn {
    width: 100%;
  }
}
</style>
