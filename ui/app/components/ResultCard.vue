<script setup lang="ts">
import { computed, ref } from 'vue'

import VerseDetailsSheet from '~/components/VerseDetailsSheet.vue'
import type { RecognizeResponse } from '~/composables/useRecognition'
import { getVerseConfidenceUi } from '~/utils/verseConfidence'

const props = defineProps<{
  result: RecognizeResponse
}>()

const isDetailsOpen = ref(false)
const copied = ref(false)

const verseLabel = computed(() => {
  if (!props.result.verse) return ''

  const { start_verse, end_verse } = props.result.verse
  return start_verse === end_verse
    ? `Verset ${start_verse}`
    : `Versets ${start_verse} à ${end_verse}`
})

const topImam = computed(() => props.result.imam_predictions?.[0] ?? null)
const confidenceUi = computed(() => {
  return getVerseConfidenceUi(props.result.verse?.similarity ?? 0)
})

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

async function copyVerse() {
  if (!props.result.verse) return

  const payload = [
    `${props.result.verse.sourate_name} — ${props.result.verse.transliteration}`,
    verseLabel.value,
    props.result.verse.text,
  ].join('\n')

  try {
    await navigator.clipboard.writeText(payload)
    copied.value = true
    window.setTimeout(() => {
      copied.value = false
    }, 1400)
  } catch (error) {
    console.error(error)
  }
}
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
            {{ result.verse.transliteration }}
          </p>

          <p class="verse-range">
            {{ verseLabel }}
          </p>
        </div>

        <div class="meta-row">
          <div class="meta-pill" :class="confidenceUi.className">
            {{ confidenceUi.label }}
          </div>

          <div class="meta-pill imam-pill">
            {{ imamName }}
          </div>
        </div>

        <p v-if="imamStatusText" class="imam-status-text">
          {{ imamStatusText }}
        </p>

        <div class="action-row">
          <button class="primary-btn" type="button" @click="isDetailsOpen = true">
            Voir le verset
          </button>

          <button class="secondary-btn" type="button" @click="copyVerse">
            {{ copied ? 'Copié' : 'Copier' }}
          </button>
        </div>
      </section>

      <VerseDetailsSheet
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
  margin-top: 24px;
  text-align: center;
}

.surah-arabic {
  margin: 0;
  font-size: 58px;
  line-height: 1.1;
  font-family: 'Amiri', serif;
  direction: rtl;
  color: #f8fafc;
  text-shadow: 0 0 24px rgba(96, 165, 250, 0.12);
}

.surah-transliteration {
  margin: 14px 0 0;
  font-size: 28px;
  font-weight: 800;
  color: #eff6ff;
}

.verse-range {
  margin: 10px 0 0;
  color: #b8c6d8;
  font-size: 18px;
}

.meta-row {
  margin-top: 24px;
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 10px;
}

.meta-pill {
  display: inline-flex;
  align-items: center;
  min-height: 38px;
  padding: 0 16px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background: rgba(255, 255, 255, 0.05);
  font-size: 13px;
  font-weight: 700;
  color: #e2e8f0;
}

.meta-pill.banner-success {
  background: rgba(34, 197, 94, 0.12);
  color: #dcfce7;
}

.meta-pill.banner-warning {
  background: rgba(245, 158, 11, 0.12);
  color: #fde68a;
}

.meta-pill.banner-error {
  background: rgba(239, 68, 68, 0.12);
  color: #fecaca;
}

.imam-pill {
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
  margin-top: 26px;
  display: flex;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
}

.primary-btn,
.secondary-btn {
  border: none;
  border-radius: 999px;
  min-height: 52px;
  padding: 0 22px;
  font-weight: 700;
  cursor: pointer;
  transition: transform 0.2s ease, opacity 0.2s ease, background 0.2s ease;
}

.primary-btn:hover,
.secondary-btn:hover {
  transform: translateY(-1px);
}

.primary-btn {
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  color: #fff;
  box-shadow: 0 10px 30px rgba(37, 99, 235, 0.22);
}

.secondary-btn {
  background: rgba(255, 255, 255, 0.06);
  color: #e2e8f0;
  border: 1px solid rgba(148, 163, 184, 0.16);
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
    border-radius: 28px;
    padding: 22px 18px;
  }

  .surah-arabic {
    font-size: 44px;
  }

  .surah-transliteration {
    font-size: 22px;
  }

  .verse-range {
    font-size: 17px;
  }

  .action-row {
    flex-direction: column;
  }

  .primary-btn,
  .secondary-btn {
    width: 100%;
  }
}
</style>
