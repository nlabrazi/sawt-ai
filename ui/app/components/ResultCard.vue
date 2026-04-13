<script setup lang="ts">
// ROLE
// ----
// Affiche le résultat principal,
// avec option d'affichage/masquage du tajwid à la demande.
// Gère aussi un fallback propre si l'imam n'a pas pu être reconnu.

import { computed, ref } from 'vue'

import { useTajwid } from '~/composables/useTajwid'
import type { RecognizeResponse } from '~/composables/useRecognition'
import { parseTajwidToHtml } from '~/utils/parseTajwid'

const props = defineProps<{
  result: RecognizeResponse
}>()

const showTajwid = ref(false)
const tajwidText = ref<string | null>(null)

const { loading, fetchTajwid, error } = useTajwid()

const verseLabel = computed(() => {
  if (!props.result.verse) return ''

  const { start_verse, end_verse } = props.result.verse

  return start_verse === end_verse
    ? `Verset ${start_verse}`
    : `Versets ${start_verse} à ${end_verse}`
})

const tajwidHtml = computed(() => {
  if (!tajwidText.value) return ''
  return parseTajwidToHtml(tajwidText.value)
})

const topImam = computed(() => props.result.imam_predictions?.[0] ?? null)

const otherImams = computed(() => {
  return props.result.imam_predictions?.slice(1) ?? []
})

const hasImamResult = computed(() => {
  return !!props.result.imam_predictions?.length
})

const imamStatusUi = computed(() => {
  switch (props.result.imam_status) {
    case 'disabled':
      return {
        label: 'Désactivée',
        className: 'imam-status-unknown',
      }

    case 'high':
      return {
        label: 'Fiable',
        className: 'imam-status-high',
      }

    case 'medium':
      return {
        label: 'À vérifier',
        className: 'imam-status-medium',
      }

    case 'unavailable':
      return {
        label: 'Indisponible',
        className: 'imam-status-unknown',
      }

    case 'low':
      return {
        label: 'Incertain',
        className: 'imam-status-low',
      }

    default:
      return {
        label: 'Indisponible',
        className: 'imam-status-unknown',
      }
  }
})

const imamFallbackText = computed(() => {
  switch (props.result.imam_status) {
    case 'disabled':
      return 'Détection imam désactivée pour cet audio.'

    case 'low':
      return 'Imam détecté avec une confiance trop faible.'

    case 'unknown':
      return 'Imam non reconnu pour cet extrait.'

    case 'unavailable':
      return 'Identification de l’imam temporairement indisponible.'

    default:
      return 'Identification de l’imam indisponible.'
  }
})

const imamFallbackDescription = computed(() => {
  if (props.result.imam_status === 'disabled') {
    return 'Le verset reste analysé normalement, mais l’identification du récitateur a été volontairement désactivée.'
  }

  if (props.result.imam_status === 'unavailable') {
    return 'Le verset reste analysé normalement, mais le service serveur de reconnaissance du récitateur n’a pas pu être chargé.'
  }

  if (props.result.verse) {
    return 'Le verset a bien été détecté, mais le récitateur n’a pas pu être confirmé avec assez de fiabilité.'
  }

  return 'L’audio a bien été analysé, mais aucun récitateur fiable n’a pu être identifié.'
})

function formatImamName(name: string) {
  return name.replace(/_/g, ' ').trim()
}

async function toggleTajwid() {
  if (!props.result.verse) return

  if (showTajwid.value) {
    showTajwid.value = false
    return
  }

  if (tajwidText.value) {
    showTajwid.value = true
    return
  }

  const { sourate_id, start_verse, end_verse } = props.result.verse

  const response = await fetchTajwid(
    sourate_id,
    start_verse,
    end_verse,
  )

  tajwidText.value = response.text
  showTajwid.value = true
}
</script>

<template>
  <div class="result-card">
    <p class="eyebrow">
      Résultat
    </p>

    <template v-if="result.verse">
      <section class="hero-panel">
        <p class="surah-index">
          Sourate {{ result.verse.sourate_id }}
        </p>

        <p class="surah-arabic">
          {{ result.verse.sourate_name }}
        </p>

        <p class="surah-transliteration">
          {{ result.verse.transliteration }}
        </p>

        <p class="verse-range">
          {{ verseLabel }}
        </p>
      </section>

      <section class="imam-panel">
        <div class="imam-panel-header">
          <p class="panel-label imam-panel-label">
            Imam récitant
          </p>

          <span class="imam-status-badge" :class="imamStatusUi.className">
            {{ imamStatusUi.label }}
          </span>
        </div>

        <template v-if="hasImamResult && topImam">
          <p class="imam-kicker">
            Récité par
          </p>

          <p class="imam-name">
            {{ formatImamName(topImam.name) }}
          </p>

          <div v-if="otherImams.length" class="imam-suggestions">
            <p class="imam-suggestions-label">
              Autres suggestions
            </p>

            <div class="imam-chip-list">
              <span v-for="imam in otherImams" :key="imam.name" class="imam-chip">
                {{ formatImamName(imam.name) }}
              </span>
            </div>
          </div>
        </template>

        <template v-else>
          <p class="imam-kicker">
            Statut
          </p>

          <p class="imam-name imam-name-fallback">
            {{ imamFallbackText }}
          </p>

          <p class="imam-fallback-description">
            {{ imamFallbackDescription }}
          </p>
        </template>
      </section>

      <section class="verse-panel">
        <p class="panel-label">
          Verset coranique identifié
        </p>

        <p class="arabic-text">
          {{ result.verse.text }}
        </p>

        <button class="tajwid-toggle" type="button" :disabled="loading" @click="toggleTajwid">
          <span v-if="loading">⏳ Chargement du tajwid...</span>
          <span v-else-if="showTajwid">➖ Masquer le tajwid</span>
          <span v-else>➕ Afficher le tajwid</span>
        </button>

        <div v-if="showTajwid && tajwidHtml" class="tajwid-container">
          <p class="tajwid-title">
            Affichage tajwid
          </p>

          <p class="tajwid-text" v-html="tajwidHtml" />
        </div>

        <p v-if="error" class="tajwid-error">
          {{ error }}
        </p>
      </section>

      <section v-if="result.transcription_text" class="transcription-panel">
        <p class="panel-label">
          Transcription brute
        </p>

        <p class="transcription-text">
          {{ result.transcription_text }}
        </p>
      </section>
    </template>

    <template v-else>
      <section class="empty-panel">
        <p class="empty-text">
          Aucun verset fiable n’a été trouvé pour cet audio.
        </p>

        <section class="imam-panel imam-panel-alone">
          <div class="imam-panel-header">
            <p class="panel-label imam-panel-label">
              Imam récitant
            </p>

            <span class="imam-status-badge" :class="imamStatusUi.className">
              {{ imamStatusUi.label }}
            </span>
          </div>

          <template v-if="hasImamResult && topImam">
            <p class="imam-kicker">
              Voix la plus probable
            </p>

            <p class="imam-name">
              {{ formatImamName(topImam.name) }}
            </p>

            <div v-if="otherImams.length" class="imam-suggestions">
              <p class="imam-suggestions-label">
                Autres suggestions
              </p>

              <div class="imam-chip-list">
                <span v-for="imam in otherImams" :key="imam.name" class="imam-chip">
                  {{ formatImamName(imam.name) }}
                </span>
              </div>
            </div>
          </template>

          <template v-else>
            <p class="imam-kicker">
              Statut
            </p>

            <p class="imam-name imam-name-fallback">
              {{ imamFallbackText }}
            </p>

            <p class="imam-fallback-description">
              {{ imamFallbackDescription }}
            </p>
          </template>
        </section>
      </section>
    </template>
  </div>
</template>

<style scoped>
.result-card {
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 28px;
  padding: 24px;
  background: rgba(15, 23, 42, 0.6);
  backdrop-filter: blur(12px);
  display: grid;
  gap: 18px;
}

.eyebrow {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #93c5fd;
}

.hero-panel,
.imam-panel,
.verse-panel,
.transcription-panel,
.empty-panel {
  padding: 20px;
  border-radius: 22px;
  background: rgba(2, 6, 23, 0.45);
  border: 1px solid rgba(148, 163, 184, 0.12);
}

.hero-panel {
  text-align: center;
}

.surah-index {
  margin: 0;
  font-size: 14px;
  color: #94a3b8;
}

.surah-arabic {
  margin: 10px 0 0;
  font-size: 34px;
  direction: rtl;
  font-family: 'Amiri', serif;
  color: #fff;
}

.surah-transliteration {
  margin: 6px 0 0;
  font-size: 16px;
  color: #93c5fd;
  font-weight: 600;
}

.verse-range {
  margin: 8px 0 0;
  color: #cbd5e1;
}

.panel-label {
  margin: 0 0 14px;
  font-size: 13px;
  text-transform: uppercase;
  color: #94a3b8;
  letter-spacing: 0.08em;
}

.imam-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.imam-panel-label {
  margin-bottom: 0;
}

.imam-status-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 32px;
  padding: 0 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
  border: 1px solid transparent;
  white-space: nowrap;
}

.imam-status-high {
  color: #bbf7d0;
  background: rgba(34, 197, 94, 0.14);
  border-color: rgba(34, 197, 94, 0.22);
}

.imam-status-medium {
  color: #fde68a;
  background: rgba(245, 158, 11, 0.14);
  border-color: rgba(245, 158, 11, 0.22);
}

.imam-status-low {
  color: #fecaca;
  background: rgba(239, 68, 68, 0.14);
  border-color: rgba(239, 68, 68, 0.22);
}

.imam-status-unknown {
  color: #cbd5e1;
  background: rgba(148, 163, 184, 0.12);
  border-color: rgba(148, 163, 184, 0.18);
}

.imam-kicker {
  margin: 18px 0 0;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #94a3b8;
}

.imam-name {
  margin: 8px 0 0;
  font-size: 24px;
  font-weight: 700;
  line-height: 1.3;
  color: #f8fafc;
}

.imam-name-fallback {
  font-size: 20px;
}

.imam-fallback-description {
  margin: 12px 0 0;
  line-height: 1.7;
  color: #cbd5e1;
}

.imam-suggestions {
  margin-top: 18px;
  padding-top: 18px;
  border-top: 1px solid rgba(148, 163, 184, 0.12);
}

.imam-suggestions-label {
  margin: 0 0 12px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #94a3b8;
}

.imam-chip-list {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.imam-chip {
  display: inline-flex;
  align-items: center;
  min-height: 36px;
  padding: 0 14px;
  border-radius: 999px;
  background: rgba(30, 41, 59, 0.68);
  border: 1px solid rgba(148, 163, 184, 0.14);
  color: #e2e8f0;
  font-size: 14px;
  font-weight: 600;
}

.imam-panel-alone {
  margin-top: 18px;
}

.arabic-text {
  margin: 0;
  font-size: 30px;
  line-height: 2;
  direction: rtl;
  text-align: right;
  font-family: 'Amiri', serif;
  color: #f8fafc;
}

.tajwid-toggle {
  margin-top: 20px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border-radius: 999px;
  padding: 11px 18px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(30, 41, 59, 0.68);
  color: #e5e7eb;
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  transition:
    transform 0.22s ease,
    background 0.22s ease,
    border-color 0.22s ease,
    box-shadow 0.22s ease;
}

.tajwid-toggle:hover {
  transform: translateY(-2px);
  background: rgba(51, 65, 85, 0.86);
  border-color: rgba(147, 197, 253, 0.24);
  box-shadow: 0 0 24px rgba(59, 130, 246, 0.12);
}

.tajwid-toggle:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.tajwid-container {
  margin-top: 18px;
  padding-top: 18px;
  border-top: 1px solid rgba(148, 163, 184, 0.12);
}

.tajwid-title {
  margin: 0 0 12px;
  font-size: 13px;
  text-transform: uppercase;
  color: #94a3b8;
  letter-spacing: 0.08em;
}

.tajwid-text {
  margin: 0;
  font-size: 30px;
  line-height: 2.1;
  direction: rtl;
  text-align: right;
  font-family: 'Amiri', serif;
  color: #f8fafc;
}

:deep(.tajwid-fragment) {
  color: #f8fafc;
}

:deep(.tajwid-rule-ghn),
:deep(.tajwid-rule-g),
:deep(.tajwid-rule-idgham),
:deep(.tajwid-rule-i) {
  color: #f59e0b;
}

:deep(.tajwid-rule-ikhf),
:deep(.tajwid-rule-k) {
  color: #22c55e;
}

:deep(.tajwid-rule-qlq),
:deep(.tajwid-rule-q) {
  color: #38bdf8;
}

:deep(.tajwid-rule-m),
:deep(.tajwid-rule-p) {
  color: #f472b6;
}

:deep(.tajwid-rule-n) {
  color: #a78bfa;
}

.tajwid-error {
  margin-top: 14px;
  color: #fecaca;
}

.transcription-text {
  margin: 0;
  line-height: 1.8;
  direction: rtl;
  text-align: right;
  color: #cbd5e1;
}

.empty-text {
  color: #cbd5e1;
}

@media (max-width: 640px) {
  .result-card {
    padding: 18px;
  }

  .surah-arabic {
    font-size: 28px;
  }

  .imam-panel-header {
    flex-direction: column;
    align-items: flex-start;
  }

  .imam-name {
    font-size: 22px;
  }

  .imam-name-fallback {
    font-size: 18px;
  }

  .arabic-text,
  .tajwid-text {
    font-size: 24px;
    line-height: 1.9;
  }
}
</style>
