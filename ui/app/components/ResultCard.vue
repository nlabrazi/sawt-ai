<script setup lang="ts">
import { useTajwid } from '~/composables/useTajwid'
import { parseTajwidToHtml } from '~/utils/parseTajwid'
import type { RecognizeResponse } from '~/composables/useRecognition'

const props = defineProps<{
  result: RecognizeResponse
}>()

const showTajwid = ref(false)
const tajwidText = ref<string | null>(null)
const tajwidHtml = computed(() => {
  if (!tajwidText.value) return ''
  return parseTajwidToHtml(tajwidText.value)
})

const { loading, fetchTajwid } = useTajwid()

const verseLabel = computed(() => {
  if (!props.result.verse) return ''

  const { start_verse, end_verse } = props.result.verse

  return start_verse === end_verse
    ? `Verset ${start_verse}`
    : `Versets ${start_verse} à ${end_verse}`
})

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
    end_verse
  )

  tajwidText.value = response.text
  showTajwid.value = true
}
</script>

<template>
  <div class="result-card">

    <p class="eyebrow">Résultat</p>

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

      <section class="verse-panel">

        <p class="panel-label">
          Texte coranique identifié
        </p>

        <p class="arabic-text">
          {{ result.verse.text }}
        </p>

        <button class="tajwid-btn" type="button" @click="toggleTajwid">
          <span v-if="loading">
            Chargement du tajwid...
          </span>

          <span v-else-if="showTajwid">
            Masquer le tajwid
          </span>

          <span v-else>
            Afficher le tajwid
          </span>
        </button>

        <div v-if="showTajwid && tajwidHtml" class="tajwid-container">
          <p class="tajwid-text" v-html="tajwidHtml" />
        </div>

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
  font-size: 13px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #93c5fd;
}

.hero-panel,
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
  font-size: 14px;
  color: #94a3b8;
}

.surah-arabic {
  margin-top: 10px;
  font-size: 34px;
  direction: rtl;
  font-family: 'Amiri', serif;
}

.surah-transliteration {
  margin-top: 6px;
  font-size: 16px;
  color: #93c5fd;
}

.verse-range {
  margin-top: 8px;
  color: #cbd5e1;
}

.panel-label {
  margin-bottom: 12px;
  font-size: 13px;
  text-transform: uppercase;
  color: #94a3b8;
}

.arabic-text {
  font-size: 30px;
  line-height: 2;
  direction: rtl;
  text-align: right;
  font-family: 'Amiri', serif;
}

.transcription-text {
  line-height: 1.8;
  direction: rtl;
}

.tajwid-btn {
  margin-top: 18px;
  border-radius: 12px;
  padding: 10px 14px;
  border: 1px solid rgba(148, 163, 184, 0.2);
  background: rgba(30, 41, 59, 0.6);
  cursor: pointer;
}

.tajwid-btn:hover {
  background: rgba(51, 65, 85, 0.7);
}

.tajwid-container {
  margin-top: 16px;
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

/* base prête pour couleurs futures */
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

.empty-text {
  color: #cbd5e1;
}
</style>
