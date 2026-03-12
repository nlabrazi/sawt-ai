<script setup lang="ts">
import type { RecognizeResponse } from '~/composables/useRecognition'

const props = defineProps<{
  result: RecognizeResponse
}>()

const verseLabel = computed(() => {
  if (!props.result.verse) return ''

  const { start_verse, end_verse } = props.result.verse
  return start_verse === end_verse
    ? `Verset ${start_verse}`
    : `Versets ${start_verse} à ${end_verse}`
})
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
        <p class="panel-label">Texte coranique identifié</p>

        <p class="arabic-text">
          {{ result.verse.text }}
        </p>
      </section>

      <section v-if="result.transcription_text" class="transcription-panel">
        <p class="panel-label">Transcription brute</p>

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
  box-shadow: 0 24px 80px rgba(2, 6, 23, 0.35);
  display: grid;
  gap: 16px;
}

.eyebrow {
  margin: 0;
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
  margin: 0;
  font-size: 14px;
  color: #94a3b8;
}

.surah-arabic {
  margin: 10px 0 0;
  font-size: 34px;
  line-height: 1.2;
  font-weight: 700;
  color: #fff;
  direction: rtl;
  font-family: 'Amiri', serif;
}

.surah-transliteration {
  margin: 8px 0 0;
  font-size: 16px;
  color: #93c5fd;
  font-weight: 600;
}

.verse-range {
  margin: 12px 0 0;
  font-size: 15px;
  color: #cbd5e1;
}

.panel-label {
  margin: 0 0 14px;
  font-size: 13px;
  font-weight: 700;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.arabic-text {
  margin: 0;
  line-height: 2.1;
  font-size: 30px;
  direction: rtl;
  text-align: right;
  color: #f8fafc;
  font-family: 'Amiri', serif;
}

.transcription-text {
  margin: 0;
  line-height: 1.9;
  font-size: 16px;
  color: #cbd5e1;
  direction: rtl;
  text-align: right;
}

.empty-text {
  margin: 0;
  color: #cbd5e1;
  line-height: 1.7;
}

@media (max-width: 640px) {
  .result-card {
    padding: 18px;
    border-radius: 22px;
  }

  .surah-arabic {
    font-size: 28px;
  }

  .arabic-text {
    font-size: 24px;
  }

  .transcription-text {
    font-size: 15px;
  }
}
</style>
