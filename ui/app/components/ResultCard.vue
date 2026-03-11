<script setup lang="ts">
// ROLE
// ----
// Carte de résultat principale.
// Affiche la transcription, le verset détecté
// et l'état temporaire de la détection imam.

import type { RecognizeResponse } from '~/composables/useRecognition'

defineProps<{
  result: RecognizeResponse
}>()
</script>

<template>
  <div class="result-card">
    <div class="header">
      <p class="eyebrow">Résultat</p>
      <h2>Analyse terminée</h2>
    </div>

    <div class="grid">
      <section class="panel">
        <p class="panel-label">Transcription</p>
        <p class="arabic-text">
          {{ result.transcription_text || 'Aucune transcription disponible.' }}
        </p>
      </section>

      <section class="panel" v-if="result.verse">
        <p class="panel-label">Verset détecté</p>

        <div class="meta-row">
          <span class="meta-key">Sourate</span>
          <span class="meta-value">{{ result.verse.sourate_name }}</span>
        </div>

        <div class="meta-row">
          <span class="meta-key">Versets</span>
          <span class="meta-value">
            {{ result.verse.start_verse }} à {{ result.verse.end_verse }}
          </span>
        </div>

        <div class="meta-row">
          <span class="meta-key">Confiance</span>
          <span class="meta-value">
            {{ (result.verse.similarity * 100).toFixed(1) }}%
          </span>
        </div>

        <div class="verse-box">
          {{ result.verse.text }}
        </div>
      </section>

      <section class="panel">
        <p class="panel-label">Imam</p>
        <p class="soon-badge">Bientôt disponible</p>
        <p class="muted-text">
          La reconnaissance de l’imam arrivera dans une prochaine étape.
        </p>
      </section>
    </div>
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
}

.header {
  margin-bottom: 20px;
}

.eyebrow {
  margin: 0 0 6px;
  font-size: 13px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #93c5fd;
}

h2 {
  margin: 0;
  font-size: 28px;
  line-height: 1.2;
}

.grid {
  display: grid;
  gap: 16px;
}

.panel {
  padding: 18px;
  border-radius: 20px;
  background: rgba(2, 6, 23, 0.45);
  border: 1px solid rgba(148, 163, 184, 0.12);
}

.panel-label {
  margin: 0 0 14px;
  font-size: 14px;
  font-weight: 700;
  color: #cbd5e1;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.arabic-text,
.verse-box {
  line-height: 2;
  font-size: 18px;
  direction: rtl;
  text-align: right;
  color: #f8fafc;
}

.meta-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 0;
  border-bottom: 1px solid rgba(148, 163, 184, 0.1);
}

.meta-row:last-of-type {
  margin-bottom: 14px;
}

.meta-key {
  color: #94a3b8;
  font-size: 14px;
}

.meta-value {
  color: #fff;
  font-weight: 600;
  text-align: right;
}

.verse-box {
  padding: 14px;
  border-radius: 16px;
  background: rgba(30, 41, 59, 0.55);
}

.soon-badge {
  display: inline-flex;
  margin: 0;
  padding: 8px 12px;
  border-radius: 999px;
  color: #fef3c7;
  background: rgba(245, 158, 11, 0.16);
  border: 1px solid rgba(245, 158, 11, 0.22);
  font-weight: 600;
}

.muted-text {
  margin: 14px 0 0;
  color: #cbd5e1;
  line-height: 1.6;
}

@media (max-width: 640px) {
  .result-card {
    padding: 18px;
    border-radius: 22px;
  }

  h2 {
    font-size: 24px;
  }

  .arabic-text,
  .verse-box {
    font-size: 16px;
  }

  .meta-row {
    flex-direction: column;
    align-items: flex-start;
  }

  .meta-value {
    text-align: left;
  }
}
</style>
