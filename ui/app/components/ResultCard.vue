<script setup lang="ts">
// ROLE
// ----
// Affiche le résultat principal de reconnaissance.

import type { RecognizeResponse } from '~/composables/useRecognition'

defineProps<{
  result: RecognizeResponse
}>()
</script>

<template>
  <div class="result-card">
    <h2>Résultat</h2>

    <div class="section">
      <h3>Transcription</h3>
      <p>{{ result.transcription_text || 'Aucune transcription' }}</p>
    </div>

    <div v-if="result.verse" class="section">
      <h3>Verset détecté</h3>
      <p><strong>Sourate :</strong> {{ result.verse.sourate_name }}</p>
      <p><strong>Versets :</strong> {{ result.verse.start_verse }} à {{ result.verse.end_verse }}</p>
      <p><strong>Texte :</strong> {{ result.verse.text }}</p>
      <p><strong>Similarité :</strong> {{ result.verse.similarity }}</p>
    </div>

    <div class="section">
      <h3>Imam</h3>
      <p class="soon">Détection imam bientôt disponible.</p>
    </div>
  </div>
</template>

<style scoped>
.result-card {
  padding: 1rem;
  border: 1px solid #2d2d2d;
  border-radius: 12px;
  background: #111;
}

.section + .section {
  margin-top: 1rem;
}

h2,
h3 {
  margin-bottom: 0.5rem;
}

p {
  color: #ddd;
}

.soon {
  color: #f0b45a;
}
</style>
