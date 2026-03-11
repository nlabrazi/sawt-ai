<script setup lang="ts">
// ROLE
// ----
// Page V1 très simple pour tester l'upload audio
// et afficher le résultat de Sawt AI.

import ResultCard from '~/components/ResultCard.vue'
import { useRecognition } from '~/composables/useRecognition'

const selectedFile = ref<File | null>(null)

const {
  loading,
  error,
  result,
  recognizeAudio,
} = useRecognition()

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  selectedFile.value = input.files?.[0] ?? null
}

async function submitAudio() {
  if (!selectedFile.value) return
  await recognizeAudio(selectedFile.value)
}
</script>

<template>
  <div class="page">
    <div class="container">
      <h1>Sawt AI</h1>
      <p class="subtitle">Reconnaissance de récitation coranique</p>

      <div class="card">
        <input type="file" accept=".wav,.mp3,.m4a,.ogg" @change="onFileChange" />

        <button :disabled="!selectedFile || loading" @click="submitAudio">
          {{ loading ? 'Analyse en cours...' : 'Lancer la reconnaissance' }}
        </button>

        <p v-if="error" class="error">{{ error }}</p>
      </div>

      <ResultCard v-if="result" :result="result" />
    </div>
  </div>
</template>

<style scoped>
.page {
  min-height: 100vh;
  background: #0b0b0b;
  color: white;
  padding: 2rem 1rem;
}

.container {
  max-width: 720px;
  margin: 0 auto;
}

.subtitle {
  color: #aaa;
  margin-bottom: 1.5rem;
}

.card {
  background: #111;
  border: 1px solid #2d2d2d;
  border-radius: 12px;
  padding: 1rem;
  margin-bottom: 1rem;
}

input {
  display: block;
  margin-bottom: 1rem;
}

button {
  padding: 0.8rem 1rem;
  border: none;
  border-radius: 10px;
  cursor: pointer;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.error {
  color: #ff6b6b;
  margin-top: 1rem;
}
</style>
