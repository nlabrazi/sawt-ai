<script setup lang="ts">
import FeedbackForm from '~/components/FeedbackForm.vue'
import ResultCard from '~/components/ResultCard.vue'
import type { RecognizeResponse } from '~/composables/useRecognition'

defineProps<{
  error: string | null
  result: RecognizeResponse | null
}>()

defineEmits<{
  reset: []
}>()
</script>

<template>
  <section class="screen result-screen">
    <div class="result-layout">
      <div class="result-top-bar">
        <div>
          <p class="brand-kicker">Sawt AI</p>
          <h1 class="result-title">Résultat de la reconnaissance</h1>
        </div>

        <button class="top-action" type="button" @click="$emit('reset')">
          Nouvelle détection
        </button>
      </div>

      <div v-if="error" class="error-banner">
        {{ error }}
      </div>

      <ResultCard v-if="result" :result="result" />

      <FeedbackForm v-if="result" :result="result" />
    </div>
  </section>
</template>

<style scoped>
.screen {
  position: relative;
  z-index: 1;
  min-height: 100vh;
  padding: 24px 16px 36px;
  box-sizing: border-box;
}

.result-screen {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding-top: 28px;
  padding-bottom: 40px;
  overflow-y: auto;
}

.result-layout {
  width: 100%;
  max-width: 920px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.result-top-bar {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.brand-kicker {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #93c5fd;
}

.result-title {
  margin: 8px 0 0;
  font-size: 28px;
  line-height: 1.15;
  font-weight: 800;
  letter-spacing: -0.03em;
}

.top-action {
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(15, 23, 42, 0.6);
  color: #e2e8f0;
  border-radius: 999px;
  padding: 10px 14px;
  cursor: pointer;
  font-weight: 600;
  backdrop-filter: blur(10px);
  transition: transform 0.2s ease, background 0.2s ease, border-color 0.2s ease;
}

.top-action:hover {
  transform: translateY(-1px);
  background: rgba(30, 41, 59, 0.78);
  border-color: rgba(147, 197, 253, 0.28);
}

.error-banner {
  padding: 14px 16px;
  border-radius: 18px;
  color: #fecaca;
  background: rgba(127, 29, 29, 0.35);
  border: 1px solid rgba(239, 68, 68, 0.18);
}

@media (max-width: 768px) {
  .screen {
    padding: 18px 14px 28px;
  }

  .result-top-bar {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
