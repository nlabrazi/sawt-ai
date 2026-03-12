<script setup lang="ts">
// ROLE
// ----
// Loader visuel type Shazam-like avec étapes simples et crédibles.

import type { LoadingStep } from '~/composables/useRecognition'

defineProps<{
  loading: boolean
  step: LoadingStep
}>()
</script>

<template>
  <div v-if="loading" class="loader-card">
    <div class="steps">
      <div class="step" :class="{ active: step === 'detecting', done: step !== 'detecting' }">
        <span class="step-icon">
          <span v-if="step === 'detecting'" class="spinner" />
          <span v-else class="check-icon">✓</span>
        </span>
        <span class="step-label">Détection en cours...</span>
      </div>

      <div class="step" :class="{
        active: step === 'result-found',
        done: step === 'retrying'
      }">
        <span class="step-icon">
          <span v-if="step === 'result-found'" class="spinner" />
          <span v-else-if="step === 'retrying'" class="check-icon">✓</span>
          <span v-else class="pending-dot" />
        </span>
        <span class="step-label">Résultat trouvé</span>
      </div>

      <div class="step" :class="{ active: step === 'retrying' }">
        <span class="step-icon">
          <span v-if="step === 'retrying'" class="spinner" />
          <span v-else class="pending-dot" />
        </span>
        <span class="step-label">Nouvelle tentative...</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.loader-card {
  width: 100%;
  border: 1px solid rgba(148, 163, 184, 0.14);
  border-radius: 28px;
  padding: 22px;
  background: rgba(15, 23, 42, 0.46);
  backdrop-filter: blur(14px);
  box-sizing: border-box;
}

.steps {
  display: grid;
  gap: 12px;
}

.step {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid rgba(148, 163, 184, 0.1);
  background: rgba(2, 6, 23, 0.36);
  opacity: 0.72;
  transition: border-color 0.2s ease, opacity 0.2s ease, transform 0.2s ease;
}

.step.active {
  opacity: 1;
  border-color: rgba(96, 165, 250, 0.28);
  transform: translateY(-1px);
}

.step.done {
  opacity: 1;
  border-color: rgba(34, 197, 94, 0.22);
}

.step-icon {
  width: 20px;
  height: 20px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.step-label {
  color: #e2e8f0;
  font-size: 15px;
}

.pending-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: #64748b;
  opacity: 0.9;
}

.check-icon {
  color: #22c55e;
  font-size: 16px;
  font-weight: 800;
}

.spinner {
  width: 18px;
  height: 18px;
  border-radius: 999px;
  border: 2px solid rgba(255, 255, 255, 0.16);
  border-top-color: #60a5fa;
  animation: spin 0.9s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
