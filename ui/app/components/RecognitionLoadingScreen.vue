<script setup lang="ts">
import { computed } from 'vue'
import RecognitionActionButton from '~/components/RecognitionActionButton.vue'
import type { LoadingStep } from '~/composables/useRecognition'

const props = defineProps<{
  loading: boolean
  step: LoadingStep
}>()

defineEmits<{
  cancel: []
}>()

const steps = computed(() => {
  return [
    {
      key: 'detecting',
      label: 'Écoute et transcription',
      active: props.step === 'detecting',
      done: props.step !== 'detecting',
    },
    {
      key: 'result-found',
      label: 'Correspondance détectée',
      active: props.step === 'result-found',
      done: false,
    },
    {
      key: 'retrying',
      label: 'Vérification en cours',
      active: props.step === 'retrying',
      done: false,
    },
  ]
})

const helperText = computed(() => {
  if (props.step === 'result-found') return 'Une correspondance a été trouvée. Finalisation en cours.'
  if (props.step === 'retrying') return 'Le système élargit la recherche pour confirmer le passage.'
  return 'Analyse du passage récité, cela prend quelques secondes.'
})
</script>

<template>
  <section class="screen loading-screen">
    <header class="loading-header">
      <p class="brand-mark">SAWT AI</p>

      <button class="cancel-btn" type="button" @click="$emit('cancel')">
        Annuler
      </button>
    </header>

    <div class="loading-shell">
      <RecognitionActionButton disabled loading />

      <div class="loading-copy">
        <p class="eyebrow">Analyse en cours</p>
        <h1 class="title">Patientez un instant</h1>
        <p class="subtitle">
          {{ helperText }}
        </p>
      </div>

      <div class="timeline">
        <div
          v-for="item in steps"
          :key="item.key"
          class="timeline-item"
          :class="{
            'is-active': item.active,
            'is-done': item.done,
          }"
        >
          <span class="timeline-dot" />
          <span class="timeline-label">{{ item.label }}</span>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.screen {
  position: relative;
  z-index: 1;
  min-height: calc(100vh - 76px);
  padding: 24px 20px 32px;
}

.loading-screen {
  display: flex;
  flex-direction: column;
}

.loading-header {
  width: min(1040px, 100%);
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.brand-mark {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.26em;
  text-transform: uppercase;
  color: #93c5fd;
}

.cancel-btn {
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(15, 23, 42, 0.45);
  color: #e2e8f0;
  border-radius: 999px;
  padding: 10px 16px;
  cursor: pointer;
  font-weight: 600;
  backdrop-filter: blur(12px);
  transition: transform 0.2s ease, background 0.2s ease;
}

.cancel-btn:hover {
  transform: translateY(-1px);
  background: rgba(15, 23, 42, 0.66);
}

.loading-shell {
  width: min(760px, 100%);
  margin: 0 auto;
  flex: 1;
  display: grid;
  justify-items: center;
  align-content: center;
  gap: 26px;
  text-align: center;
}

.loading-copy {
  max-width: 520px;
}

.eyebrow {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #60a5fa;
}

.title {
  margin: 14px 0 0;
  font-size: clamp(36px, 5vw, 54px);
  line-height: 1;
  letter-spacing: -0.04em;
}

.subtitle {
  margin: 14px 0 0;
  color: #cbd5e1;
  line-height: 1.7;
}

.timeline {
  width: min(540px, 100%);
  display: grid;
  gap: 12px;
}

.timeline-item {
  display: grid;
  grid-template-columns: 18px 1fr;
  gap: 14px;
  align-items: center;
  padding: 14px 16px;
  border-radius: 20px;
  background: rgba(2, 8, 23, 0.44);
  border: 1px solid rgba(148, 163, 184, 0.08);
  text-align: left;
  transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease;
}

.timeline-item.is-active {
  transform: translateY(-1px);
  background: rgba(12, 74, 110, 0.26);
  border-color: rgba(96, 165, 250, 0.24);
}

.timeline-item.is-done {
  border-color: rgba(74, 222, 128, 0.18);
}

.timeline-dot {
  width: 12px;
  height: 12px;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.42);
  box-shadow: 0 0 0 6px rgba(148, 163, 184, 0.06);
}

.timeline-item.is-active .timeline-dot {
  background: #60a5fa;
  box-shadow: 0 0 0 8px rgba(59, 130, 246, 0.14);
  animation: pulse 1.4s ease-in-out infinite;
}

.timeline-item.is-done .timeline-dot {
  background: #4ade80;
  box-shadow: 0 0 0 8px rgba(74, 222, 128, 0.08);
}

.timeline-label {
  color: #e2e8f0;
  font-weight: 600;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 0.9; }
  50% { transform: scale(1.14); opacity: 1; }
}

@media (max-width: 768px) {
  .screen {
    min-height: calc(100vh - 70px);
    padding: 18px 16px 24px;
  }

  .title {
    font-size: 40px;
  }

  .loading-shell {
    gap: 22px;
  }
}
</style>
