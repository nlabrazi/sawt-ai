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

const steps: Array<{ key: LoadingStep; title: string }> = [
  { key: 'transcribing', title: 'Écoute' },
  { key: 'matching', title: 'Recherche' },
  { key: 'done', title: 'Résultat' },
]

const activeStepIndex = computed(() => {
  return Math.max(
    0,
    steps.findIndex((item) => item.key === props.step),
  )
})

const activeLabel = computed(() => {
  if (props.step === 'transcribing') return 'Nous écoutons votre récitation.'
  if (props.step === 'matching') return 'Nous comparons le passage aux versets du Coran.'
  return 'Votre résultat est presque prêt.'
})

function getState(stepKey: LoadingStep) {
  const stepIndex = steps.findIndex((item) => item.key === stepKey)

  if (stepIndex < activeStepIndex.value) return 'done'
  if (stepIndex === activeStepIndex.value) return 'active'
  return 'idle'
}
</script>

<template>
  <section
    class="screen loading-screen"
    aria-labelledby="loading-title"
    :aria-busy="loading"
  >
    <header class="top-bar">
      <div class="brand" aria-label="Sawt AI">
        <span class="brand-name">Sawt</span>
        <span class="brand-mark">AI</span>
      </div>

      <button class="cancel-action" type="button" @click="$emit('cancel')">
        Annuler
      </button>
    </header>

    <div class="center-stack">
      <p class="eyebrow">Étape {{ activeStepIndex + 1 }} sur {{ steps.length }}</p>

      <h1 id="loading-title" class="main-title">Recherche du passage</h1>

      <p class="main-subtitle" role="status" aria-live="polite">
        {{ activeLabel }}
      </p>

      <RecognitionActionButton class="loading-action" disabled loading />

      <ol class="loading-steps" aria-label="Progression de l’analyse">
        <li
          v-for="item in steps"
          :key="item.key"
          class="loading-step"
          :class="`is-${getState(item.key)}`"
          :aria-current="getState(item.key) === 'active' ? 'step' : undefined"
        >
          <span class="step-indicator" aria-hidden="true">
            <span class="step-dot" />
          </span>
          <span class="step-title">{{ item.title }}</span>
        </li>
      </ol>
    </div>
  </section>
</template>

<style scoped>
.screen {
  position: relative;
  z-index: 1;
  flex: 1;
  min-height: 0;
  box-sizing: border-box;
}

.loading-screen {
  width: min(100%, 860px);
  margin: 0 auto;
  padding: 26px 20px 36px;
  display: flex;
  flex-direction: column;
}

.top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.brand {
  display: inline-flex;
  align-items: baseline;
  gap: 5px;
  color: #f8fafc;
  font-size: 17px;
  line-height: 1;
  font-weight: 800;
  letter-spacing: -0.02em;
}

.brand-mark {
  color: #60a5fa;
  font-size: 12px;
  letter-spacing: 0.04em;
}

.cancel-action {
  min-height: 42px;
  padding: 0 14px;
  border: 1px solid transparent;
  border-radius: 999px;
  background: transparent;
  color: #9fb0c4;
  font: inherit;
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  transition:
    color 180ms ease,
    border-color 180ms ease,
    background 180ms ease;
}

.cancel-action:hover {
  border-color: rgba(148, 163, 184, 0.16);
  background: rgba(15, 23, 42, 0.4);
  color: #fff;
}

.cancel-action:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 3px;
}

.center-stack {
  flex: 1;
  width: 100%;
  display: grid;
  justify-items: center;
  align-content: center;
  text-align: center;
}

.eyebrow {
  margin: 0;
  color: #93c5fd;
  font-size: 12px;
  font-weight: 750;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.main-title {
  margin: 14px 0 0;
  font-size: clamp(38px, 6vw, 58px);
  line-height: 1.02;
  font-weight: 800;
  letter-spacing: -0.052em;
  text-wrap: balance;
}

.main-subtitle {
  min-height: 52px;
  margin: 15px 0 0;
  max-width: 500px;
  color: #aebdd0;
  font-size: 17px;
  line-height: 1.55;
  text-wrap: balance;
}

.loading-action {
  margin-top: 14px;
}

.loading-steps {
  width: min(100%, 420px);
  margin: 8px 0 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  list-style: none;
}

.loading-step {
  position: relative;
  display: grid;
  justify-items: center;
  gap: 8px;
  color: #66788f;
  font-size: 12px;
  font-weight: 700;
}

.loading-step:not(:last-child)::after {
  content: '';
  position: absolute;
  top: 5px;
  left: calc(50% + 8px);
  width: calc(100% - 16px);
  height: 1px;
  background: rgba(148, 163, 184, 0.16);
}

.loading-step.is-done:not(:last-child)::after {
  background: rgba(96, 165, 250, 0.7);
}

.step-indicator {
  position: relative;
  z-index: 1;
  width: 11px;
  height: 11px;
  display: grid;
  place-items: center;
  border-radius: 999px;
  background: #101c2d;
}

.step-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: #53657b;
}

.loading-step.is-active {
  color: #dbeafe;
}

.loading-step.is-active .step-dot {
  background: #93c5fd;
  box-shadow: 0 0 0 5px rgba(96, 165, 250, 0.12);
  animation: activeStepPulse 1.5s ease-in-out infinite;
}

.loading-step.is-done {
  color: #91a2b8;
}

.loading-step.is-done .step-dot {
  background: #60a5fa;
}

@keyframes activeStepPulse {
  0%,
  100% {
    transform: scale(0.9);
  }

  50% {
    transform: scale(1.18);
  }
}

@media (max-width: 640px) {
  .loading-screen {
    padding: 20px 16px 28px;
  }

  .main-title {
    font-size: clamp(34px, 10vw, 46px);
  }

  .main-subtitle {
    min-height: 48px;
    max-width: 340px;
    font-size: 16px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .loading-step.is-active .step-dot {
    animation: none;
  }
}
</style>
