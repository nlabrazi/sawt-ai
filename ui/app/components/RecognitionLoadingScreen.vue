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

const steps = computed(() => [
  {
    key: 'transcribing',
    title: 'Écoute en cours',
    text: 'Analyse audio',
  },
  {
    key: 'matching',
    title: 'Reconnaissance du passage',
    text: 'Recherche du passage',
  },
  {
    key: 'done',
    title: 'Préparation du résultat',
    text: 'Affichage imminent',
  },
])

const activeLabel = computed(() => {
  if (props.step === 'transcribing') return 'Transcription'
  if (props.step === 'matching') return 'Détection'
  return 'Finalisation'
})

function getState(stepKey: string) {
  const order = ['transcribing', 'matching', 'done']
  const currentIndex = order.indexOf(props.step)
  const stepIndex = order.indexOf(stepKey)

  if (stepIndex < currentIndex) return 'done'
  if (stepIndex === currentIndex) return 'active'
  return 'idle'
}
</script>

<template>
  <section class="screen loading-screen">
    <div class="top-bar">
      <button class="top-action" type="button" @click="$emit('cancel')">
        Annuler
      </button>
    </div>

    <div class="center-stack">
      <RecognitionActionButton disabled loading />

      <p class="eyebrow">Analyse en cours</p>
      <h2 class="main-title">Détection du passage</h2>

      <p class="main-subtitle">
        <span>{{ activeLabel }}</span>
        <span class="loading-dots" aria-hidden="true">
          <span>.</span><span>.</span><span>.</span>
        </span>
      </p>

      <div class="orbit-shell" aria-hidden="true">
        <span class="orbit orbit-1" />
        <span class="orbit orbit-2" />
        <span class="orbit orbit-3" />
        <span class="orbit-dot" />
      </div>

      <div class="loading-steps">
        <div v-for="item in steps" :key="item.key" class="loading-step" :class="`is-${getState(item.key)}`">
          <span class="step-indicator" :class="`is-${getState(item.key)}`" aria-hidden="true">
            <span v-if="getState(item.key) === 'active'" class="step-spinner" />
            <span v-else class="step-dot" />
          </span>
          <div class="step-copy">
            <p class="step-title">{{ item.title }}</p>
            <p class="step-text">{{ item.text }}</p>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.screen {
  position: relative;
  z-index: 1;
  flex: 1;
  min-height: 0;
  padding: 24px 16px 36px;
  box-sizing: border-box;
}

.loading-screen {
  display: flex;
  flex-direction: column;
}

.top-bar {
  display: flex;
  justify-content: flex-end;
}

.top-action {
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(15, 23, 42, 0.52);
  color: #e2e8f0;
  border-radius: 999px;
  min-height: 44px;
  padding: 0 16px;
  cursor: pointer;
  font-weight: 700;
  backdrop-filter: blur(10px);
  transition: transform 0.2s ease, background 0.2s ease, border-color 0.2s ease;
}

.top-action:hover {
  transform: translateY(-1px);
  background: rgba(30, 41, 59, 0.74);
  border-color: rgba(147, 197, 253, 0.26);
}

.center-stack {
  flex: 1;
  width: 100%;
  max-width: 760px;
  margin: 0 auto;
  display: grid;
  justify-items: center;
  align-content: center;
  text-align: center;
}

.eyebrow {
  margin: 20px 0 0;
  font-size: 13px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: #93c5fd;
}

.main-title {
  margin: 14px 0 0;
  font-size: clamp(34px, 5vw, 54px);
  line-height: 1.02;
  font-weight: 800;
  letter-spacing: -0.04em;
}

.main-subtitle {
  margin: 14px 0 0;
  max-width: 480px;
  min-height: 30px;
  font-size: 18px;
  line-height: 1.6;
  color: #cbd5e1;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.loading-dots {
  display: inline-flex;
  min-width: 24px;
}

.loading-dots span {
  animation: dotBlink 1.4s infinite;
  opacity: 0.25;
}

.loading-dots span:nth-child(2) {
  animation-delay: 0.2s;
}

.loading-dots span:nth-child(3) {
  animation-delay: 0.4s;
}

.orbit-shell {
  position: relative;
  width: 132px;
  height: 132px;
  margin-top: 30px;
  display: grid;
  place-items: center;
}

.orbit {
  position: absolute;
  border-radius: 999px;
  border: 1px solid rgba(147, 197, 253, 0.18);
}

.orbit-1 {
  inset: 0;
  animation: orbitPulse 2.1s ease-in-out infinite;
}

.orbit-2 {
  inset: 14px;
  border-color: rgba(96, 165, 250, 0.24);
  animation: orbitPulse 2.1s ease-in-out infinite 0.35s;
}

.orbit-3 {
  inset: 28px;
  border-color: rgba(59, 130, 246, 0.3);
  animation: orbitPulse 2.1s ease-in-out infinite 0.7s;
}

.orbit-dot {
  width: 14px;
  height: 14px;
  border-radius: 999px;
  background: #93c5fd;
  box-shadow: 0 0 24px rgba(96, 165, 250, 0.6);
  animation: dotPulse 1.4s ease-in-out infinite;
}

.loading-steps {
  width: 100%;
  max-width: 420px;
  margin-top: 26px;
  display: grid;
  gap: 12px;
}

.loading-step {
  position: relative;
  overflow: hidden;
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: center;
  gap: 14px;
  padding: 14px 16px;
  border-radius: 20px;
  border: 1px solid rgba(148, 163, 184, 0.1);
  background: rgba(8, 17, 32, 0.52);
  text-align: left;
  transition: transform 0.22s ease, border-color 0.22s ease, background 0.22s ease, opacity 0.22s ease;
}

.step-indicator {
  position: relative;
  width: 18px;
  height: 18px;
  display: grid;
  place-items: center;
  flex: 0 0 18px;
}

.step-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.34);
  transition: all 0.22s ease;
}

.step-spinner {
  width: 18px;
  height: 18px;
  border-radius: 999px;
  border: 2px solid rgba(147, 197, 253, 0.18);
  border-top-color: #93c5fd;
  border-right-color: rgba(147, 197, 253, 0.7);
  animation: stepSpin 0.85s linear infinite;
  box-shadow: 0 0 0 4px rgba(147, 197, 253, 0.06);
}

.step-title,
.step-text {
  margin: 0;
}

.step-title {
  font-size: 15px;
  font-weight: 700;
  color: #dbe6f3;
}

.step-text {
  margin-top: 4px;
  font-size: 13px;
  color: #8ea1b9;
}

.loading-step.is-idle .step-dot {
  background: rgba(148, 163, 184, 0.32);
  opacity: 0.7;
}

.loading-step.is-active .step-indicator {
  border-color: rgba(147, 197, 253, 0.22);
  background: rgba(15, 23, 42, 0.8);
  transform: scale(1.02);
  box-shadow: 0 14px 34px rgba(2, 6, 23, 0.14);
  filter: drop-shadow(0 0 10px rgba(147, 197, 253, 0.24));
}

.loading-step.is-active::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(110deg,
      transparent 0%,
      rgba(147, 197, 253, 0.06) 35%,
      rgba(147, 197, 253, 0.18) 50%,
      rgba(147, 197, 253, 0.06) 65%,
      transparent 100%);
  transform: translateX(-100%);
  animation: shimmer 1.8s linear infinite;
}

.loading-step.is-done .step-dot {
  background: #4ade80;
  box-shadow: 0 0 0 6px rgba(74, 222, 128, 0.08);
}

@keyframes shimmer {
  100% {
    transform: translateX(100%);
  }
}

@keyframes bulletPulse {

  0%,
  100% {
    transform: scale(0.92);
  }

  50% {
    transform: scale(1.18);
  }
}

@keyframes dotBlink {

  0%,
  80%,
  100% {
    opacity: 0.22;
    transform: translateY(0);
  }

  40% {
    opacity: 1;
    transform: translateY(-1px);
  }
}

@keyframes orbitPulse {

  0%,
  100% {
    transform: scale(0.96);
    opacity: 0.58;
  }

  50% {
    transform: scale(1.04);
    opacity: 1;
  }
}

@keyframes dotPulse {

  0%,
  100% {
    transform: scale(0.92);
    opacity: 0.82;
  }

  50% {
    transform: scale(1.15);
    opacity: 1;
  }
}

@keyframes stepSpin {
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 768px) {
  .screen {
    padding: 18px 14px 28px;
  }

  .main-title {
    font-size: 40px;
  }

  .main-subtitle {
    font-size: 17px;
    max-width: 320px;
  }

  .loading-steps {
    max-width: 100%;
  }
}
</style>
