<script setup lang="ts">
import { Check } from '@lucide/vue'
import { Motion } from 'motion-v'
import { computed } from 'vue'

import RecognitionActionButton from '~/components/RecognitionActionButton.vue'
import MotionReveal from '~/components/MotionReveal.vue'
import type { LoadingStep } from '~/composables/useRecognition'
import { useUiMotion } from '~/composables/useUiMotion'

const props = defineProps<{
  loading: boolean
  step: LoadingStep
}>()

defineEmits<{
  cancel: []
}>()
const { canAnimate, spring } = useUiMotion()

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

      <button class="cancel-action" type="button" @click="$emit('cancel')">
        Annuler
      </button>
    </header>

    <div class="center-stack">
      <MotionReveal><h1 id="loading-title" class="main-title">Recherche du passage</h1></MotionReveal>

      <p class="main-subtitle" role="status" aria-live="polite">
        {{ activeLabel }}
      </p>

      <RecognitionActionButton class="loading-action" disabled loading :show-label="false" />

      <ol class="loading-steps" aria-label="Progression de l’analyse">
        <li class="progress-rail" aria-hidden="true">
          <Motion
            class="progress-fill"
            :initial="false"
            :animate="{ scaleX: activeStepIndex / (steps.length - 1) }"
            :transition="spring"
          />
        </li>
        <li
          v-for="item in steps"
          :key="item.key"
          class="loading-step"
          :class="`is-${getState(item.key)}`"
          :aria-current="getState(item.key) === 'active' ? 'step' : undefined"
        >
          <Motion
            as="span"
            class="step-indicator"
            aria-hidden="true"
            :initial="false"
            :animate="{ scale: canAnimate && getState(item.key) === 'active' ? 1.12 : 1 }"
            :transition="spring"
          >
            <Check v-if="getState(item.key) === 'done'" :size="14" :stroke-width="2.5" />
            <Motion
              v-else
              as="span"
              class="step-dot"
              :initial="false"
              :animate="{ opacity: canAnimate && getState(item.key) === 'active' ? [0.5, 1, 0.5] : 1 }"
              :transition="{ type: 'tween', duration: canAnimate ? 1.8 : 0, repeat: canAnimate && getState(item.key) === 'active' ? Infinity : 0 }"
            />
          </Motion>
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
  justify-content: flex-end;
  gap: 16px;
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

.main-title {
  margin: 14px 0 0;
  font-size: clamp(28px, 4vw, 40px);
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
  position: relative;
  width: min(100%, 420px);
  margin: 8px 0 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  list-style: none;
}

.progress-rail {
  position: absolute;
  top: 13px;
  left: calc(100% / 6);
  right: calc(100% / 6);
  height: 2px;
  background: rgba(148, 163, 184, .16);
}

.progress-fill {
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #7dd3fc);
  transform-origin: left;
  box-shadow: 0 0 12px rgba(96, 165, 250, .3);
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

.step-indicator {
  position: relative;
  z-index: 1;
  width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  border-radius: 999px;
  background: #101c2d;
  border: 1px solid #33465e;
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
}

.loading-step.is-active .step-indicator {
  border-color: #60a5fa;
  background: #142d4e;
}

.loading-step.is-done .step-indicator {
  border-color: #3b82f6;
  background: #1d4ed8;
  color: #fff;
}

.loading-step.is-done {
  color: #91a2b8;
}

.loading-step.is-done .step-dot {
  background: #60a5fa;
}

@media (max-width: 640px) {
  .loading-screen {
    padding: 20px 16px 28px;
  }

  .main-title {
    font-size: clamp(28px, 7vw, 34px);
  }

  .main-subtitle {
    min-height: 48px;
    max-width: 340px;
    font-size: 16px;
  }
}

</style>
