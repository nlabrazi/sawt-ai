<script setup lang="ts">
import RotateCcw from '@lucide/vue/dist/esm/icons/rotate-ccw.mjs'
import { computed } from 'vue'

import FeedbackForm from '~/components/FeedbackForm.vue'
import ResultCard from '~/components/ResultCard.vue'
import type { RecognizeResponse } from '~/composables/useRecognition'

const props = defineProps<{
  result: RecognizeResponse | null
  error?: string | null
}>()

defineEmits<{
  reset: []
}>()

const statusPanel = computed(() => {
  if (props.error) {
    return {
      label: 'Analyse incomplète',
      text: props.error,
    }
  }

  if (!props.result) {
    return {
      label: 'Aucun résultat',
      text: 'Aucun résultat n’est disponible pour cette tentative.',
    }
  }

  return null
})
</script>

<template>
  <section class="screen result-screen">
    <div class="content-stack">
      <div v-if="statusPanel" class="status-panel">
        <p class="status-label">{{ statusPanel.label }}</p>
        <p class="status-text">{{ statusPanel.text }}</p>
      </div>

      <ResultCard v-if="result" :result="result" />
      <FeedbackForm v-if="result?.verse" :result="result" />
    </div>

    <button class="reset-action" type="button" aria-label="Recommencer la détection" @click="$emit('reset')">
      <RotateCcw class="reset-icon" :stroke-width="2" aria-hidden="true" />
      Recommencer
    </button>
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

.result-screen {
  max-width: 980px;
  width: 100%;
  margin: 0 auto;
  display: grid;
  align-content: start;
  gap: 22px;
}

.status-panel {
  border-radius: 24px;
  padding: 18px 20px;
  border: 1px solid rgba(148, 163, 184, 0.14);
  background: linear-gradient(180deg, rgba(127, 29, 29, 0.24) 0%, rgba(49, 18, 18, 0.2) 100%);
  backdrop-filter: blur(14px);
  box-shadow: 0 18px 46px rgba(2, 6, 23, 0.12);
}

.status-label {
  margin: 0;
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #fecaca;
}

.status-text {
  margin: 8px 0 0;
  line-height: 1.65;
  color: #e2e8f0;
}

.content-stack {
  display: grid;
  gap: 22px;
}

.reset-action {
  justify-self: center;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(15, 23, 42, 0.56);
  color: #e2e8f0;
  border-radius: 999px;
  min-height: 46px;
  padding: 0 18px;
  cursor: pointer;
  font-weight: 700;
  backdrop-filter: blur(10px);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  transition:
    transform 0.2s ease,
    background 0.2s ease,
    border-color 0.2s ease;
}

.reset-action:hover {
  transform: translateY(-1px);
  background: rgba(30, 41, 59, 0.78);
  border-color: rgba(147, 197, 253, 0.28);
}

.reset-icon {
  width: 18px;
  height: 18px;
}

@media (max-width: 768px) {
  .screen {
    padding: 14px 12px 28px;
  }

  .result-screen,
  .content-stack {
    gap: 16px;
  }

  .status-panel {
    border-radius: 22px;
    padding: 16px;
  }

  .reset-action {
    width: 100%;
  }
}
</style>
