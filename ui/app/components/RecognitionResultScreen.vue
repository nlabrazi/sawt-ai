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

const rejectionCopy = computed(() => {
  switch (props.result?.detection?.rejection_reason) {
    case 'insufficient_speech':
      return {
        heading: 'Récitation trop courte',
        introduction:
          'Récitez distinctement pendant quelques secondes, puis laissez Sawt AI analyser.',
      }
    case 'non_arabic_speech':
      return {
        heading: 'Récitation en arabe non détectée',
        introduction:
          'Cet audio semble contenir une autre langue. Essayez à nouveau avec un passage du Coran.',
      }
    case 'low_transcription_confidence':
      return {
        heading: 'Audio trop difficile à analyser',
        introduction:
          'La voix est trop couverte ou incertaine. Rapprochez-vous du micro et réduisez le bruit ambiant.',
      }
    case 'ambiguous_match':
      return {
        heading: 'Plusieurs passages sont possibles',
        introduction: 'Récitez quelques mots supplémentaires pour départager les correspondances.',
      }
    default:
      return {
        heading: 'Aucun passage reconnu',
        introduction: 'Essayez un extrait un peu plus long, avec moins de bruit autour de vous.',
      }
  }
})

const heading = computed(() => {
  if (props.error) return 'Analyse interrompue'
  if (props.result?.verse) return 'Passage proposé'
  return rejectionCopy.value.heading
})

const introduction = computed(() => {
  if (props.error) return 'La récitation n’a pas pu être analysée cette fois-ci.'
  if (props.result?.verse) return 'Vérifiez la sourate et les versets avant de valider le résultat.'
  return rejectionCopy.value.introduction
})

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
  <section class="screen result-screen" aria-labelledby="result-title">
    <header class="brand" aria-label="Sawt AI">
      <span class="brand-name">Sawt</span>
      <span class="brand-mark">AI</span>
    </header>

    <div class="result-intro">
      <p class="eyebrow">Résultat de l’analyse</p>
      <h1 id="result-title" class="main-title">{{ heading }}</h1>
      <p class="main-subtitle">{{ introduction }}</p>
    </div>

    <div class="content-stack">
      <div v-if="statusPanel" class="status-panel" role="alert">
        <p class="status-label">{{ statusPanel.label }}</p>
        <p class="status-text">{{ statusPanel.text }}</p>
      </div>

      <ResultCard v-if="result?.verse" :result="result" />
      <FeedbackForm v-if="result?.verse" :result="result" />
    </div>

    <button class="reset-action" type="button" @click="$emit('reset')">
      <RotateCcw class="reset-icon" :stroke-width="2" aria-hidden="true" />
      Nouvelle récitation
    </button>
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

.result-screen {
  width: min(100%, 780px);
  margin: 0 auto;
  padding: 26px 20px 42px;
  display: grid;
  align-content: start;
  gap: 24px;
}

.brand {
  display: inline-flex;
  align-items: baseline;
  justify-content: center;
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

.result-intro {
  margin-top: 22px;
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
  margin: 12px 0 0;
  font-size: clamp(36px, 6vw, 54px);
  line-height: 1.04;
  font-weight: 800;
  letter-spacing: -0.05em;
  text-wrap: balance;
}

.main-subtitle {
  margin: 14px auto 0;
  max-width: 540px;
  color: #9fb0c4;
  font-size: 16px;
  line-height: 1.55;
  text-wrap: balance;
}

.content-stack {
  display: grid;
  gap: 18px;
}

.status-panel {
  padding: 16px 18px;
  border: 1px solid rgba(248, 113, 113, 0.2);
  border-radius: 18px;
  background: rgba(127, 29, 29, 0.2);
}

.status-label {
  margin: 0;
  color: #fecaca;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.status-text {
  margin: 6px 0 0;
  color: #e2e8f0;
  line-height: 1.55;
}

.reset-action {
  justify-self: center;
  min-height: 46px;
  padding: 0 18px;
  border: 1px solid rgba(147, 197, 253, 0.22);
  border-radius: 999px;
  background: rgba(30, 64, 175, 0.18);
  color: #dbeafe;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  font: inherit;
  font-size: 14px;
  font-weight: 750;
  cursor: pointer;
  transition:
    color 180ms ease,
    border-color 180ms ease,
    background 180ms ease;
}

.reset-action:hover {
  border-color: rgba(147, 197, 253, 0.4);
  background: rgba(37, 99, 235, 0.24);
  color: #fff;
}

.reset-action:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 3px;
}

.reset-icon {
  width: 17px;
  height: 17px;
}

@media (max-width: 640px) {
  .result-screen {
    padding: 20px 14px 30px;
    gap: 18px;
  }

  .result-intro {
    margin-top: 14px;
  }

  .main-title {
    font-size: clamp(34px, 10vw, 44px);
  }

  .content-stack {
    gap: 14px;
  }

  .reset-action {
    width: 100%;
  }
}
</style>
