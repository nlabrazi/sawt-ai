<script setup lang="ts">
import { computed } from 'vue'

import FeedbackForm from '~/components/FeedbackForm.vue'
import ResultCard from '~/components/ResultCard.vue'
import type { RecognizeResponse } from '~/composables/useRecognition'
import { getVerseConfidenceUi } from '~/utils/verseConfidence'

const props = defineProps<{
  result: RecognizeResponse | null
  error?: string | null
}>()

defineEmits<{
  reset: []
}>()

const banner = computed(() => {
  if (props.error) {
    return {
      label: 'Analyse incomplète',
      text: props.error,
      className: 'banner-error',
    }
  }

  if (!props.result?.verse) {
    return {
      label: 'Aucun verset confirmé',
      text: 'Aucune correspondance suffisamment fiable n’a été trouvée pour cet extrait.',
      className: 'banner-error',
    }
  }

  const ui = getVerseConfidenceUi(props.result.verse.similarity)

  return {
    label: ui.label,
    text: ui.description,
    className: ui.className,
  }
})

const introText = computed(() => {
  if (props.error) return 'Le résultat demande une nouvelle tentative.'
  if (!props.result?.verse) return 'Aucun passage fiable n’a pu être confirmé.'
  return 'Le passage le plus probable a été préparé pour vérification.'
})
</script>

<template>
  <section class="screen result-screen">
    <div class="top-bar">
      <div class="title-block">
        <p class="brand-kicker">Sawt AI</p>
        <h1 class="page-title">Résultat</h1>
        <p class="page-subtitle">{{ introText }}</p>
      </div>

      <button class="top-action" type="button" @click="$emit('reset')">
        Nouvelle détection
      </button>
    </div>

    <div class="banner" :class="banner.className">
      <p class="banner-label">{{ banner.label }}</p>
      <p class="banner-text">{{ banner.text }}</p>
    </div>

    <div class="content-stack">
      <ResultCard v-if="result" :result="result" />
      <FeedbackForm v-if="result?.verse" :result="result" />
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

.result-screen {
  max-width: 980px;
  width: 100%;
  margin: 0 auto;
}

.top-bar {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
}

.title-block {
  max-width: 540px;
}

.brand-kicker {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: #93c5fd;
}

.page-title {
  margin: 10px 0 0;
  font-size: clamp(38px, 6vw, 56px);
  line-height: 0.98;
  font-weight: 800;
  letter-spacing: -0.05em;
}

.page-subtitle {
  margin: 14px 0 0;
  font-size: 17px;
  line-height: 1.6;
  color: #c8d4e4;
}

.top-action {
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(15, 23, 42, 0.56);
  color: #e2e8f0;
  border-radius: 999px;
  min-height: 46px;
  padding: 0 16px;
  cursor: pointer;
  font-weight: 700;
  backdrop-filter: blur(10px);
  transition: transform 0.2s ease, background 0.2s ease, border-color 0.2s ease;
}

.top-action:hover {
  transform: translateY(-1px);
  background: rgba(30, 41, 59, 0.78);
  border-color: rgba(147, 197, 253, 0.28);
}

.banner {
  margin-top: 22px;
  border-radius: 26px;
  padding: 18px 20px;
  border: 1px solid rgba(148, 163, 184, 0.14);
  backdrop-filter: blur(14px);
  box-shadow: 0 18px 46px rgba(2, 6, 23, 0.12);
}

.banner-label {
  margin: 0;
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.banner-text {
  margin: 8px 0 0;
  line-height: 1.65;
  color: #e2e8f0;
}

.banner-success {
  background: linear-gradient(180deg, rgba(20, 83, 45, 0.26) 0%, rgba(7, 36, 22, 0.22) 100%);
}

.banner-success .banner-label {
  color: #bbf7d0;
}

.banner-warning {
  background: linear-gradient(180deg, rgba(120, 53, 15, 0.26) 0%, rgba(50, 27, 8, 0.22) 100%);
}

.banner-warning .banner-label {
  color: #fde68a;
}

.banner-error {
  background: linear-gradient(180deg, rgba(127, 29, 29, 0.24) 0%, rgba(49, 18, 18, 0.2) 100%);
}

.banner-error .banner-label {
  color: #fecaca;
}

.content-stack {
  margin-top: 22px;
  display: grid;
  gap: 22px;
}

@media (max-width: 768px) {
  .screen {
    padding: 18px 14px 28px;
  }

  .top-bar {
    flex-direction: column;
    align-items: stretch;
  }

  .page-subtitle {
    font-size: 16px;
  }

  .top-action {
    width: 100%;
  }
}
</style>
