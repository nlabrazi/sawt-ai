<script setup lang="ts">
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

const LOW_CONFIDENCE_THRESHOLD = 0.8

const banner = computed(() => {
  if (props.error) {
    return {
      text: props.error,
      className: 'banner-error',
    }
  }

  if (!props.result?.verse) {
    return {
      text: 'Aucun verset fiable trouvé.',
      className: 'banner-error',
    }
  }

  const similarity = props.result.verse.similarity
  const isLowConfidence = similarity <= 1
    ? similarity < LOW_CONFIDENCE_THRESHOLD
    : similarity < 80

  if (isLowConfidence) {
    return {
      text: 'Résultat à vérifier.',
      className: 'banner-warning',
    }
  }

  return {
    text: 'Résultat fiable.',
    className: 'banner-success',
  }
})
</script>

<template>
  <section class="screen result-screen">
    <div class="top-bar">
      <div>
        <p class="brand-kicker">Sawt AI</p>
        <h1 class="page-title">Résultat de la reconnaissance</h1>
      </div>

      <button class="top-action" type="button" @click="$emit('reset')">
        Nouvelle détection
      </button>
    </div>

    <div class="banner" :class="banner.className">
      {{ banner.text }}
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
  min-height: 100vh;
  padding: 24px 16px 36px;
  box-sizing: border-box;
}

.result-screen {
  max-width: 980px;
  margin: 0 auto;
}

.top-bar {
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

.page-title {
  margin: 10px 0 0;
  font-size: 34px;
  line-height: 1.08;
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

.banner {
  margin-top: 22px;
  border-radius: 18px;
  padding: 16px 18px;
  font-size: 16px;
  font-weight: 600;
}

.banner-success {
  color: #dcfce7;
  background: rgba(34, 197, 94, 0.14);
  border: 1px solid rgba(34, 197, 94, 0.24);
}

.banner-warning {
  color: #fde68a;
  background: rgba(245, 158, 11, 0.14);
  border: 1px solid rgba(245, 158, 11, 0.24);
}

.banner-error {
  color: #fecaca;
  background: rgba(239, 68, 68, 0.14);
  border: 1px solid rgba(239, 68, 68, 0.24);
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

  .page-title {
    font-size: 28px;
  }
}
</style>
