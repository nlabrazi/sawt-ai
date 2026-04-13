<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import FeedbackForm from '~/components/FeedbackForm.vue'
import VerseDetailsSheet from '~/components/VerseDetailsSheet.vue'
import type { RecognizeResponse } from '~/composables/useRecognition'

const props = defineProps<{
  result: RecognizeResponse | null
  error?: string | null
}>()

defineEmits<{
  reset: []
}>()

const detailsOpen = ref(false)
const feedbackCompleted = ref(false)

watch(() => props.result, () => {
  detailsOpen.value = false
  feedbackCompleted.value = false
})

const heroState = computed(() => {
  const verse = props.result?.verse
  const score = verse?.similarity ?? 0

  if (props.error || !verse) {
    return {
      label: 'Résultat à confirmer',
      tone: 'warning',
      description: props.error ?? 'Aucune correspondance suffisamment fiable n’a été trouvée pour cet extrait.',
    }
  }

  if (score >= 90) {
    return {
      label: 'Résultat fiable',
      tone: 'success',
      description: 'Le passage détecté présente une forte cohérence.',
    }
  }

  if (score >= 70) {
    return {
      label: 'Correspondance probable',
      tone: 'neutral',
      description: 'Une correspondance a été trouvée, mais elle mérite une validation rapide.',
    }
  }

  return {
    label: 'Résultat à confirmer',
    tone: 'warning',
    description: 'Le passage détecté reste plausible, mais doit être vérifié.',
  }
})

const verseLabel = computed(() => {
  const verse = props.result?.verse
  if (!verse) return 'Aucun verset confirmé'
  return verse.start_verse === verse.end_verse
    ? `Verset ${verse.start_verse}`
    : `Versets ${verse.start_verse} à ${verse.end_verse}`
})

const imamLabel = computed(() => {
  const name = props.result?.imam_predictions?.[0]?.name
  if (name) return name.replace(/_/g, ' ').trim()

  switch (props.result?.imam_status) {
    case 'disabled':
      return 'Détection imam désactivée'
    case 'unavailable':
      return 'Reconnaissance imam indisponible'
    case 'low':
      return 'Imam à confirmer'
    case 'unknown':
      return 'Imam non reconnu'
    default:
      return 'Imam indisponible'
  }
})
</script>

<template>
  <section class="screen result-screen">
    <header class="result-header">
      <p class="brand-mark">SAWT AI</p>

      <button class="top-action" type="button" @click="$emit('reset')">
        Nouvelle détection
      </button>
    </header>

    <div class="result-shell">
      <transition name="hero-fade" appear>
        <section v-if="result?.verse" class="hero">
          <div class="hero-aura" :class="`tone-${heroState.tone}`" />

          <p class="state-label" :class="`tone-${heroState.tone}`">
            {{ heroState.label }}
          </p>

          <div class="hero-main">
            <p class="surah-arabic">
              {{ result.verse.sourate_name }}
            </p>

            <h1 class="surah-latin">
              {{ result.verse.transliteration }}
            </h1>

            <p class="verse-line">
              {{ verseLabel }}
            </p>

            <p class="imam-line">
              {{ imamLabel }}
            </p>
          </div>

          <div class="hero-actions">
            <button class="primary-btn" type="button" @click="detailsOpen = true">
              Voir le verset
            </button>

            <button class="ghost-btn" type="button" @click="$emit('reset')">
              Recommencer
            </button>
          </div>

          <p class="hero-description">
            {{ heroState.description }}
          </p>
        </section>

        <section v-else class="hero empty-hero">
          <p class="state-label tone-warning">Analyse incomplète</p>
          <h1 class="surah-latin">Aucun résultat confirmé</h1>
          <p class="hero-description">
            {{ heroState.description }}
          </p>

          <div class="hero-actions">
            <button class="primary-btn" type="button" @click="$emit('reset')">
              Réessayer
            </button>
          </div>
        </section>
      </transition>

      <section v-if="result?.verse" class="feedback-zone">
        <FeedbackForm
          :result="result"
          @completed="feedbackCompleted = true"
        />

        <p v-if="feedbackCompleted" class="feedback-helper">
          Retour enregistré. Vous pouvez relancer une nouvelle détection à tout moment.
        </p>
      </section>
    </div>

    <VerseDetailsSheet
      v-if="result?.verse && result"
      :open="detailsOpen"
      :result="result"
      @close="detailsOpen = false"
    />
  </section>
</template>

<style scoped>
.screen {
  position: relative;
  z-index: 1;
  min-height: calc(100vh - 76px);
  padding: 24px 20px 36px;
}

.result-screen {
  width: min(980px, 100%);
  margin: 0 auto;
  display: flex;
  flex-direction: column;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}

.brand-mark {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.26em;
  text-transform: uppercase;
  color: #93c5fd;
}

.top-action {
  border: 1px solid rgba(148, 163, 184, 0.16);
  background: rgba(15, 23, 42, 0.42);
  color: #e2e8f0;
  border-radius: 999px;
  padding: 10px 16px;
  font-weight: 700;
  cursor: pointer;
  backdrop-filter: blur(10px);
  transition: transform 0.2s ease, background 0.2s ease;
}

.top-action:hover {
  transform: translateY(-1px);
  background: rgba(15, 23, 42, 0.66);
}

.result-shell {
  flex: 1;
  display: grid;
  align-content: center;
  gap: 28px;
  padding: 28px 0 8px;
}

.hero {
  position: relative;
  overflow: hidden;
  padding: 34px 28px 28px;
  border-radius: 36px;
  background:
    linear-gradient(180deg, rgba(4, 14, 36, 0.84), rgba(1, 8, 20, 0.92));
  border: 1px solid rgba(96, 165, 250, 0.1);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.04),
    0 30px 80px rgba(2, 6, 23, 0.28);
  text-align: center;
}

.hero-aura {
  position: absolute;
  inset: auto 50% -90px;
  width: 360px;
  height: 220px;
  transform: translateX(-50%);
  border-radius: 999px;
  filter: blur(54px);
  opacity: 0.75;
  pointer-events: none;
}

.tone-success.hero-aura { background: rgba(16, 185, 129, 0.16); }
.tone-neutral.hero-aura { background: rgba(59, 130, 246, 0.18); }
.tone-warning.hero-aura { background: rgba(245, 158, 11, 0.18); }

.state-label {
  position: relative;
  z-index: 1;
  display: inline-flex;
  min-height: 38px;
  padding: 0 16px;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  font-size: 12px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-weight: 800;
}

.state-label.tone-success {
  color: #bbf7d0;
  background: rgba(16, 185, 129, 0.12);
}

.state-label.tone-neutral {
  color: #dbeafe;
  background: rgba(59, 130, 246, 0.14);
}

.state-label.tone-warning {
  color: #fde68a;
  background: rgba(245, 158, 11, 0.14);
}

.hero-main {
  position: relative;
  z-index: 1;
  margin-top: 24px;
}

.surah-arabic {
  margin: 0;
  font-family: "Amiri", serif;
  font-size: clamp(58px, 11vw, 88px);
  line-height: 0.96;
  text-shadow: 0 0 26px rgba(147, 197, 253, 0.08);
}

.surah-latin {
  margin: 12px 0 0;
  font-size: clamp(34px, 7vw, 52px);
  line-height: 0.98;
  letter-spacing: -0.05em;
}

.verse-line {
  margin: 12px 0 0;
  font-size: 18px;
  color: #cbd5e1;
}

.imam-line {
  margin: 8px 0 0;
  color: #93c5fd;
  font-weight: 700;
  font-size: 16px;
}

.hero-description {
  position: relative;
  z-index: 1;
  margin: 18px auto 0;
  max-width: 520px;
  line-height: 1.7;
  color: #94a3b8;
}

.hero-actions {
  position: relative;
  z-index: 1;
  margin-top: 24px;
  display: flex;
  gap: 12px;
  justify-content: center;
  flex-wrap: wrap;
}

.primary-btn,
.ghost-btn {
  border: none;
  min-height: 52px;
  padding: 0 22px;
  border-radius: 999px;
  font-weight: 800;
  cursor: pointer;
  transition: transform 0.18s ease, opacity 0.18s ease, background 0.18s ease;
}

.primary-btn:hover,
.ghost-btn:hover {
  transform: translateY(-1px);
}

.primary-btn {
  background: linear-gradient(135deg, #3b82f6, #2563eb 60%, #1d4ed8);
  color: #fff;
  box-shadow: 0 16px 34px rgba(37, 99, 235, 0.24);
}

.ghost-btn {
  background: rgba(255, 255, 255, 0.06);
  color: #dbeafe;
}

.feedback-zone {
  width: min(640px, 100%);
  margin: 0 auto;
  padding: 22px;
  border-radius: 30px;
  background: rgba(3, 12, 31, 0.58);
  border: 1px solid rgba(148, 163, 184, 0.08);
  backdrop-filter: blur(10px);
}

.feedback-helper {
  margin: 14px 0 0;
  text-align: center;
  color: #64748b;
  line-height: 1.6;
}

.empty-hero {
  max-width: 720px;
  margin: 0 auto;
}

.hero-fade-enter-active,
.hero-fade-leave-active {
  transition: opacity 0.24s ease, transform 0.24s ease;
}

.hero-fade-enter-from,
.hero-fade-leave-to {
  opacity: 0;
  transform: translateY(10px) scale(0.985);
}

@media (max-width: 768px) {
  .screen {
    min-height: calc(100vh - 70px);
    padding: 18px 16px 24px;
  }

  .result-shell {
    padding-top: 22px;
    gap: 18px;
  }

  .result-header {
    flex-direction: column;
    align-items: stretch;
  }

  .hero {
    min-height: 58vh;
    display: grid;
    align-content: center;
    padding: 30px 20px 24px;
    border-radius: 30px;
  }

  .hero-actions {
    flex-direction: column;
  }

  .primary-btn,
  .ghost-btn {
    width: 100%;
  }

  .feedback-zone {
    padding: 18px;
    border-radius: 24px;
  }
}
</style>
