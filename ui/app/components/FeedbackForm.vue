<script setup lang="ts">
// ROLE
// ----
// Feedback utilisateur plus visuel :
// - pouce vert = validation rapide avec petite animation
// - pouce rouge = ouverture du formulaire de correction
// - envoi backend conservé

import { computed, onBeforeUnmount, ref, watch } from 'vue'

import { useFeedback } from '~/composables/useFeedback'
import { useSurahOptions } from '~/composables/useSurahOptions'
import type { RecognizeResponse } from '~/composables/useRecognition'

const props = defineProps<{
  result: RecognizeResponse
}>()

const feedbackSent = ref(false)
const showCorrectionForm = ref(false)
const successBurst = ref(false)
const selectedSurahId = ref<number | null>(null)
const selectedStartVerse = ref<number | null>(null)
const selectedEndVerse = ref<number | null>(null)
const correctionComment = ref('')
let successBurstTimeoutId: number | null = null

const { sending, error, sendFeedback } = useFeedback()
const {
  surahs,
  loading: surahsLoading,
  error: surahsError,
  fetchSurahOptions,
} = useSurahOptions()

const selectedSurah = computed(() => {
  return surahs.value.find(surah => surah.id === selectedSurahId.value) ?? null
})

const availableVerseNumbers = computed(() => {
  const totalVerses = selectedSurah.value?.total_verses ?? 0
  return Array.from({ length: totalVerses }, (_unused, index) => index + 1)
})

const availableEndVerseNumbers = computed(() => {
  return availableVerseNumbers.value.filter((value) => {
    return selectedStartVerse.value === null || value >= selectedStartVerse.value
  })
})

const formError = computed(() => error.value ?? surahsError.value)

const canSubmitCorrection = computed(() => {
  return !!selectedSurah.value
    && selectedStartVerse.value !== null
    && selectedEndVerse.value !== null
    && selectedEndVerse.value >= selectedStartVerse.value
    && !sending.value
    && !surahsLoading.value
})

function clampVerse(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

function clearSuccessBurstTimeout() {
  if (successBurstTimeoutId === null) {
    return
  }

  window.clearTimeout(successBurstTimeoutId)
  successBurstTimeoutId = null
}

function syncCorrectionSelection(force = false) {
  if (!surahs.value.length) return

  const detectedVerse = props.result.verse
  const currentSurah = surahs.value.find(surah => surah.id === selectedSurahId.value)
  const nextSurah = !force && currentSurah
    ? currentSurah
    : (surahs.value.find(surah => surah.id === detectedVerse?.sourate_id) ?? surahs.value[0])

  selectedSurahId.value = nextSurah.id

  const defaultStartVerse = detectedVerse?.sourate_id === nextSurah.id
    ? detectedVerse.start_verse
    : 1
  const defaultEndVerse = detectedVerse?.sourate_id === nextSurah.id
    ? detectedVerse.end_verse
    : defaultStartVerse

  const nextStartVerse = clampVerse(
    selectedStartVerse.value ?? defaultStartVerse,
    1,
    nextSurah.total_verses,
  )
  const nextEndVerse = clampVerse(
    selectedEndVerse.value ?? defaultEndVerse,
    nextStartVerse,
    nextSurah.total_verses,
  )

  selectedStartVerse.value = nextStartVerse
  selectedEndVerse.value = nextEndVerse
}

function formatSurahOptionLabel(id: number, name: string, transliteration: string) {
  return `${id} - ${name} - ${transliteration}`
}

async function submitPositiveFeedback() {
  try {
    await sendFeedback({
      is_correct: true,
      transcription_text: props.result.transcription_text,
      detected_verse: props.result.verse,
      correction: null,
      comment: null,
    })

    clearSuccessBurstTimeout()
    successBurst.value = true
    feedbackSent.value = true

    successBurstTimeoutId = window.setTimeout(() => {
      successBurst.value = false
      successBurstTimeoutId = null
    }, 900)
  } catch {
    // erreur déjà gérée dans le composable
  }
}

async function openCorrectionForm() {
  showCorrectionForm.value = true

  try {
    await fetchSurahOptions()
    syncCorrectionSelection()
  } catch {
    // erreur déjà gérée dans le composable
  }
}

async function submitCorrection() {
  if (!selectedSurah.value || selectedStartVerse.value === null || selectedEndVerse.value === null) return

  try {
    await sendFeedback({
      is_correct: false,
      transcription_text: props.result.transcription_text,
      detected_verse: props.result.verse,
      correction: {
        sourate_id: selectedSurah.value.id,
        sourate_name: selectedSurah.value.name,
        transliteration: selectedSurah.value.transliteration,
        start_verse: selectedStartVerse.value,
        end_verse: selectedEndVerse.value,
      },
      comment: correctionComment.value.trim() || null,
    })

    feedbackSent.value = true
  } catch {
    // erreur déjà gérée dans le composable
  }
}

watch(selectedSurahId, (nextSurahId) => {
  const nextSurah = surahs.value.find(surah => surah.id === nextSurahId)

  if (!nextSurah) return

  const nextStartVerse = clampVerse(selectedStartVerse.value ?? 1, 1, nextSurah.total_verses)

  selectedStartVerse.value = nextStartVerse
  selectedEndVerse.value = clampVerse(
    selectedEndVerse.value ?? nextStartVerse,
    nextStartVerse,
    nextSurah.total_verses,
  )
})

watch(selectedStartVerse, (nextStartVerse) => {
  const totalVerses = selectedSurah.value?.total_verses

  if (nextStartVerse === null || totalVerses === undefined) return

  selectedEndVerse.value = clampVerse(
    selectedEndVerse.value ?? nextStartVerse,
    nextStartVerse,
    totalVerses,
  )
})

onBeforeUnmount(() => {
  clearSuccessBurstTimeout()
})
</script>

<template>
  <div class="feedback-card">
    <template v-if="!feedbackSent">
      <p class="eyebrow">Feedback</p>
      <h3 class="title">Le verset détecté est-il correct ?</h3>

      <div v-if="!showCorrectionForm" class="feedback-actions">
        <button type="button" class="feedback-btn positive" :disabled="sending" @click="submitPositiveFeedback">
          <span class="emoji">👍</span>
          <span class="label">Correct</span>
        </button>

        <button type="button" class="feedback-btn negative" :disabled="sending" @click="openCorrectionForm">
          <span class="emoji">👎</span>
          <span class="label">Corriger</span>
        </button>

        <div v-if="successBurst" class="success-burst" aria-hidden="true">
          <span>✨</span>
          <span>✨</span>
          <span>✨</span>
        </div>
      </div>

      <div v-else class="correction-form">
        <div class="field">
          <label for="sourate">Sourate correcte</label>
          <select id="sourate" v-model.number="selectedSurahId" :disabled="sending || surahsLoading || !surahs.length">
            <option v-for="surah in surahs" :key="surah.id" :value="surah.id">
              {{ formatSurahOptionLabel(surah.id, surah.name, surah.transliteration) }}
            </option>
          </select>
          <p v-if="surahsLoading" class="field-hint">Chargement des sourates...</p>
        </div>

        <div class="field-row">
          <div class="field">
            <label for="start-verse">Verset début</label>
            <select id="start-verse" v-model.number="selectedStartVerse" :disabled="sending || surahsLoading || !availableVerseNumbers.length">
              <option v-for="verseNumber in availableVerseNumbers" :key="`start-${verseNumber}`" :value="verseNumber">
                {{ verseNumber }}
              </option>
            </select>
          </div>

          <div class="field">
            <label for="end-verse">Verset fin</label>
            <select id="end-verse" v-model.number="selectedEndVerse" :disabled="sending || surahsLoading || !availableVerseNumbers.length">
              <option v-for="verseNumber in availableEndVerseNumbers" :key="`end-${verseNumber}`" :value="verseNumber">
                {{ verseNumber }}
              </option>
            </select>
          </div>
        </div>

        <div class="field">
          <label for="comment">Commentaire optionnel</label>
          <textarea id="comment" v-model="correctionComment" rows="4"
            placeholder="Précision utile pour améliorer la détection" />
        </div>

        <div class="correction-actions">
          <button type="button" class="secondary-btn" :disabled="sending" @click="showCorrectionForm = false">
            Retour
          </button>

          <button type="button" class="submit-btn" :disabled="!canSubmitCorrection" @click="submitCorrection">
            {{ sending ? 'Envoi...' : (surahsLoading ? 'Chargement...' : 'Envoyer la correction') }}
          </button>
        </div>
      </div>

      <p v-if="formError" class="error-message">
        {{ formError }}
      </p>
    </template>

    <template v-else>
      <p class="eyebrow">Feedback</p>
      <h3 class="title">Merci pour votre retour</h3>
      <p class="thank-you">
        Votre validation aidera à améliorer Sawt AI.
      </p>
    </template>
  </div>
</template>

<style scoped>
.feedback-card {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 28px;
  padding: 24px;
  background: rgba(15, 23, 42, 0.6);
  backdrop-filter: blur(12px);
  box-shadow: 0 24px 80px rgba(2, 6, 23, 0.35);
}

.eyebrow {
  margin: 0 0 6px;
  font-size: 13px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #93c5fd;
}

.title {
  margin: 0;
  font-size: 22px;
  line-height: 1.3;
  font-weight: 800;
}

.feedback-actions {
  position: relative;
  display: flex;
  justify-content: center;
  gap: 24px;
  margin-top: 26px;
  flex-wrap: wrap;
}

.feedback-btn {
  min-width: 150px;
  border: none;
  border-radius: 22px;
  padding: 20px 22px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  transition:
    transform 0.22s ease,
    box-shadow 0.22s ease,
    border-color 0.22s ease,
    background 0.22s ease;
  font-weight: 700;
}

.feedback-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.feedback-btn .emoji {
  font-size: 42px;
  line-height: 1;
}

.feedback-btn .label {
  font-size: 15px;
}

.positive {
  background: rgba(34, 197, 94, 0.12);
  border: 1px solid rgba(74, 222, 128, 0.28);
  color: #4ade80;
  box-shadow: 0 0 0 rgba(34, 197, 94, 0);
}

.positive:hover {
  transform: translateY(-6px) scale(1.04);
  box-shadow: 0 0 28px rgba(34, 197, 94, 0.28);
}

.negative {
  background: rgba(239, 68, 68, 0.12);
  border: 1px solid rgba(248, 113, 113, 0.28);
  color: #f87171;
  box-shadow: 0 0 0 rgba(239, 68, 68, 0);
}

.negative:hover {
  transform: translateY(-6px) scale(1.04);
  box-shadow: 0 0 28px rgba(239, 68, 68, 0.28);
}

.success-burst {
  position: absolute;
  left: 50%;
  top: 10px;
  transform: translateX(-50%);
  pointer-events: none;
  animation: burst-fade 0.9s ease-out forwards;
}

.success-burst span {
  position: absolute;
  font-size: 24px;
  color: #fde68a;
}

.success-burst span:nth-child(1) {
  transform: translate(-42px, 0);
}

.success-burst span:nth-child(2) {
  transform: translate(0, -20px);
}

.success-burst span:nth-child(3) {
  transform: translate(42px, 0);
}

.correction-form {
  margin-top: 22px;
  display: grid;
  gap: 14px;
}

.field-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.field {
  display: grid;
  gap: 8px;
}

label {
  font-size: 14px;
  color: #cbd5e1;
}

input,
select,
textarea {
  width: 100%;
  box-sizing: border-box;
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(2, 6, 23, 0.45);
  color: #fff;
  font: inherit;
  transition:
    border-color 0.2s ease,
    box-shadow 0.2s ease,
    background 0.2s ease;
}

input:focus,
select:focus,
textarea:focus {
  outline: none;
  border-color: rgba(96, 165, 250, 0.45);
  box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.12);
  background: rgba(2, 6, 23, 0.62);
}

textarea {
  resize: vertical;
}

.field-hint {
  margin: 0;
  font-size: 13px;
  color: #94a3b8;
}

.correction-actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.secondary-btn,
.submit-btn {
  border: none;
  border-radius: 999px;
  padding: 12px 18px;
  font-weight: 700;
  cursor: pointer;
  transition: transform 0.2s ease, opacity 0.2s ease;
}

.secondary-btn:hover,
.submit-btn:hover {
  transform: translateY(-1px);
}

.secondary-btn:disabled,
.submit-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.secondary-btn {
  background: rgba(255, 255, 255, 0.06);
  color: #e2e8f0;
  border: 1px solid rgba(148, 163, 184, 0.16);
}

.submit-btn {
  background: linear-gradient(135deg, #2563eb, #1d4ed8);
  color: #fff;
}

.error-message {
  margin-top: 14px;
  color: #fecaca;
}

.thank-you {
  margin-top: 12px;
  color: #cbd5e1;
  line-height: 1.6;
}

@keyframes burst-fade {
  0% {
    opacity: 0;
    transform: translateX(-50%) scale(0.4);
  }

  35% {
    opacity: 1;
    transform: translateX(-50%) scale(1.15);
  }

  100% {
    opacity: 0;
    transform: translateX(-50%) scale(1.55);
  }
}

@media (max-width: 640px) {
  .feedback-actions {
    gap: 16px;
  }

  .feedback-btn {
    min-width: 130px;
    padding: 18px 18px;
  }

  .field-row {
    grid-template-columns: 1fr;
  }
}
</style>
