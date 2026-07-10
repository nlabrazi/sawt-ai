<script setup lang="ts">
import ThumbsDown from '@lucide/vue/dist/esm/icons/thumbs-down.mjs'
import ThumbsUp from '@lucide/vue/dist/esm/icons/thumbs-up.mjs'
import { computed, ref, watch } from 'vue'

import MiniToast from '~/components/MiniToast.vue'
import { useFeedback } from '~/composables/useFeedback'
import { useMiniToast } from '~/composables/useMiniToast'
import { useSurahOptions } from '~/composables/useSurahOptions'
import type { RecognizeResponse } from '~/composables/useRecognition'

const props = defineProps<{
  result: RecognizeResponse
}>()

const emit = defineEmits<{
  completed: []
}>()

const feedbackSent = ref(false)
const showCorrectionForm = ref(false)
const selectedSurahId = ref<number | null>(null)
const selectedStartVerse = ref<number | null>(null)
const selectedEndVerse = ref<number | null>(null)
const correctionComment = ref('')
const { message: toastMessage, visible: toastVisible, show: showToast } = useMiniToast()

const { sending, error, sendFeedback } = useFeedback()
const { surahs, loading: surahsLoading, error: surahsError, fetchSurahOptions } = useSurahOptions()

const selectedSurah = computed(() => {
  return surahs.value.find((surah) => surah.id === selectedSurahId.value) ?? null
})

const availableVerseNumbers = computed(() => {
  const totalVerses = selectedSurah.value?.total_verses ?? 0
  return Array.from({ length: totalVerses }, (_, index) => index + 1)
})

const availableEndVerseNumbers = computed(() => {
  return availableVerseNumbers.value.filter((value) => {
    return selectedStartVerse.value === null || value >= selectedStartVerse.value
  })
})

const formError = computed(() => error.value ?? surahsError.value)

const canSubmitCorrection = computed(() => {
  return (
    !!selectedSurah.value &&
    selectedStartVerse.value !== null &&
    selectedEndVerse.value !== null &&
    selectedEndVerse.value >= selectedStartVerse.value &&
    !sending.value &&
    !surahsLoading.value
  )
})

function clampVerse(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

function syncCorrectionSelection(force = false) {
  if (!surahs.value.length) return

  const detectedVerse = props.result.verse
  const currentSurah = surahs.value.find((surah) => surah.id === selectedSurahId.value)
  const nextSurah =
    !force && currentSurah
      ? currentSurah
      : (surahs.value.find((surah) => surah.id === detectedVerse?.sourate_id) ?? surahs.value[0])

  if (!nextSurah) return

  selectedSurahId.value = nextSurah.id

  const defaultStartVerse =
    detectedVerse?.sourate_id === nextSurah.id ? detectedVerse.start_verse : 1
  const defaultEndVerse =
    detectedVerse?.sourate_id === nextSurah.id ? detectedVerse.end_verse : defaultStartVerse

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
  return `${id} · ${name} · ${transliteration}`
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

    feedbackSent.value = true
    showToast('Retour envoyé, merci de votre contribution !')
    emit('completed')
  } catch {
    // handled in composable
  }
}

async function openCorrectionForm() {
  showCorrectionForm.value = true

  try {
    await fetchSurahOptions()
    syncCorrectionSelection()
  } catch {
    // handled in composable
  }
}

async function submitCorrection() {
  if (!selectedSurah.value || selectedStartVerse.value === null || selectedEndVerse.value === null)
    return

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
    showToast('Retour envoyé, merci de votre contribution !')
    emit('completed')
  } catch {
    // handled in composable
  }
}

watch(selectedSurahId, (nextSurahId) => {
  const nextSurah = surahs.value.find((surah) => surah.id === nextSurahId)
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
</script>

<template>
  <div v-if="!feedbackSent" class="feedback">
      <div class="feedback-head">
        <p class="feedback-label">Validation</p>
        <p class="feedback-title">Ce résultat est-il correct&nbsp;?</p>
      </div>

      <div v-if="!showCorrectionForm" class="feedback-actions">
        <button class="feedback-action feedback-action-primary" type="button"
          aria-label="Le résultat est correct" title="Résultat correct" :disabled="sending"
          @click="submitPositiveFeedback">
          <ThumbsUp :stroke-width="2" aria-hidden="true" />
        </button>

        <button class="feedback-action feedback-action-secondary" type="button"
          aria-label="Le résultat est incorrect" title="Résultat incorrect" :disabled="sending"
          @click="openCorrectionForm">
          <ThumbsDown :stroke-width="2" aria-hidden="true" />
        </button>
      </div>

      <div v-else class="correction-panel">
        <div class="panel-head">
          <p class="panel-title">Correction</p>
          <p class="panel-subtitle">
            Renseignez le passage correct pour améliorer les prochains résultats.
          </p>
        </div>

        <div class="field">
          <label for="sourate">Sourate correcte</label>
          <select id="sourate" v-model.number="selectedSurahId" :disabled="sending || surahsLoading || !surahs.length">
            <option v-for="surah in surahs" :key="surah.id" :value="surah.id">
              {{ formatSurahOptionLabel(surah.id, surah.name, surah.transliteration) }}
            </option>
          </select>
        </div>

        <div class="field-row">
          <div class="field">
            <label for="start-verse">Verset de début</label>
            <select id="start-verse" v-model.number="selectedStartVerse"
              :disabled="sending || surahsLoading || !availableVerseNumbers.length">
              <option v-for="verseNumber in availableVerseNumbers" :key="`start-${verseNumber}`" :value="verseNumber">
                {{ verseNumber }}
              </option>
            </select>
          </div>

          <div class="field">
            <label for="end-verse">Verset de fin</label>
            <select id="end-verse" v-model.number="selectedEndVerse"
              :disabled="sending || surahsLoading || !availableVerseNumbers.length">
              <option v-for="verseNumber in availableEndVerseNumbers" :key="`end-${verseNumber}`" :value="verseNumber">
                {{ verseNumber }}
              </option>
            </select>
          </div>
        </div>

        <div class="field">
          <label for="comment">Commentaire complémentaire</label>
          <textarea id="comment" v-model="correctionComment" rows="3" placeholder="Optionnel" />
        </div>

        <div class="correction-actions">
          <button class="subtle-btn" type="button" :disabled="sending" @click="showCorrectionForm = false">
            Retour
          </button>

          <button class="submit-btn" type="button" :disabled="!canSubmitCorrection" @click="submitCorrection">
            {{ sending ? 'Envoi…' : (surahsLoading ? 'Chargement…' : 'Envoyer la correction') }}
          </button>
        </div>
      </div>

      <p v-if="formError" class="form-error">
        {{ formError }}
      </p>
  </div>

  <MiniToast :open="toastVisible" :message="toastMessage" />
</template>

<style scoped>
.feedback {
  width: 100%;
  border-radius: 28px;
  border: 1px solid rgba(148, 163, 184, 0.12);
  background:
    linear-gradient(180deg, rgba(9, 18, 34, 0.78) 0%, rgba(7, 15, 29, 0.68) 100%);
  backdrop-filter: blur(14px);
  box-shadow:
    0 20px 54px rgba(2, 6, 23, 0.14),
    inset 0 1px 0 rgba(255, 255, 255, 0.04);
  padding: 20px;
}

.feedback-head {
  display: grid;
  gap: 6px;
  justify-items: center;
  text-align: center;
}

.feedback-label {
  margin: 0;
  font-size: 12px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #93c5fd;
}

.feedback-title {
  margin: 0;
  font-size: 20px;
  line-height: 1.2;
  font-weight: 800;
  color: #eff6ff;
}

.feedback-actions {
  margin-top: 14px;
  display: flex;
  justify-content: center;
  gap: 10px;
}

.feedback-action {
  width: 46px;
  height: 42px;
  border-radius: 14px;
  border: 1px solid rgba(148, 163, 184, 0.12);
  padding: 0;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #bfdbfe;
  transition:
    transform 0.2s ease,
    border-color 0.2s ease,
    background 0.2s ease,
    box-shadow 0.2s ease;
}

.feedback-action:hover {
  transform: translateY(-1px);
}

.feedback-action:disabled {
  cursor: default;
  opacity: 0.7;
}

.feedback-action:disabled:hover {
  transform: none;
}

.feedback-action-primary {
  background: rgba(37, 99, 235, 0.16);
  border-color: rgba(147, 197, 253, 0.22);
}

.feedback-action-primary:hover {
  border-color: rgba(147, 197, 253, 0.42);
  background: rgba(37, 99, 235, 0.28);
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.14);
}

.feedback-action-secondary {
  background: rgba(51, 65, 85, 0.42);
  border-color: rgba(148, 163, 184, 0.2);
}

.feedback-action-secondary:hover {
  border-color: rgba(248, 113, 113, 0.34);
  background: rgba(127, 29, 29, 0.24);
  color: #fecaca;
  box-shadow: 0 10px 24px rgba(127, 29, 29, 0.12);
}

.feedback-action-primary:focus-visible {
  outline: 3px solid rgba(96, 165, 250, 0.28);
  outline-offset: 3px;
}

.feedback-action-secondary:focus-visible {
  outline: 3px solid rgba(148, 163, 184, 0.24);
  outline-offset: 3px;
}

.feedback-action svg {
  width: 19px;
  height: 19px;
  fill: none;
  stroke: currentColor;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.correction-panel {
  margin-top: 18px;
  display: grid;
  gap: 14px;
}

.panel-head {
  display: grid;
  gap: 6px;
}

.panel-title {
  margin: 0;
  font-size: 18px;
  font-weight: 800;
  color: #eff6ff;
}

.panel-subtitle {
  margin: 0;
  font-size: 14px;
  line-height: 1.6;
  color: #8ea1b9;
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
  font-size: 13px;
  font-weight: 600;
  color: #dbe4f0;
}

select,
textarea {
  width: 100%;
  border: 1px solid rgba(148, 163, 184, 0.14);
  border-radius: 16px;
  padding: 13px 14px;
  background: rgba(2, 6, 23, 0.74);
  color: #fff;
  font: inherit;
  transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

select:focus,
textarea:focus {
  outline: none;
  border-color: rgba(125, 211, 252, 0.3);
  box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.08);
}

textarea {
  resize: vertical;
  min-height: 96px;
}

.correction-actions {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
  flex-wrap: wrap;
  margin-top: 4px;
}

.subtle-btn,
.submit-btn {
  min-height: 46px;
  border-radius: 999px;
  padding: 0 18px;
  font-weight: 700;
  cursor: pointer;
  transition:
    transform 0.2s ease,
    border-color 0.2s ease,
    background 0.2s ease,
    opacity 0.2s ease;
}

.subtle-btn {
  border: 1px solid rgba(148, 163, 184, 0.14);
  background: rgba(255, 255, 255, 0.04);
  color: #e2e8f0;
}

.subtle-btn:hover {
  transform: translateY(-1px);
  border-color: rgba(148, 163, 184, 0.24);
}

.submit-btn {
  border: 1px solid rgba(125, 211, 252, 0.18);
  background:
    linear-gradient(180deg, rgba(37, 99, 235, 0.94) 0%, rgba(29, 78, 216, 1) 100%);
  color: #fff;
  box-shadow: 0 12px 28px rgba(37, 99, 235, 0.18);
}

.submit-btn:hover:not(:disabled) {
  transform: translateY(-1px);
}

.submit-btn:disabled,
.subtle-btn:disabled {
  cursor: default;
  opacity: 0.62;
}

.submit-btn:disabled:hover,
.subtle-btn:disabled:hover {
  transform: none;
}

.form-error {
  margin: 14px 0 0;
  border-radius: 16px;
  padding: 12px 14px;
  background: rgba(127, 29, 29, 0.2);
  border: 1px solid rgba(248, 113, 113, 0.14);
  color: #fecaca;
  text-align: center;
  font-size: 14px;
  line-height: 1.55;
}

@media (max-width: 768px) {
  .feedback {
    padding: 16px;
    border-radius: 24px;
  }

  .field-row {
    grid-template-columns: 1fr;
  }

  .correction-actions {
    flex-direction: column;
  }

  .subtle-btn,
  .submit-btn {
    width: 100%;
  }
}

</style>
