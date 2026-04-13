<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useFeedback } from '~/composables/useFeedback'
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
  return Array.from({ length: totalVerses }, (_, index) => index + 1)
})

const availableEndVerseNumbers = computed(() => {
  return availableVerseNumbers.value.filter(value => {
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

    feedbackSent.value = true
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
    emit('completed')
  } catch {
    // handled in composable
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
</script>

<template>
  <div class="feedback">
    <template v-if="!feedbackSent">
      <div class="feedback-head">
        <p class="feedback-label">Validation rapide</p>
        <p class="feedback-title">Ce résultat est-il correct ?</p>
      </div>

      <div v-if="!showCorrectionForm" class="feedback-actions">
        <button class="feedback-action positive" type="button" :disabled="sending" @click="submitPositiveFeedback">
          <span class="feedback-icon">👍</span>
          <span class="feedback-text">Oui</span>
        </button>

        <button class="feedback-action neutral" type="button" :disabled="sending" @click="openCorrectionForm">
          <span class="feedback-icon">✍️</span>
          <span class="feedback-text">Corriger</span>
        </button>
      </div>

      <div v-else class="correction-panel">
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
            <label for="start-verse">Début</label>
            <select id="start-verse" v-model.number="selectedStartVerse" :disabled="sending || surahsLoading || !availableVerseNumbers.length">
              <option v-for="verseNumber in availableVerseNumbers" :key="`start-${verseNumber}`" :value="verseNumber">
                {{ verseNumber }}
              </option>
            </select>
          </div>

          <div class="field">
            <label for="end-verse">Fin</label>
            <select id="end-verse" v-model.number="selectedEndVerse" :disabled="sending || surahsLoading || !availableVerseNumbers.length">
              <option v-for="verseNumber in availableEndVerseNumbers" :key="`end-${verseNumber}`" :value="verseNumber">
                {{ verseNumber }}
              </option>
            </select>
          </div>
        </div>

        <div class="field">
          <label for="comment">Commentaire</label>
          <textarea
            id="comment"
            v-model="correctionComment"
            rows="3"
            placeholder="Précision utile"
          />
        </div>

        <div class="correction-actions">
          <button class="subtle-btn" type="button" :disabled="sending" @click="showCorrectionForm = false">
            Retour
          </button>

          <button class="submit-btn" type="button" :disabled="!canSubmitCorrection" @click="submitCorrection">
            {{ sending ? 'Envoi…' : (surahsLoading ? 'Chargement…' : 'Envoyer') }}
          </button>
        </div>
      </div>

      <p v-if="formError" class="form-error">
        {{ formError }}
      </p>
    </template>

    <template v-else>
      <div class="feedback-done">
        <span class="done-icon">✓</span>
        <p class="done-text">Merci pour votre retour</p>
      </div>
    </template>
  </div>
</template>

<style scoped>
.feedback {
  width: 100%;
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
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #93c5fd;
}

.feedback-title {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
  color: #e2e8f0;
}

.feedback-actions {
  margin-top: 16px;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.feedback-action {
  min-height: 88px;
  border-radius: 24px;
  border: 1px solid rgba(148, 163, 184, 0.12);
  background: rgba(2, 8, 23, 0.62);
  color: #fff;
  cursor: pointer;
  display: grid;
  place-items: center;
  gap: 6px;
  transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
}

.feedback-action:hover {
  transform: translateY(-1px);
}

.feedback-action.positive {
  background: rgba(4, 120, 87, 0.14);
  border-color: rgba(16, 185, 129, 0.22);
}

.feedback-action.neutral {
  background: rgba(30, 64, 175, 0.14);
  border-color: rgba(96, 165, 250, 0.22);
}

.feedback-icon {
  font-size: 28px;
}

.feedback-text {
  font-size: 15px;
  font-weight: 700;
}

.correction-panel {
  margin-top: 18px;
  display: grid;
  gap: 12px;
  text-align: left;
}

.field-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.field {
  display: grid;
  gap: 7px;
}

label {
  font-size: 13px;
  color: #cbd5e1;
}

select,
textarea {
  width: 100%;
  border: 1px solid rgba(148, 163, 184, 0.12);
  border-radius: 16px;
  padding: 12px 14px;
  background: rgba(2, 6, 23, 0.72);
  color: #fff;
  font: inherit;
}

textarea {
  resize: vertical;
}

.correction-actions {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
  flex-wrap: wrap;
}

.subtle-btn,
.submit-btn {
  border: none;
  border-radius: 999px;
  padding: 12px 16px;
  font-weight: 700;
  cursor: pointer;
}

.subtle-btn {
  background: rgba(255, 255, 255, 0.06);
  color: #e2e8f0;
}

.submit-btn {
  background: linear-gradient(135deg, #3b82f6, #1d4ed8);
  color: #fff;
}

.form-error {
  margin: 14px 0 0;
  color: #fca5a5;
  text-align: center;
}

.feedback-done {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(34, 197, 94, 0.12);
  color: #bbf7d0;
}

.done-icon {
  display: inline-grid;
  place-items: center;
  width: 22px;
  height: 22px;
  border-radius: 999px;
  background: rgba(34, 197, 94, 0.18);
  font-weight: 800;
}

.done-text {
  margin: 0;
  font-weight: 700;
}

@media (max-width: 768px) {
  .feedback-actions,
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
