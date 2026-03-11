<script setup lang="ts">
// ROLE
// ----
// Formulaire de feedback utilisateur pour confirmer
// ou corriger le verset détecté, puis envoyer le tout au backend.

import { useFeedback } from '~/composables/useFeedback'
import type { RecognizeResponse } from '~/composables/useRecognition'

const props = defineProps<{
  result: RecognizeResponse
}>()

const feedbackSent = ref(false)
const isCorrect = ref<boolean | null>(null)

const correctedSourate = ref('')
const correctedStartVerse = ref('')
const correctedEndVerse = ref('')
const correctionComment = ref('')

const { sending, error, sendFeedback } = useFeedback()

async function submitPositiveFeedback() {
  try {
    isCorrect.value = true

    await sendFeedback({
      is_correct: true,
      transcription_text: props.result.transcription_text,
      detected_verse: props.result.verse,
      correction: null,
      comment: null,
    })

    feedbackSent.value = true
  } catch {
    // erreur déjà gérée dans le composable
  }
}

function submitNegativeFeedback() {
  isCorrect.value = false
}

async function submitCorrection() {
  const startVerse = Number(correctedStartVerse.value)
  const endVerse = Number(correctedEndVerse.value)

  if (!correctedSourate.value.trim() || !Number.isInteger(startVerse) || !Number.isInteger(endVerse)) {
    return
  }

  try {
    await sendFeedback({
      is_correct: false,
      transcription_text: props.result.transcription_text,
      detected_verse: props.result.verse,
      correction: {
        sourate_name: correctedSourate.value.trim(),
        start_verse: startVerse,
        end_verse: endVerse,
      },
      comment: correctionComment.value.trim() || null,
    })

    feedbackSent.value = true
  } catch {
    // erreur déjà gérée dans le composable
  }
}
</script>

<template>
  <div class="feedback-card">
    <template v-if="!feedbackSent">
      <p class="eyebrow">Feedback</p>
      <h3>Le verset détecté est-il correct&nbsp;?</h3>

      <div class="actions">
        <button type="button" class="action-button success" :disabled="sending" @click="submitPositiveFeedback">
          Oui, c’est correct
        </button>

        <button type="button" class="action-button danger" :disabled="sending" @click="submitNegativeFeedback">
          Non, il faut corriger
        </button>
      </div>

      <div v-if="isCorrect === false" class="correction-form">
        <div class="field">
          <label for="sourate">Sourate correcte</label>
          <input id="sourate" v-model="correctedSourate" type="text" placeholder="Ex: Al-Fatiha" />
        </div>

        <div class="field-row">
          <div class="field">
            <label for="start-verse">Verset début</label>
            <input id="start-verse" v-model="correctedStartVerse" type="number" min="1" placeholder="Ex: 1" />
          </div>

          <div class="field">
            <label for="end-verse">Verset fin</label>
            <input id="end-verse" v-model="correctedEndVerse" type="number" min="1" placeholder="Ex: 7" />
          </div>
        </div>

        <div class="field">
          <label for="comment">Commentaire optionnel</label>
          <textarea id="comment" v-model="correctionComment" rows="4"
            placeholder="Précision utile pour corriger plus tard" />
        </div>

        <button type="button" class="submit-button" :disabled="sending" @click="submitCorrection">
          {{ sending ? 'Envoi...' : 'Envoyer la correction' }}
        </button>
      </div>

      <p v-if="error" class="error-message">
        {{ error }}
      </p>
    </template>

    <template v-else>
      <p class="eyebrow">Feedback</p>
      <h3>Merci pour votre retour</h3>
      <p class="thank-you">
        Votre validation aidera à améliorer Sawt AI.
      </p>
    </template>
  </div>
</template>

<style scoped>
.feedback-card {
  margin-top: 20px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 24px;
  padding: 22px;
  background: rgba(15, 23, 42, 0.6);
  backdrop-filter: blur(12px);
}

.eyebrow {
  margin: 0 0 6px;
  font-size: 13px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #93c5fd;
}

h3 {
  margin: 0;
  font-size: 22px;
  line-height: 1.3;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 18px;
}

.action-button,
.submit-button {
  border: none;
  border-radius: 999px;
  padding: 12px 18px;
  font-weight: 700;
  cursor: pointer;
  transition: transform 0.2s ease, opacity 0.2s ease;
}

.action-button:hover,
.submit-button:hover {
  transform: translateY(-1px);
}

.action-button:disabled,
.submit-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.success {
  background: #f8fafc;
  color: #08111f;
}

.danger {
  background: rgba(239, 68, 68, 0.14);
  color: #fecaca;
  border: 1px solid rgba(239, 68, 68, 0.22);
}

.correction-form {
  margin-top: 20px;
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
textarea {
  width: 100%;
  box-sizing: border-box;
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(2, 6, 23, 0.45);
  color: #fff;
  font: inherit;
  transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

input:focus,
textarea:focus {
  outline: none;
  border-color: rgba(96, 165, 250, 0.45);
  box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.12);
  background: rgba(2, 6, 23, 0.62);
}

textarea {
  resize: vertical;
}

.submit-button {
  width: fit-content;
  background: linear-gradient(135deg, #2563eb, #1d4ed8);
  color: white;
}

.thank-you {
  margin-top: 12px;
  color: #cbd5e1;
  line-height: 1.6;
}

.error-message {
  margin-top: 14px;
  color: #fecaca;
}

@media (max-width: 640px) {
  .field-row {
    grid-template-columns: 1fr;
  }
}
</style>
