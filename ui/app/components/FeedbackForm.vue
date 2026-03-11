<script setup lang="ts">
// ROLE
// ----
// Formulaire de feedback utilisateur pour confirmer
// ou corriger le verset détecté.

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

function submitPositiveFeedback() {
  isCorrect.value = true
  feedbackSent.value = true

  console.log('Feedback positif', {
    is_correct: true,
    detected_verse: props.result.verse,
    transcription_text: props.result.transcription_text,
  })
}

function submitNegativeFeedback() {
  isCorrect.value = false
}

function submitCorrection() {
  feedbackSent.value = true

  console.log('Feedback négatif avec correction', {
    is_correct: false,
    transcription_text: props.result.transcription_text,
    detected_verse: props.result.verse,
    corrected_sourate: correctedSourate.value,
    corrected_start_verse: correctedStartVerse.value,
    corrected_end_verse: correctedEndVerse.value,
    correction_comment: correctionComment.value,
  })
}
</script>

<template>
  <div class="feedback-card">
    <template v-if="!feedbackSent">
      <p class="eyebrow">Feedback</p>
      <h3>Le verset détecté est-il correct&nbsp;?</h3>

      <div class="actions">
        <button type="button" class="action-button success" @click="submitPositiveFeedback">
          Oui, c’est correct
        </button>

        <button type="button" class="action-button danger" @click="submitNegativeFeedback">
          Non, il faut corriger
        </button>
      </div>

      <div v-if="isCorrect === false" class="correction-form">
        <div class="field">
          <label for="sourate">Sourate correcte</label>
          <input id="sourate" v-model="correctedSourate" type="text" placeholder="Ex: Al-Baqara" />
        </div>

        <div class="field-row">
          <div class="field">
            <label for="start-verse">Verset début</label>
            <input id="start-verse" v-model="correctedStartVerse" type="text" placeholder="Ex: 1" />
          </div>

          <div class="field">
            <label for="end-verse">Verset fin</label>
            <input id="end-verse" v-model="correctedEndVerse" type="text" placeholder="Ex: 3" />
          </div>
        </div>

        <div class="field">
          <label for="comment">Commentaire optionnel</label>
          <textarea id="comment" v-model="correctionComment" rows="4"
            placeholder="Précision utile pour corriger plus tard" />
        </div>

        <button type="button" class="submit-button" @click="submitCorrection">
          Envoyer la correction
        </button>
      </div>
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

@media (max-width: 640px) {
  .field-row {
    grid-template-columns: 1fr;
  }
}
</style>
