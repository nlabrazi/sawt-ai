<script setup lang="ts">
import RecognitionActionButton from '~/components/RecognitionActionButton.vue'

const props = defineProps<{
  uploadError?: string | null
  micError?: string | null
  isRecording?: boolean
  recordingSeconds?: number
  maxRecordingSeconds?: number | null
  audioLevel?: number
  uploadAccept?: string
  uploadHint?: string | null
  detectImam?: boolean
  imamDetectionAvailable?: boolean
  imamDetectionMessage?: string | null
}>()

const emit = defineEmits<{
  'micro-click': []
  'select-file': [file: File]
  'update:detect-imam': [value: boolean]
}>()

const fileInputId = 'recognition-audio-file-input'

function onMicroButtonClick() {
  emit('micro-click')
}

function onDetectImamChange(event: Event) {
  if (props.imamDetectionAvailable === false) return

  const input = event.target as HTMLInputElement
  emit('update:detect-imam', input.checked)
}

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]

  if (!file) return

  emit('select-file', file)

  input.value = ''
}
</script>

<template>
  <section class="screen idle-screen">
    <div class="brand">
      <p class="brand-kicker">Sawt AI</p>
    </div>

    <div class="center-stack">
      <RecognitionActionButton :is-recording="isRecording" :audio-level="audioLevel" @click="onMicroButtonClick" />

      <h1 class="main-title">Touchez pour réciter</h1>
      <p class="main-subtitle">
        Détection du verset en quelques secondes
      </p>

      <div class="secondary-action">
        <span class="secondary-text">Vous préférez importer un fichier audio ?</span>
        <label class="secondary-link" :for="fileInputId">
          Choisir un fichier
        </label>

        <label
          class="imam-toggle"
          :class="{ 'is-disabled': imamDetectionAvailable === false }"
          :title="imamDetectionAvailable === false ? (imamDetectionMessage ?? undefined) : undefined"
        >
          <input
            class="imam-toggle-checkbox"
            type="checkbox"
            :checked="detectImam"
            :disabled="imamDetectionAvailable === false"
            @change="onDetectImamChange"
          >
          <span class="imam-toggle-text">
            Reconnaître l’imam (récitateurs connus)
          </span>
        </label>

        <p class="imam-toggle-hint" :class="{ 'is-unavailable': imamDetectionAvailable === false }">
          {{ imamDetectionAvailable === false
            ? (imamDetectionMessage ?? 'La reconnaissance de l’imam est temporairement indisponible.')
            : 'Analyse du réciteur en plus du verset' }}
        </p>

        <p v-if="uploadHint" class="upload-hint">
          {{ uploadHint }}
        </p>

        <p v-if="uploadError" class="upload-error">
          {{ uploadError }}
        </p>

        <p v-if="micError" class="upload-error">
          {{ micError }}
        </p>
      </div>

      <input :id="fileInputId" class="hidden-input" type="file" :accept="uploadAccept ?? 'audio/*'" @change="onFileChange" />
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

.idle-screen {
  display: flex;
  flex-direction: column;
}

.brand {
  display: flex;
  justify-content: center;
}

.brand-kicker {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #93c5fd;
}

.center-stack {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
}

.main-title {
  margin: 28px 0 0;
  font-size: 34px;
  line-height: 1.08;
  font-weight: 800;
  letter-spacing: -0.03em;
}

.main-subtitle {
  margin: 12px 0 0;
  max-width: 420px;
  font-size: 16px;
  line-height: 1.6;
  color: #cbd5e1;
}

.secondary-action {
  margin-top: 30px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.secondary-text {
  color: #94a3b8;
  font-size: 14px;
}

.imam-toggle {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 2px;
  padding: 4px 8px;
  border-radius: 999px;
  color: #94a3b8;
  cursor: pointer;
  transition: background 0.2s ease, color 0.2s ease;
}

.imam-toggle:hover {
  background: rgba(147, 197, 253, 0.08);
}

.imam-toggle.is-disabled {
  color: #64748b;
  cursor: not-allowed;
}

.imam-toggle.is-disabled:hover {
  background: transparent;
}

.imam-toggle:hover .imam-toggle-text {
  color: #bfdbfe;
}

.imam-toggle.is-disabled:hover .imam-toggle-text {
  color: #64748b;
}

.imam-toggle-checkbox {
  width: 15px;
  height: 15px;
  accent-color: #60a5fa;
  cursor: pointer;
  opacity: 0.9;
}

.imam-toggle-checkbox:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.imam-toggle-text {
  font-size: 13px;
  font-weight: 500;
  color: #94a3b8;
}

.imam-toggle-hint {
  margin: 0;
  max-width: 360px;
  font-size: 12px;
  line-height: 1.45;
  color: #64748b;
}

.imam-toggle-hint.is-unavailable {
  color: #fca5a5;
  font-weight: 600;
}

.secondary-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: none;
  background: transparent;
  color: #93c5fd;
  font-weight: 700;
  font-size: 15px;
  cursor: pointer;
  transition: opacity 0.2s ease, transform 0.2s ease, color 0.2s ease;
}

.secondary-link:hover {
  opacity: 0.9;
  transform: translateY(-1px);
  color: #bfdbfe;
}

.upload-hint {
  margin: 2px 0 0;
  font-size: 13px;
  color: #64748b;
}

.upload-error {
  margin: 6px 0 0;
  color: #fca5a5;
  font-size: 14px;
}

.hidden-input {
  position: absolute;
  opacity: 0;
  pointer-events: none;
  width: 0;
  height: 0;
}

@media (max-width: 768px) {
  .screen {
    padding: 18px 14px 28px;
  }

  .main-title {
    font-size: 28px;
  }
}
</style>
