<script setup lang="ts">
import { FlaskConical, Upload } from '@lucide/vue'
import { computed, ref } from 'vue'

import RecognitionActionButton from '~/components/RecognitionActionButton.vue'
import MotionReveal from '~/components/MotionReveal.vue'

const props = withDefaults(
  defineProps<{
    uploadError?: string | null
    micError?: string | null
    isRecording?: boolean
    isFinalizingRecording?: boolean
    recordingSeconds?: number
    maxRecordingSeconds?: number | null
    audioLevel?: number
    uploadAccept?: string
    uploadHint?: string | null
    detectImam?: boolean
    imamDetectionAvailable?: boolean
    imamDetectionMessage?: string | null
  }>(),
  {
    imamDetectionAvailable: true,
  },
)

const emit = defineEmits<{
  'micro-click': []
  'select-file': [file: File]
  'update:detect-imam': [value: boolean]
}>()

const fileInput = ref<HTMLInputElement | null>(null)

const title = computed(() => {
  if (props.isFinalizingRecording) return 'Préparation de l’audio'
  return props.isRecording ? 'Je vous écoute' : 'Récitez un passage du Coran'
})

const recordingTime = computed(() => `${props.recordingSeconds ?? 0}s`)

const recordingProgressPercent = computed(() => {
  const maxSeconds = props.maxRecordingSeconds

  if (!props.isRecording || !maxSeconds || maxSeconds <= 0) {
    return 0
  }

  const seconds = Math.max(0, props.recordingSeconds ?? 0)
  return Math.min(100, (seconds / maxSeconds) * 100)
})

const recordingProgressLabel = computed(() => {
  const maxSeconds = props.maxRecordingSeconds

  if (!props.isRecording || !maxSeconds || maxSeconds <= 0) {
    return ''
  }

  return `${props.recordingSeconds ?? 0}s / ${maxSeconds}s`
})

const recordingError = computed(() => {
  if (!props.uploadError?.toLowerCase().includes('enregistrement')) {
    return null
  }

  return props.uploadError
})

const fileError = computed(() => {
  if (recordingError.value) return null
  return props.uploadError ?? null
})

const micErrorHint = computed(() => {
  if (!props.micError) return ''

  const normalizedError = props.micError.toLowerCase()

  if (normalizedError.includes('https')) return 'Ouvrez Sawt AI en HTTPS.'
  if (normalizedError.includes('navigateur')) return 'Essayez un navigateur récent.'
  return 'Autorisez le micro dans votre navigateur.'
})

const recordingErrorHint = computed(() => {
  if (!recordingError.value) return ''
  return 'Relancez une prise courte.'
})

const fileErrorHint = computed(() => {
  if (!fileError.value) return ''

  const normalizedError = fileError.value.toLowerCase()

  if (normalizedError.includes('format')) return 'Utilisez wav, mp3, m4a, ogg ou webm.'
  if (normalizedError.includes('volumineux')) return 'Choisissez un extrait plus léger.'
  if (normalizedError.includes('trop long')) return 'Gardez un extrait plus court.'
  return 'Essayez un autre fichier audio.'
})

function onMicroButtonClick() {
  emit('micro-click')
}

function onDetectImamChange(event: Event) {
  if (props.imamDetectionAvailable === false) return
  const input = event.target as HTMLInputElement
  emit('update:detect-imam', input.checked)
}

function openFilePicker() {
  fileInput.value?.click()
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
  <section
    class="screen idle-screen"
    :class="{ 'is-recording': isRecording }"
    aria-labelledby="recognition-title"
  >

    <div class="hero-shell">
      <MotionReveal class="hero-copy">
        <h1 id="recognition-title" class="main-title" aria-live="polite">{{ title }}</h1>

        <div
          v-if="isRecording"
          class="recording-time"
          role="timer"
          :aria-label="`Durée de l’enregistrement : ${recordingTime}`"
        >
          <span class="recording-dot" aria-hidden="true" />
          {{ maxRecordingSeconds ? recordingProgressLabel : recordingTime }}
        </div>

        <div v-if="isRecording && maxRecordingSeconds" class="recording-progress">
          <div
            class="recording-progress-track"
            role="progressbar"
            aria-label="Progression de l’enregistrement"
            :aria-valuenow="recordingSeconds ?? 0"
            aria-valuemin="0"
            :aria-valuemax="maxRecordingSeconds"
          >
            <span
              class="recording-progress-fill"
              :style="{ width: `${recordingProgressPercent}%` }"
            />
          </div>
        </div>
      </MotionReveal>

      <MotionReveal class="hero-action" :delay="0.08" :distance="12">
        <RecognitionActionButton
          :is-recording="isRecording"
          :loading="isFinalizingRecording"
          :show-label="!isFinalizingRecording"
          :disabled="isFinalizingRecording"
          loading-label="Préparation de l’audio"
          :audio-level="audioLevel"
          @click="onMicroButtonClick"
        />

        <div v-if="micError || recordingError" class="status-message is-error" role="alert">
          <p class="status-title">{{ micError ?? recordingError }}</p>
          <p class="status-hint">{{ micError ? micErrorHint : recordingErrorHint }}</p>
        </div>
      </MotionReveal>

      <MotionReveal v-if="!isRecording && !isFinalizingRecording" class="secondary-actions" :delay="0.16">
        <button class="file-button" type="button" @click="openFilePicker">
          <Upload :size="16" aria-hidden="true" />
          Importer un fichier audio
        </button>

        <div v-if="fileError" class="status-message is-error" role="alert">
          <p class="status-title">{{ fileError }}</p>
          <p class="status-hint">{{ fileErrorHint }}</p>
        </div>

        <details class="options-shell">
          <summary>Options</summary>

          <div class="option-content">
            <label
              class="imam-toggle"
              :class="{ 'is-disabled': imamDetectionAvailable === false }"
              :title="imamDetectionAvailable === false ? (imamDetectionMessage ?? undefined) : undefined"
            >
              <input
                type="checkbox"
                class="imam-toggle-checkbox"
                :checked="detectImam"
                :disabled="imamDetectionAvailable === false"
                @change="onDetectImamChange"
              >
              <span class="imam-toggle-text">Reconnaître l’imam</span>
              <span class="imam-beta-badge">
                <FlaskConical class="imam-beta-icon" :stroke-width="1.9" aria-hidden="true" />
                Bêta
              </span>
            </label>

            <p
              v-if="imamDetectionAvailable === false"
              class="imam-toggle-hint"
              :class="{ 'is-unavailable': imamDetectionAvailable === false }"
            >
              {{ imamDetectionMessage ?? 'La reconnaissance de l’imam est temporairement indisponible.' }}
            </p>
            <p class="upload-hint">
              {{ uploadHint ?? 'wav, mp3, m4a, ogg ou webm · 12 Mo et 90 sec maximum' }}
            </p>
          </div>
        </details>
      </MotionReveal>
    </div>

    <input
      ref="fileInput"
      class="hidden-input"
      type="file"
      :accept="uploadAccept ?? 'audio/*'"
      tabindex="-1"
      @change="onFileChange"
    >
  </section>
</template>

<style scoped>
.screen {
  position: relative;
  z-index: 1;
  flex: 1;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
}

.idle-screen {
  width: min(100%, 860px);
  margin: 0 auto;
  padding: 26px 20px 34px;
}

.hero-shell {
  width: 100%;
  min-width: 0;
  flex: 1;
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  justify-items: center;
  align-content: center;
  gap: 28px;
  padding: 32px 0;
  text-align: center;
}

.hero-copy {
  width: 100%;
  min-width: 0;
  display: grid;
  justify-items: center;
}

.recording-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: #60a5fa;
  box-shadow: 0 0 0 5px rgba(96, 165, 250, 0.1);
}

.main-title {
  width: 100%;
  margin: 0;
  max-width: 440px;
  font-size: clamp(28px, 4vw, 40px);
  line-height: 1.2;
  font-weight: 800;
  letter-spacing: -0.052em;
  text-wrap: balance;
}

.recording-time {
  margin-top: 20px;
  display: inline-flex;
  align-items: center;
  gap: 12px;
  color: #f8fafc;
  font-variant-numeric: tabular-nums;
  font-size: 22px;
  line-height: 1;
  font-weight: 750;
  letter-spacing: -0.03em;
}

.recording-progress {
  width: min(76vw, 280px);
  margin-top: 14px;
  display: grid;
  gap: 7px;
}

.recording-progress-track {
  width: 100%;
  height: 4px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.18);
}

.recording-progress-fill {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: #60a5fa;
  transition: width 0.24s linear;
}

.hero-action {
  width: min(100%, 420px);
  display: grid;
  justify-items: center;
  gap: 8px;
}

.secondary-actions {
  width: min(100%, 420px);
  min-width: 0;
  margin-top: 6px;
  display: grid;
  justify-items: center;
  gap: 9px;
}

.file-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 44px;
  padding: 0 16px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 999px;
  background: transparent;
  color: #dbeafe;
  font: inherit;
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  transition:
    color 180ms ease,
    border-color 180ms ease,
    background 180ms ease;
}

.file-button:hover {
  border-color: rgba(147, 197, 253, 0.36);
  background: rgba(30, 41, 59, 0.58);
  color: #fff;
}

.file-button:focus-visible,
.options-shell summary:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 3px;
}

.upload-hint {
  max-width: 100%;
  margin: 14px 0 0;
  color: #a1b0c5;
  font-size: 12px;
  line-height: 1.45;
  overflow-wrap: anywhere;
}

.options-shell {
  width: 100%;
  margin-top: 4px;
  color: #aebdd0;
}

.options-shell summary {
  width: fit-content;
  margin: 0 auto;
  min-height: 44px;
  padding: 12px 8px;
  color: #91a2b8;
  font-size: 13px;
  cursor: pointer;
}

.option-content {
  margin-top: 10px;
  padding: 14px 16px;
  border: 1px solid rgba(148, 163, 184, 0.12);
  border-radius: 18px;
  background: rgba(8, 17, 32, 0.46);
}

.imam-toggle {
  display: inline-flex;
  align-items: center;
  gap: 9px;
  min-height: 36px;
  cursor: pointer;
  user-select: none;
}

.imam-toggle.is-disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.imam-toggle-checkbox {
  width: 18px;
  height: 18px;
  accent-color: #3b82f6;
  cursor: pointer;
}

.imam-toggle-checkbox:disabled {
  cursor: not-allowed;
}

.imam-toggle-text {
  color: #dce6f2;
  font-size: 14px;
  font-weight: 700;
}

.imam-beta-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  min-height: 21px;
  padding: 0 7px;
  border: 1px solid rgba(147, 197, 253, 0.22);
  border-radius: 999px;
  background: rgba(59, 130, 246, 0.1);
  color: #bfdbfe;
  font-size: 9px;
  font-weight: 800;
  text-transform: uppercase;
}

.imam-beta-icon {
  width: 11px;
  height: 11px;
}

.imam-toggle-hint {
  margin: 7px auto 0;
  max-width: 330px;
  color: #7f91a8;
  font-size: 12px;
  line-height: 1.5;
}

.imam-toggle-hint.is-unavailable {
  color: #fbbf24;
}

.status-message {
  width: min(100%, 380px);
  margin-top: 4px;
  padding: 12px 14px;
  border-radius: 14px;
  text-align: left;
  font-size: 13px;
  line-height: 1.5;
}

.status-message.is-error {
  border: 1px solid rgba(248, 113, 113, 0.2);
  background: rgba(127, 29, 29, 0.2);
  color: #fecaca;
}

.status-title,
.status-hint {
  margin: 0;
}

.status-title {
  font-weight: 750;
}

.status-hint {
  margin-top: 3px;
  color: #fda4af;
}

.hidden-input {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

@media (max-width: 640px) {
  .idle-screen {
    padding: 24px 16px;
  }

  .hero-shell {
    gap: 24px;
    padding: 24px 0;
  }

  .main-title {
    max-width: 320px;
    font-size: clamp(28px, 7vw, 34px);
  }
}

@media (max-height: 600px) {
  .hero-shell {
    gap: 14px;
    padding: 12px 0;
  }
}

@media (prefers-reduced-motion: reduce) {
  .recording-progress-fill {
    transition: none;
  }
}
</style>
