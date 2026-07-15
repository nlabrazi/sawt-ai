<script setup lang="ts">
import FlaskConical from '@lucide/vue/dist/esm/icons/flask-conical.mjs'
import { computed, ref } from 'vue'

import RecognitionActionButton from '~/components/RecognitionActionButton.vue'

const props = withDefaults(
  defineProps<{
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
  return props.isRecording ? 'Je vous écoute' : 'Récitez un passage du Coran'
})

const subtitle = computed(() => {
  return props.isRecording
    ? 'Récitez naturellement, dans un environnement aussi calme que possible.'
    : 'Sawt AI vous propose la sourate et les versets correspondants.'
})

const recognitionActionHint = computed(() => {
  return props.isRecording
    ? 'Appuyez à nouveau pour arrêter et analyser.'
    : 'Appuyez pour commencer, puis une seconde fois pour arrêter et analyser.'
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
    <header class="brand" aria-label="Sawt AI">
      <span class="brand-name">Sawt</span>
      <span class="brand-mark">AI</span>
    </header>

    <div class="hero-shell">
      <div class="hero-copy">
        <p class="state-label">
          <span v-if="isRecording" class="recording-dot" aria-hidden="true" />
          {{ isRecording ? 'Enregistrement en cours' : 'Reconnaissance coranique' }}
        </p>

        <h1 id="recognition-title" class="main-title">{{ title }}</h1>

        <p id="recognition-guidance" class="main-subtitle">
          {{ subtitle }}
        </p>

        <div
          v-if="isRecording"
          class="recording-time"
          role="timer"
          :aria-label="`Durée de l’enregistrement : ${recordingTime}`"
        >
          {{ recordingTime }}
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

          <p class="recording-progress-label">{{ recordingProgressLabel }}</p>
        </div>
      </div>

      <div class="hero-action">
        <RecognitionActionButton
          :is-recording="isRecording"
          :audio-level="audioLevel"
          aria-describedby="recognition-guidance recognition-action-hint"
          @click="onMicroButtonClick"
        />

        <p id="recognition-action-hint" class="recognition-action-hint">
          {{ recognitionActionHint }}
        </p>

        <div v-if="micError || recordingError" class="status-message is-error" role="alert">
          <p class="status-title">{{ micError ?? recordingError }}</p>
          <p class="status-hint">{{ micError ? micErrorHint : recordingErrorHint }}</p>
        </div>
      </div>

      <div v-if="!isRecording" class="secondary-actions">
        <button class="file-button" type="button" @click="openFilePicker">
          Importer un fichier audio
        </button>

        <p class="upload-hint">
          {{ uploadHint ?? 'wav, mp3, m4a, ogg ou webm · 12 Mo et 90 sec maximum' }}
        </p>

        <div v-if="fileError" class="status-message is-error" role="alert">
          <p class="status-title">{{ fileError }}</p>
          <p class="status-hint">{{ fileErrorHint }}</p>
        </div>

        <details class="options-shell">
          <summary>Options de reconnaissance</summary>

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
              class="imam-toggle-hint"
              :class="{ 'is-unavailable': imamDetectionAvailable === false }"
            >
              {{ imamDetectionAvailable === false
                ? (imamDetectionMessage ?? 'La reconnaissance de l’imam est temporairement indisponible.')
                : 'Analyse du réciteur en plus du passage proposé.' }}
            </p>
          </div>
        </details>
      </div>
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

.brand {
  display: inline-flex;
  align-items: baseline;
  justify-content: center;
  gap: 5px;
  color: #f8fafc;
  font-size: 17px;
  line-height: 1;
  font-weight: 800;
  letter-spacing: -0.02em;
}

.brand-mark {
  color: #60a5fa;
  font-size: 12px;
  letter-spacing: 0.04em;
}

.hero-shell {
  width: 100%;
  min-width: 0;
  flex: 1;
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  justify-items: center;
  align-content: center;
  gap: 14px;
  padding: 34px 0 24px;
  text-align: center;
}

.hero-copy {
  width: 100%;
  min-width: 0;
  display: grid;
  justify-items: center;
}

.state-label {
  margin: 0;
  min-height: 22px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: #93c5fd;
  font-size: 12px;
  line-height: 1.4;
  font-weight: 750;
  letter-spacing: 0.12em;
  text-transform: uppercase;
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
  margin: 14px 0 0;
  max-width: 720px;
  font-size: clamp(38px, 6vw, 60px);
  line-height: 1.02;
  font-weight: 800;
  letter-spacing: -0.052em;
  text-wrap: balance;
}

.main-subtitle {
  margin: 16px 0 0;
  max-width: 520px;
  color: #bac7d8;
  font-size: clamp(17px, 2vw, 19px);
  line-height: 1.55;
  text-wrap: balance;
}

.recording-time {
  margin-top: 18px;
  color: #f8fafc;
  font-variant-numeric: tabular-nums;
  font-size: 34px;
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

.recording-progress-label {
  margin: 0;
  color: #7f91a8;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.hero-action {
  width: min(100%, 420px);
  display: grid;
  justify-items: center;
  gap: 8px;
}

.recognition-action-hint {
  margin: -2px 0 0;
  max-width: 390px;
  color: #91a2b8;
  font-size: 14px;
  line-height: 1.5;
  text-wrap: balance;
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
  min-height: 42px;
  padding: 0 16px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.38);
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
  margin: 0;
  color: #708198;
  font-size: 12px;
  line-height: 1.45;
  overflow-wrap: anywhere;
}

.options-shell {
  width: 100%;
  margin-top: 4px;
  border-top: 1px solid rgba(148, 163, 184, 0.11);
  color: #aebdd0;
}

.options-shell summary {
  width: fit-content;
  margin: 0 auto;
  padding: 13px 8px 4px;
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
    padding: 20px 16px 26px;
  }

  .hero-shell {
    gap: 10px;
    padding: 28px 0 18px;
  }

  .main-title {
    margin-top: 12px;
    font-size: clamp(34px, 10vw, 46px);
  }

  .main-subtitle {
    margin-top: 13px;
    font-size: 16px;
  }

  .recording-time {
    margin-top: 15px;
    font-size: 30px;
  }

  .recognition-action-hint {
    max-width: 320px;
    font-size: 13px;
  }
}

@media (max-height: 760px) and (min-width: 641px) {
  .hero-shell {
    align-content: start;
    padding-top: 26px;
  }

  .main-title {
    font-size: 46px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .recording-progress-fill {
    transition: none;
  }
}
</style>
