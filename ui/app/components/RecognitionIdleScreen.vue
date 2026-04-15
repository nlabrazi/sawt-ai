<script setup lang="ts">
import { computed } from 'vue'
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

const eyebrow = computed(() => {
  return props.isRecording ? 'Écoute en direct' : 'Sawt AI'
})

const title = computed(() => {
  return props.isRecording ? 'Enregistrement' : 'Touchez pour réciter'
})

const subtitle = computed(() => {
  return props.isRecording
    ? 'Touchez à nouveau pour arrêter et lancer l’analyse.'
    : 'Récitez quelques secondes. Sawt AI reconnaît le passage.'
})

const helperText = computed(() => {
  return props.isRecording ? 'Nous écoutons en direct.' : 'Une seule action, un seul geste.'
})

const recordingTime = computed(() => {
  if (!props.isRecording) return ''
  return `${props.recordingSeconds ?? 0}s`
})

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
  <section class="screen idle-screen" :class="{ 'is-recording': isRecording }">
    <div class="brand">
      <p class="brand-kicker">{{ eyebrow }}</p>
    </div>

    <div class="hero-shell">
      <div class="hero-copy">
        <p v-if="isRecording" class="section-label">Analyse du passage</p>

        <h1 class="main-title">{{ title }}</h1>

        <p class="main-subtitle">
          {{ subtitle }}
        </p>

        <div v-if="isRecording" class="recording-time">
          <span class="pulse-dot" />
          <span>{{ recordingTime }}</span>
        </div>

        <p class="helper-line">
          {{ helperText }}
        </p>
      </div>

      <div class="hero-action">
        <RecognitionActionButton :is-recording="isRecording" :audio-level="audioLevel" @click="onMicroButtonClick" />
      </div>
    </div>

    <div class="secondary-shell">
      <div class="secondary-top">
        <label class="file-button" :for="fileInputId">
          Choisir un fichier
        </label>

        <p class="upload-hint">
          {{ uploadHint ?? 'Formats : wav, mp3, m4a, ogg, webm · max 12 Mo · max 90 sec' }}
        </p>
      </div>

      <div class="secondary-divider" />

      <label class="imam-toggle" :class="{ 'is-disabled': imamDetectionAvailable === false }"
        :title="imamDetectionAvailable === false ? (imamDetectionMessage ?? undefined) : undefined">
        <input class="imam-toggle-checkbox" type="checkbox" :checked="detectImam"
          :disabled="imamDetectionAvailable === false" @change="onDetectImamChange">
        <span class="imam-toggle-box" aria-hidden="true" />
        <span class="imam-toggle-text">Reconnaître l’imam</span>
      </label>

      <p class="imam-toggle-hint" :class="{ 'is-unavailable': imamDetectionAvailable === false }">
        {{ imamDetectionAvailable === false
          ? (imamDetectionMessage ?? 'La reconnaissance de l’imam est temporairement indisponible.')
          : 'Analyse du réciteur en plus du verset détecté.' }}
      </p>

      <p v-if="uploadError" class="status-message is-error">
        {{ uploadError }}
      </p>

      <p v-if="micError" class="status-message is-error">
        {{ micError }}
      </p>
    </div>

    <input :id="fileInputId" class="hidden-input" type="file" :accept="uploadAccept ?? 'audio/*'"
      @change="onFileChange">
  </section>
</template>

<style scoped>
.screen {
  position: relative;
  z-index: 1;
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 24px 16px 36px;
  box-sizing: border-box;
}

.idle-screen {
  max-width: 980px;
  width: 100%;
  margin: 0 auto;
}

.brand {
  display: flex;
  justify-content: center;
}

.brand-kicker,
.section-label {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: #93c5fd;
}

.hero-shell {
  flex: 1;
  display: grid;
  align-content: center;
  justify-items: center;
  gap: 28px;
  padding: 28px 0 20px;
  text-align: center;
}

.hero-copy {
  display: grid;
  justify-items: center;
}

.main-title {
  margin: 14px 0 0;
  max-width: 9ch;
  font-size: clamp(48px, 8vw, 82px);
  line-height: 0.94;
  font-weight: 800;
  letter-spacing: -0.055em;
  text-wrap: balance;
}

.main-subtitle {
  margin: 18px 0 0;
  max-width: 560px;
  font-size: 20px;
  line-height: 1.55;
  color: #dbe4f0;
  text-wrap: balance;
}

.recording-time {
  margin-top: 20px;
  display: inline-flex;
  align-items: center;
  gap: 12px;
  min-height: 46px;
  padding: 0 16px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.34);
  border: 1px solid rgba(148, 163, 184, 0.12);
  font-size: 28px;
  font-weight: 700;
  color: #dbeafe;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.pulse-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: #93c5fd;
  box-shadow: 0 0 0 0 rgba(147, 197, 253, 0.54);
  animation: pulseDot 1.4s ease-in-out infinite;
}

.helper-line {
  margin: 14px 0 0;
  font-size: 14px;
  color: #8ba0bb;
  letter-spacing: 0.01em;
}

.hero-action {
  display: grid;
  place-items: center;
  min-height: 220px;
}

.secondary-shell {
  width: min(100%, 440px);
  margin: 0 auto;
  padding: 24px;
  border-radius: 30px;
  border: 1px solid rgba(148, 163, 184, 0.12);
  background:
    linear-gradient(180deg, rgba(9, 18, 34, 0.76) 0%, rgba(7, 15, 29, 0.66) 100%);
  backdrop-filter: blur(14px);
  box-shadow:
    0 24px 70px rgba(2, 6, 23, 0.16),
    inset 0 1px 0 rgba(255, 255, 255, 0.04);
  text-align: center;
}

.secondary-top {
  display: grid;
  gap: 14px;
  justify-items: center;
}

.file-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 220px;
  min-height: 56px;
  padding: 0 24px;
  border-radius: 999px;
  border: 1px solid rgba(96, 165, 250, 0.18);
  background:
    linear-gradient(180deg, rgba(30, 64, 175, 0.3) 0%, rgba(30, 41, 59, 0.42) 100%);
  color: #eff6ff;
  font-size: 18px;
  font-weight: 700;
  cursor: pointer;
  transition:
    transform 0.22s ease,
    border-color 0.22s ease,
    background 0.22s ease,
    box-shadow 0.22s ease;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.14);
}

.file-button:hover {
  transform: translateY(-1px);
  border-color: rgba(147, 197, 253, 0.34);
  background:
    linear-gradient(180deg, rgba(37, 99, 235, 0.36) 0%, rgba(30, 41, 59, 0.5) 100%);
}

.upload-hint {
  margin: 0;
  max-width: 320px;
  font-size: 14px;
  line-height: 1.6;
  color: #8ea1b9;
}

.secondary-divider {
  width: 100%;
  height: 1px;
  margin: 18px 0;
  background: linear-gradient(90deg, transparent 0%, rgba(148, 163, 184, 0.14) 15%, rgba(148, 163, 184, 0.14) 85%, transparent 100%);
}

.imam-toggle {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 12px;
  min-height: 48px;
  margin: 0 auto;
  cursor: pointer;
  user-select: none;
}

.imam-toggle.is-disabled {
  cursor: not-allowed;
  opacity: 0.62;
}

.imam-toggle-checkbox {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}

.imam-toggle-box {
  width: 22px;
  height: 22px;
  border-radius: 7px;
  border: 1px solid rgba(148, 163, 184, 0.2);
  background: rgba(15, 23, 42, 0.82);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
  transition: all 0.2s ease;
}

.imam-toggle-checkbox:checked+.imam-toggle-box {
  border-color: rgba(96, 165, 250, 0.5);
  background:
    linear-gradient(180deg, rgba(96, 165, 250, 0.92) 0%, rgba(59, 130, 246, 1) 100%);
  box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.12);
}

.imam-toggle-checkbox:checked+.imam-toggle-box::after {
  content: '';
  position: absolute;
  width: 6px;
  height: 10px;
  border-right: 2px solid #eff6ff;
  border-bottom: 2px solid #eff6ff;
  transform: translate(7px, 3px) rotate(45deg);
}

.imam-toggle-text {
  font-size: 16px;
  font-weight: 700;
  color: #e6edf7;
}

.imam-toggle-hint {
  margin: 10px auto 0;
  max-width: 330px;
  font-size: 14px;
  line-height: 1.55;
  color: #8ea1b9;
}

.imam-toggle-hint.is-unavailable {
  color: #fbbf24;
}

.status-message {
  margin: 14px 0 0;
  border-radius: 18px;
  padding: 12px 14px;
  font-size: 14px;
  line-height: 1.55;
}

.status-message.is-error {
  background: rgba(127, 29, 29, 0.22);
  border: 1px solid rgba(248, 113, 113, 0.16);
  color: #fecaca;
}

.hidden-input {
  position: absolute;
  opacity: 0;
  pointer-events: none;
  inset: auto;
  width: 1px;
  height: 1px;
}

@keyframes pulseDot {

  0%,
  100% {
    box-shadow: 0 0 0 0 rgba(147, 197, 253, 0.54);
    transform: scale(0.94);
  }

  50% {
    box-shadow: 0 0 0 10px rgba(147, 197, 253, 0);
    transform: scale(1.08);
  }
}

@media (max-width: 768px) {
  .screen {
    padding: 18px 14px 28px;
  }

  .hero-shell {
    gap: 22px;
    padding: 18px 0 18px;
  }

  .main-title {
    max-width: 8.5ch;
    font-size: clamp(42px, 13vw, 62px);
  }

  .main-subtitle {
    font-size: 17px;
    max-width: 330px;
  }

  .recording-time {
    font-size: 22px;
    min-height: 42px;
  }

  .helper-line {
    font-size: 13px;
  }

  .secondary-shell {
    width: 100%;
    padding: 20px 18px;
    border-radius: 26px;
  }

  .file-button {
    min-width: 100%;
    font-size: 17px;
  }
}
</style>
