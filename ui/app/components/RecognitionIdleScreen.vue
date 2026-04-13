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

const recordingLabel = computed(() => {
  if (!props.isRecording) return 'Touchez pour réciter'
  return `Enregistrement${props.recordingSeconds ? ` · ${props.recordingSeconds}s` : ''}`
})

function onMicroButtonClick() {
  emit('micro-click')
}

function onDetectImamChange(event: Event) {
  if (props.imamDetectionAvailable === false) return
  emit('update:detect-imam', (event.target as HTMLInputElement).checked)
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
    <header class="top-brand">
      <p class="brand-mark">SAWT AI</p>
    </header>

    <div class="hero-shell">
      <div class="hero-copy">
        <p class="hero-kicker">Reconnaissance coranique</p>
        <h1 class="hero-title">{{ recordingLabel }}</h1>
        <p class="hero-subtitle">
          Une écoute rapide, une réponse claire, un parcours pensé d’abord pour mobile.
        </p>
      </div>

      <div class="action-zone">
        <div class="action-ring" :class="{ 'is-recording': isRecording }">
          <RecognitionActionButton
            :is-recording="isRecording"
            :audio-level="audioLevel"
            @click="onMicroButtonClick"
          />
        </div>

        <p class="action-hint">
          {{ isRecording ? 'Touchez à nouveau pour arrêter et lancer l’analyse.' : 'Appuyez pour enregistrer ou importez un fichier audio.' }}
        </p>

        <div class="upload-panel">
          <label class="upload-cta" :for="fileInputId">
            Choisir un fichier
          </label>

          <p class="upload-caption">
            {{ uploadHint ?? 'Formats : wav, mp3, m4a, ogg, webm · max 12 Mo · max 90 sec' }}
          </p>

          <label class="imam-toggle" :class="{ 'is-disabled': imamDetectionAvailable === false }">
            <input
              class="imam-toggle-checkbox"
              type="checkbox"
              :checked="detectImam"
              :disabled="imamDetectionAvailable === false"
              @change="onDetectImamChange"
            >
            <span class="imam-toggle-text">
              Reconnaître l’imam
            </span>
          </label>

          <p class="imam-toggle-hint" :class="{ 'is-unavailable': imamDetectionAvailable === false }">
            {{ imamDetectionAvailable === false
              ? (imamDetectionMessage ?? 'La reconnaissance de l’imam est temporairement indisponible.')
              : 'Analyse du réciteur en plus du verset détecté.' }}
          </p>

          <p v-if="uploadError" class="message-error">
            {{ uploadError }}
          </p>

          <p v-if="micError" class="message-error">
            {{ micError }}
          </p>
        </div>

        <input
          :id="fileInputId"
          class="hidden-input"
          type="file"
          :accept="uploadAccept ?? 'audio/*'"
          @change="onFileChange"
        >
      </div>
    </div>
  </section>
</template>

<style scoped>
.screen {
  position: relative;
  z-index: 1;
  min-height: calc(100vh - 76px);
  padding: 24px 20px 32px;
}

.idle-screen {
  display: flex;
  flex-direction: column;
}

.top-brand {
  display: flex;
  justify-content: center;
}

.brand-mark {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.26em;
  text-transform: uppercase;
  color: #93c5fd;
}

.hero-shell {
  width: min(1080px, 100%);
  margin: 0 auto;
  flex: 1;
  display: grid;
  grid-template-columns: 1.05fr 0.95fr;
  align-items: center;
  gap: 48px;
}

.hero-copy {
  max-width: 520px;
}

.hero-kicker {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #60a5fa;
}

.hero-title {
  margin: 18px 0 0;
  font-size: clamp(42px, 7vw, 72px);
  line-height: 0.96;
  font-weight: 800;
  letter-spacing: -0.05em;
}

.hero-subtitle {
  margin: 18px 0 0;
  max-width: 460px;
  font-size: 18px;
  line-height: 1.72;
  color: #cbd5e1;
}

.action-zone {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.action-ring {
  position: relative;
  display: grid;
  place-items: center;
  width: 360px;
  height: 360px;
  border-radius: 999px;
  background:
    radial-gradient(circle at center, rgba(29, 78, 216, 0.22) 0%, rgba(29, 78, 216, 0.08) 45%, transparent 70%);
  box-shadow:
    0 0 0 1px rgba(148, 163, 184, 0.06),
    inset 0 0 90px rgba(29, 78, 216, 0.06);
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.action-ring.is-recording {
  transform: scale(1.02);
  box-shadow:
    0 0 0 1px rgba(96, 165, 250, 0.18),
    inset 0 0 110px rgba(37, 99, 235, 0.12),
    0 0 80px rgba(37, 99, 235, 0.18);
}

.action-hint {
  margin: 22px 0 0;
  max-width: 380px;
  text-align: center;
  line-height: 1.7;
  color: #94a3b8;
}

.upload-panel {
  margin-top: 26px;
  width: min(430px, 100%);
  padding: 18px;
  border-radius: 28px;
  background: linear-gradient(180deg, rgba(4, 15, 40, 0.72), rgba(2, 8, 23, 0.68));
  border: 1px solid rgba(96, 165, 250, 0.12);
  backdrop-filter: blur(12px);
  text-align: center;
}

.upload-cta {
  display: inline-flex;
  min-height: 52px;
  padding: 0 18px;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: rgba(59, 130, 246, 0.16);
  color: #e0f2fe;
  font-weight: 700;
  cursor: pointer;
  transition: transform 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
  box-shadow: inset 0 0 0 1px rgba(96, 165, 250, 0.22);
}

.upload-cta:hover {
  transform: translateY(-1px);
  background: rgba(59, 130, 246, 0.22);
}

.upload-caption {
  margin: 14px 0 0;
  color: #94a3b8;
  font-size: 14px;
  line-height: 1.6;
}

.imam-toggle {
  margin-top: 16px;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  color: #dbeafe;
  cursor: pointer;
}

.imam-toggle.is-disabled {
  cursor: not-allowed;
  color: #64748b;
}

.imam-toggle-checkbox {
  width: 16px;
  height: 16px;
  accent-color: #60a5fa;
}

.imam-toggle-text {
  font-size: 14px;
  font-weight: 600;
}

.imam-toggle-hint {
  margin: 8px 0 0;
  font-size: 13px;
  line-height: 1.5;
  color: #64748b;
}

.imam-toggle-hint.is-unavailable,
.message-error {
  color: #fca5a5;
}

.hidden-input {
  display: none;
}

@media (max-width: 980px) {
  .hero-shell {
    grid-template-columns: 1fr;
    justify-items: center;
    text-align: center;
    gap: 30px;
  }

  .hero-copy {
    max-width: 620px;
  }

  .hero-subtitle {
    margin-left: auto;
    margin-right: auto;
  }
}

@media (max-width: 768px) {
  .screen {
    min-height: calc(100vh - 70px);
    padding: 18px 16px 26px;
  }

  .hero-shell {
    gap: 22px;
  }

  .hero-copy {
    margin-top: 22px;
  }

  .hero-title {
    font-size: 48px;
  }

  .hero-subtitle {
    font-size: 17px;
  }

  .action-ring {
    width: 100%;
    max-width: 320px;
    height: 320px;
  }

  .upload-panel {
    border-radius: 24px;
    padding: 16px;
  }
}
</style>
