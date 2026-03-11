<script setup lang="ts">
// ROLE
// ----
// Écran principal Sawt AI en mode "single screen states".
// États :
// - idle    : gros CTA central
// - loading : même écran, focus sur l'analyse
// - result  : affichage du résultat sans garder toute la hero

import FeedbackForm from '~/components/FeedbackForm.vue'
import RecognitionLoader from '~/components/RecognitionLoader.vue'
import ResultCard from '~/components/ResultCard.vue'
import { useRecognition } from '~/composables/useRecognition'

const selectedFile = ref<File | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)

const {
  loading,
  error,
  result,
  recognizeAudio,
  reset,
} = useRecognition()

const screenState = computed<'idle' | 'loading' | 'result'>(() => {
  if (loading.value) return 'loading'
  if (result.value) return 'result'
  return 'idle'
})

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  selectedFile.value = input.files?.[0] ?? null

  if (selectedFile.value) {
    submitAudio()
  }
}

async function submitAudio() {
  if (!selectedFile.value || loading.value) return
  await recognizeAudio(selectedFile.value)
}

function openFilePicker() {
  fileInput.value?.click()
}

function onMicroClick() {
  alert('Enregistrement micro bientôt disponible.')
}

function resetApp() {
  selectedFile.value = null

  if (fileInput.value) {
    fileInput.value.value = ''
  }

  reset()
}
</script>

<template>
  <main class="page">
    <div class="background-glow background-glow-1" />
    <div class="background-glow background-glow-2" />

    <!-- IDLE -->
    <section v-if="screenState === 'idle'" class="screen idle-screen">
      <div class="brand">
        <p class="brand-kicker">Sawt AI</p>
      </div>

      <div class="center-stack">
        <button class="shazam-button" type="button" @click="onMicroClick">
          <span class="shazam-button-outer" />
          <span class="shazam-button-inner">
            <span class="micro-icon">🎙️</span>
          </span>
        </button>

        <h1 class="main-title">Touchez pour réciter</h1>
        <p class="main-subtitle">
          Détection du verset en quelques secondes
        </p>

        <div class="secondary-action">
          <span class="secondary-text">Vous préférez importer un fichier audio ?</span>
          <button class="secondary-link" type="button" @click="openFilePicker">
            Choisir un fichier
          </button>
        </div>

        <input ref="fileInput" class="hidden-input" type="file" accept=".wav,.mp3,.m4a,.ogg" @change="onFileChange" />
      </div>
    </section>

    <!-- LOADING -->
    <section v-else-if="screenState === 'loading'" class="screen loading-screen">
      <div class="top-bar">
        <button class="top-action" type="button" @click="resetApp">
          Annuler
        </button>
      </div>

      <div class="center-stack compact-center">
        <button class="shazam-button is-loading" type="button" disabled>
          <span class="shazam-button-outer pulse-ring" />
          <span class="shazam-button-inner">
            <span class="micro-icon">🎙️</span>
          </span>
        </button>

        <h2 class="main-title loading-title">Détection en cours</h2>
        <p class="main-subtitle">
          Analyse de la récitation et recherche du verset
        </p>

        <div class="loader-shell">
          <RecognitionLoader :loading="loading" />
        </div>
      </div>
    </section>

    <!-- RESULT -->
    <section v-else class="screen result-screen">
      <div class="result-layout">
        <div class="result-top-bar">
          <div>
            <p class="brand-kicker">Sawt AI</p>
            <h1 class="result-title">Résultat de la reconnaissance</h1>
          </div>

          <button class="top-action" type="button" @click="resetApp">
            Nouvelle détection
          </button>
        </div>

        <div v-if="error" class="error-banner">
          {{ error }}
        </div>

        <ResultCard v-if="result" :result="result" />

        <FeedbackForm v-if="result" :result="result" />
      </div>
    </section>
  </main>
</template>

<style scoped>
:global(body) {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #020617;
}

:global(html, body, #__nuxt) {
  min-height: 100%;
}

.page {
  position: relative;
  min-height: 100vh;
  overflow: hidden;
  color: #fff;
  background:
    radial-gradient(circle at top, rgba(59, 130, 246, 0.18), transparent 28%),
    linear-gradient(180deg, #030712 0%, #020617 100%);
}

.background-glow {
  position: absolute;
  border-radius: 999px;
  filter: blur(80px);
  pointer-events: none;
}

.background-glow-1 {
  top: -120px;
  left: 50%;
  width: 420px;
  height: 420px;
  transform: translateX(-50%);
  background: rgba(59, 130, 246, 0.12);
}

.background-glow-2 {
  right: -80px;
  bottom: -120px;
  width: 320px;
  height: 320px;
  background: rgba(14, 165, 233, 0.08);
}

.screen {
  position: relative;
  z-index: 1;
  min-height: 100vh;
  padding: 24px 16px 36px;
  box-sizing: border-box;
}

.idle-screen,
.loading-screen {
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

.compact-center {
  justify-content: center;
  max-width: 760px;
  margin: 0 auto;
}

.analyze-button {
  flex-shrink: 0;
  border: none;
  border-radius: 999px;
  padding: 13px 18px;
  cursor: pointer;
  font-weight: 700;
  background: #f8fafc;
  color: #0f172a;
  transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease;
}

.analyze-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 30px rgba(255, 255, 255, 0.12);
}

.action-button,
.submit-button {
  border: none;
  border-radius: 999px;
  padding: 12px 18px;
  font-weight: 700;
  cursor: pointer;
  transition: transform 0.2s ease, opacity 0.2s ease, box-shadow 0.2s ease;
}

.action-button:hover,
.submit-button:hover {
  transform: translateY(-1px);
}

.success:hover {
  box-shadow: 0 10px 24px rgba(255, 255, 255, 0.1);
}

.danger:hover {
  background: rgba(239, 68, 68, 0.2);
}

.shazam-button {
  position: relative;
  width: 210px;
  height: 210px;
  border: none;
  background: transparent;
  cursor: pointer;
  padding: 0;
  transition: transform 0.25s ease;
}

.shazam-button:hover {
  transform: scale(1.02);
}

.shazam-button-outer {
  position: absolute;
  inset: 0;
  border-radius: 999px;
  background:
    radial-gradient(circle, rgba(96, 165, 250, 0.32) 0%, rgba(37, 99, 235, 0.16) 42%, rgba(37, 99, 235, 0.03) 68%, transparent 72%);
}

.shazam-button-inner {
  position: absolute;
  inset: 20px;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(180deg, #3b82f6 0%, #1d4ed8 100%);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.22),
    0 18px 50px rgba(37, 99, 235, 0.35);
  transition: box-shadow 0.25s ease, transform 0.25s ease;
}

.shazam-button:hover .shazam-button-inner {
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.22),
    0 24px 60px rgba(37, 99, 235, 0.45);
}

.micro-icon {
  font-size: 78px;
  transform: translateY(3px);
}

.main-title {
  margin: 28px 0 0;
  font-size: 34px;
  line-height: 1.08;
  font-weight: 800;
  letter-spacing: -0.03em;
}

.loading-title {
  margin-top: 22px;
}

.main-subtitle {
  margin: 12px 0 0;
  max-width: 420px;
  font-size: 16px;
  line-height: 1.6;
  color: #cbd5e1;
}

.secondary-action {
  margin-top: 34px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.secondary-text {
  color: #94a3b8;
  font-size: 14px;
}

.secondary-link {
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

.hidden-input {
  display: none;
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

.top-bar {
  display: flex;
  justify-content: flex-end;
}

.top-action {
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(15, 23, 42, 0.6);
  color: #e2e8f0;
  border-radius: 999px;
  padding: 10px 14px;
  cursor: pointer;
  font-weight: 600;
  backdrop-filter: blur(10px);
}

.loader-shell {
  width: 100%;
  max-width: 760px;
  margin-top: 28px;
}

.is-loading {
  cursor: default;
}

.pulse-ring {
  animation: pulse 1.8s ease-in-out infinite;
}

@keyframes pulse {
  0% {
    transform: scale(0.98);
    opacity: 0.8;
  }

  50% {
    transform: scale(1.04);
    opacity: 1;
  }

  100% {
    transform: scale(0.98);
    opacity: 0.8;
  }
}

.result-screen {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding-top: 28px;
  padding-bottom: 40px;
  overflow-y: auto;
}

.result-layout {
  width: 100%;
  max-width: 920px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.result-top-bar {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.result-title {
  margin: 8px 0 0;
  font-size: 28px;
  line-height: 1.15;
  font-weight: 800;
  letter-spacing: -0.03em;
}

.error-banner {
  padding: 14px 16px;
  border-radius: 18px;
  color: #fecaca;
  background: rgba(127, 29, 29, 0.35);
  border: 1px solid rgba(239, 68, 68, 0.18);
}

@media (max-width: 768px) {
  .screen {
    padding: 18px 14px 28px;
  }

  .shazam-button {
    width: 180px;
    height: 180px;
  }

  .shazam-button-inner {
    inset: 16px;
  }

  .micro-icon {
    font-size: 64px;
  }

  .main-title {
    font-size: 28px;
  }

  .result-top-bar {
    flex-direction: column;
    align-items: stretch;
  }

  .top-action {
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(15, 23, 42, 0.6);
    color: #e2e8f0;
    border-radius: 999px;
    padding: 10px 14px;
    cursor: pointer;
    font-weight: 600;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, background 0.2s ease, border-color 0.2s ease;
  }

  .top-action:hover {
    transform: translateY(-1px);
    background: rgba(30, 41, 59, 0.78);
    border-color: rgba(147, 197, 253, 0.28);
  }
}
</style>
