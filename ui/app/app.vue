<script setup lang="ts">
import AppFooter from '~/components/AppFooter.vue'
import RecognitionIdleScreen from '~/components/RecognitionIdleScreen.vue'
import RecognitionLoadingScreen from '~/components/RecognitionLoadingScreen.vue'
import RecognitionResultScreen from '~/components/RecognitionResultScreen.vue'
import { useRecognitionFlow } from '~/composables/useRecognitionFlow'

const {
  screenState,
  onMicroClick,
  submitAudio,
  loading,
  loadingStep,
  resetApp,
  error,
  result,
  uploadError,
  micError,
  isRecording,
  recordingSeconds,
  maxRecordingSeconds,
  audioLevel,
  uploadAccept,
  uploadHint,
  detectImam,
  imamDetectionAvailable,
  imamDetectionMessage,
} = useRecognitionFlow()
</script>

<template>
  <main class="page">
    <div class="background-glow background-glow-1" />
    <div class="background-glow background-glow-2" />
    <div class="background-glow background-glow-3" />
    <div class="background-glow background-glow-4" />
    <link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&display=swap" rel="stylesheet">

    <div class="page-grid" />

    <div class="page-noise" aria-hidden="true" />

    <div class="page-content">
      <Transition name="screen-transition" mode="out-in">
        <RecognitionIdleScreen v-if="screenState === 'idle'" key="idle" :upload-error="uploadError"
          :mic-error="micError" :is-recording="isRecording" :recording-seconds="recordingSeconds"
          :max-recording-seconds="maxRecordingSeconds" :upload-accept="uploadAccept" :upload-hint="uploadHint"
          :audio-level="audioLevel" :imam-detection-available="imamDetectionAvailable"
          :imam-detection-message="imamDetectionMessage" v-model:detect-imam="detectImam"
          @micro-click="onMicroClick" @select-file="submitAudio" />

        <RecognitionLoadingScreen v-else-if="screenState === 'loading'" key="loading" :loading="loading"
          :step="loadingStep" @cancel="resetApp" />

        <RecognitionResultScreen v-else key="result" :error="error" :result="result" @reset="resetApp" />
      </Transition>

      <AppFooter />
    </div>
  </main>
</template>

<style scoped>
:global(html, body, #__nuxt) {
  min-height: 100%;
}

:global(body) {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #081120;
  color: #fff;
}

:global(*) {
  box-sizing: border-box;
}

.page {
  position: relative;
  min-height: 100vh;
  overflow-x: hidden;
  color: #fff;
  background:
    radial-gradient(circle at 50% 12%, rgba(59, 130, 246, 0.24), transparent 28%),
    radial-gradient(circle at 78% 74%, rgba(14, 165, 233, 0.1), transparent 20%),
    radial-gradient(circle at 14% 78%, rgba(59, 130, 246, 0.08), transparent 18%),
    linear-gradient(180deg, #040b16 0%, #07101d 44%, #060d18 100%);
}

.page-content {
  position: relative;
  z-index: 1;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.screen-transition-enter-active {
  transition:
    opacity 180ms ease-out,
    transform 180ms ease-out;
}

.screen-transition-leave-active {
  transition:
    opacity 120ms ease-in,
    transform 120ms ease-in;
}

.screen-transition-enter-from {
  opacity: 0;
  transform: translateY(8px) scale(0.995);
}

.screen-transition-leave-to {
  opacity: 0;
  transform: translateY(-6px) scale(0.995);
}

.page-grid {
  position: absolute;
  inset: 0;
  z-index: 0;
  opacity: 0.18;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(148, 163, 184, 0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148, 163, 184, 0.04) 1px, transparent 1px);
  background-size: 48px 48px;
  mask-image: radial-gradient(circle at center, rgba(0, 0, 0, 0.92), transparent 92%);
}

.page-noise {
  position: absolute;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  opacity: 0.05;
  background-image:
    radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.12) 0 1px, transparent 1px),
    radial-gradient(circle at 80% 40%, rgba(255, 255, 255, 0.08) 0 1px, transparent 1px),
    radial-gradient(circle at 40% 70%, rgba(255, 255, 255, 0.08) 0 1px, transparent 1px);
  background-size: 180px 180px;
}

.background-glow {
  position: absolute;
  border-radius: 999px;
  filter: blur(96px);
  pointer-events: none;
}

.background-glow-1 {
  top: -90px;
  left: 50%;
  width: 520px;
  height: 520px;
  transform: translateX(-50%);
  background: rgba(59, 130, 246, 0.18);
}

.background-glow-2 {
  right: -120px;
  top: 24%;
  width: 380px;
  height: 380px;
  background: rgba(14, 165, 233, 0.09);
}

.background-glow-3 {
  left: -110px;
  bottom: 10%;
  width: 320px;
  height: 320px;
  background: rgba(96, 165, 250, 0.08);
}

.background-glow-4 {
  left: 50%;
  bottom: -120px;
  width: 420px;
  height: 420px;
  transform: translateX(-50%);
  background: rgba(30, 64, 175, 0.08);
}

@media (max-width: 768px) {
  .page {
    background:
      radial-gradient(circle at 50% 14%, rgba(59, 130, 246, 0.28), transparent 34%),
      radial-gradient(circle at 82% 70%, rgba(14, 165, 233, 0.08), transparent 24%),
      linear-gradient(180deg, #050c17 0%, #07101d 50%, #060d18 100%);
  }

  .page-grid {
    opacity: 0.12;
    background-size: 42px 42px;
  }

  .background-glow-1 {
    width: 360px;
    height: 360px;
    top: -60px;
  }

  .background-glow-2,
  .background-glow-3,
  .background-glow-4 {
    filter: blur(80px);
  }
}

@media (prefers-reduced-motion: reduce) {
  .screen-transition-enter-active,
  .screen-transition-leave-active {
    transition: none;
  }

  .screen-transition-enter-from,
  .screen-transition-leave-to {
    transform: none;
  }
}
</style>
