<script setup lang="ts">
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
} = useRecognitionFlow()
</script>

<template>
  <main class="page">
    <div class="background-glow background-glow-1" />
    <div class="background-glow background-glow-2" />
    <link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&display=swap" rel="stylesheet">

    <RecognitionIdleScreen v-if="screenState === 'idle'" :upload-error="uploadError" :mic-error="micError"
      :is-recording="isRecording" :recording-seconds="recordingSeconds" :max-recording-seconds="maxRecordingSeconds"
      :audio-level="audioLevel" @micro-click="onMicroClick" @select-file="submitAudio" />

    <RecognitionLoadingScreen v-else-if="screenState === 'loading'" :loading="loading" :step="loadingStep"
      @cancel="resetApp" />

    <RecognitionResultScreen v-else :error="error" :result="result" @reset="resetApp" />
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

.quran-text {
  font-family: "Amiri", serif;
  font-size: 28px;
  line-height: 1.8;
}
</style>
