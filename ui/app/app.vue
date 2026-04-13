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
  <main class="app-shell">
    <link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&display=swap" rel="stylesheet">

    <div class="backdrop-grid" />
    <div class="backdrop-orb orb-1" />
    <div class="backdrop-orb orb-2" />
    <div class="backdrop-orb orb-3" />

    <div class="page-frame">
      <RecognitionIdleScreen
        v-if="screenState === 'idle'"
        :upload-error="uploadError"
        :mic-error="micError"
        :is-recording="isRecording"
        :recording-seconds="recordingSeconds"
        :max-recording-seconds="maxRecordingSeconds"
        :upload-accept="uploadAccept"
        :upload-hint="uploadHint"
        :audio-level="audioLevel"
        :imam-detection-available="imamDetectionAvailable"
        :imam-detection-message="imamDetectionMessage"
        v-model:detect-imam="detectImam"
        @micro-click="onMicroClick"
        @select-file="submitAudio"
      />

      <RecognitionLoadingScreen
        v-else-if="screenState === 'loading'"
        :loading="loading"
        :step="loadingStep"
        @cancel="resetApp"
      />

      <RecognitionResultScreen
        v-else
        :error="error"
        :result="result"
        @reset="resetApp"
      />
    </div>

    <AppFooter />
  </main>
</template>

<style scoped>
:global(html, body, #__nuxt) {
  min-height: 100%;
}

:global(body) {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top, rgba(39, 99, 235, 0.22), transparent 20%),
    linear-gradient(180deg, #010818 0%, #020617 52%, #010511 100%);
  color: #fff;
}

:global(*) {
  box-sizing: border-box;
}

.app-shell {
  position: relative;
  min-height: 100vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  color: #fff;
  isolation: isolate;
}

.page-frame {
  position: relative;
  z-index: 1;
  flex: 1;
}

.backdrop-grid {
  position: absolute;
  inset: 0;
  z-index: 0;
  background-image:
    linear-gradient(rgba(148, 163, 184, 0.028) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148, 163, 184, 0.028) 1px, transparent 1px);
  background-size: 44px 44px;
  mask-image: radial-gradient(circle at center, rgba(0, 0, 0, 0.85), transparent 90%);
  pointer-events: none;
}

.backdrop-orb {
  position: absolute;
  border-radius: 999px;
  filter: blur(88px);
  pointer-events: none;
  opacity: 0.9;
}

.orb-1 {
  top: -120px;
  left: 50%;
  width: 420px;
  height: 420px;
  transform: translateX(-50%);
  background: rgba(59, 130, 246, 0.14);
}

.orb-2 {
  right: -80px;
  top: 24%;
  width: 260px;
  height: 260px;
  background: rgba(14, 165, 233, 0.08);
}

.orb-3 {
  left: -60px;
  bottom: 14%;
  width: 220px;
  height: 220px;
  background: rgba(56, 189, 248, 0.06);
}

@media (max-width: 768px) {
  .backdrop-grid {
    background-size: 32px 32px;
  }

  .orb-1 {
    width: 300px;
    height: 300px;
    top: -90px;
  }

  .orb-2,
  .orb-3 {
    filter: blur(68px);
  }
}
</style>
