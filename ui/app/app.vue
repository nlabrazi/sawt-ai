<script setup lang="ts">
import { defineAsyncComponent, watch } from 'vue'

import AppFooter from '~/components/AppFooter.vue'
import RecognitionIdleScreen from '~/components/RecognitionIdleScreen.vue'
import RecognitionLoadingScreen from '~/components/RecognitionLoadingScreen.vue'
import { useRecognitionFlow } from '~/composables/useRecognitionFlow'

const loadRecognitionResultScreen = () => import('~/components/RecognitionResultScreen.vue')
const RecognitionResultScreen = defineAsyncComponent(loadRecognitionResultScreen)

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
  isFinalizingRecording,
  recordingSeconds,
  maxRecordingSeconds,
  audioLevel,
  uploadAccept,
  uploadHint,
  detectImam,
  imamDetectionAvailable,
  imamDetectionMessage,
} = useRecognitionFlow()

watch(screenState, (state) => {
  if (state === 'loading') {
    void loadRecognitionResultScreen()
  }
})
</script>

<template>
  <main class="page">
    <div class="page-content">
      <Transition name="screen-transition" mode="out-in">
        <RecognitionIdleScreen
          v-if="screenState === 'idle'"
          key="idle"
          :upload-error="uploadError"
          :mic-error="micError"
          :is-recording="isRecording"
          :is-finalizing-recording="isFinalizingRecording"
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
          key="loading"
          :loading="loading"
          :step="loadingStep"
          @cancel="resetApp"
        />

        <RecognitionResultScreen
          v-else
          key="result"
          :error="error"
          :result="result"
          @reset="resetApp"
        />
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
  background: #07101d;
  color: #fff;
}

:global(*) {
  box-sizing: border-box;
}

.page {
  position: relative;
  min-height: 100vh;
  min-height: 100svh;
  overflow-x: hidden;
  color: #fff;
  background: linear-gradient(180deg, #07111f 0%, #08101c 56%, #060d17 100%);
  isolation: isolate;
}

.page::before {
  content: '';
  position: absolute;
  z-index: -1;
  top: -260px;
  left: 50%;
  width: min(820px, 120vw);
  height: 620px;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.18);
  filter: blur(120px);
  transform: translateX(-50%);
  pointer-events: none;
}

.page-content {
  min-height: 100vh;
  min-height: 100svh;
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
  transform: translateY(6px);
}

.screen-transition-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}

@media (max-width: 640px) {
  .page::before {
    top: -220px;
    height: 520px;
    filter: blur(96px);
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
