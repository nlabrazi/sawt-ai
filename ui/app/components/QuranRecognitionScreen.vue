<script setup lang="ts">
import { defineAsyncComponent, watch } from 'vue'

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
</template>
