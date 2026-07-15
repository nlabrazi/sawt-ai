<script setup lang="ts">
import LoaderCircle from '@lucide/vue/dist/esm/icons/loader-circle.mjs'
import Mic from '@lucide/vue/dist/esm/icons/mic.mjs'
import Square from '@lucide/vue/dist/esm/icons/square.mjs'
import { computed, ref } from 'vue'

const props = withDefaults(
  defineProps<{
    disabled?: boolean
    loading?: boolean
    isRecording?: boolean
    audioLevel?: number
    loadingLabel?: string
  }>(),
  {
    disabled: false,
    loading: false,
    isRecording: false,
    audioLevel: 0,
    loadingLabel: 'Analyse en cours',
  },
)

const emit = defineEmits<{
  click: []
}>()

const isPressed = ref(false)

const safeLevel = computed(() => Math.max(0, Math.min(1, props.audioLevel)))

const visualScale = computed(() => {
  if (!props.isRecording) return 1
  return 1 + safeLevel.value * 0.06
})

const signalOpacity = computed(() => {
  if (props.loading) return 0.5
  if (props.isRecording) return 0.32 + safeLevel.value * 0.36
  return 0.2
})

const actionLabel = computed(() => {
  if (props.loading) return props.loadingLabel
  return props.isRecording ? 'Arrêter et analyser' : 'Commencer la récitation'
})

const visibleLabel = computed(() => {
  if (props.loading) return props.loadingLabel
  return props.isRecording ? 'Arrêter et analyser' : 'Commencer'
})

function handleClick() {
  if (props.disabled || props.loading) return
  emit('click')
}

function handlePressStart() {
  if (props.disabled || props.loading) return
  isPressed.value = true
}

function handlePressEnd() {
  isPressed.value = false
}
</script>

<template>
  <button
    class="action-button"
    :class="{
      'is-loading': loading,
      'is-recording': isRecording,
      'is-pressed': isPressed,
    }"
    :style="{
      '--visual-scale': String(visualScale),
      '--signal-opacity': String(signalOpacity),
    }"
    :aria-busy="loading"
    :aria-label="actionLabel"
    :aria-pressed="isRecording"
    :disabled="disabled || loading"
    type="button"
    @click="handleClick"
    @mousedown="handlePressStart"
    @mouseup="handlePressEnd"
    @mouseleave="handlePressEnd"
    @touchstart="handlePressStart"
    @touchend="handlePressEnd"
    @touchcancel="handlePressEnd"
  >
    <span class="button-visual" aria-hidden="true">
      <span class="button-signal" />
      <span class="button-ring" />
      <span class="button-core">
        <LoaderCircle v-if="loading" class="button-icon loading-icon" :stroke-width="1.9" />
        <Square v-else-if="isRecording" class="button-icon stop-icon" :stroke-width="2" />
        <Mic v-else class="button-icon" :stroke-width="1.8" />
      </span>
    </span>

    <span class="button-label">{{ visibleLabel }}</span>
  </button>
</template>

<style scoped>
.action-button {
  --visual-size: 172px;
  width: 230px;
  min-height: 224px;
  padding: 8px 12px 4px;
  border: 0;
  border-radius: 32px;
  background: transparent;
  color: #f8fafc;
  display: inline-grid;
  justify-items: center;
  align-content: center;
  gap: 18px;
  cursor: pointer;
  -webkit-tap-highlight-color: transparent;
}

.action-button:focus-visible {
  outline: 3px solid rgba(147, 197, 253, 0.9);
  outline-offset: 5px;
}

.action-button:disabled {
  cursor: default;
}

.button-visual {
  position: relative;
  width: var(--visual-size);
  height: var(--visual-size);
  display: grid;
  place-items: center;
  transform: scale(var(--visual-scale, 1));
  transition: transform 90ms linear;
}

.button-signal,
.button-ring,
.button-core {
  position: absolute;
  border-radius: 999px;
}

.button-signal {
  inset: -14px;
  opacity: var(--signal-opacity, 0.2);
  background: rgba(59, 130, 246, 0.3);
  filter: blur(16px);
  transition: opacity 120ms linear;
}

.button-ring {
  inset: 0;
  border: 1px solid rgba(147, 197, 253, 0.3);
  background: rgba(37, 99, 235, 0.08);
  transition:
    border-color 180ms ease,
    transform 180ms ease;
}

.button-core {
  inset: 10px;
  display: grid;
  place-items: center;
  overflow: hidden;
  background: #2563eb;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.24),
    0 18px 48px rgba(37, 99, 235, 0.3);
  transition:
    background 180ms ease,
    box-shadow 180ms ease,
    transform 180ms ease;
}

.button-icon {
  width: 60px;
  height: 60px;
  color: #fff;
}

.stop-icon {
  width: 44px;
  height: 44px;
  fill: currentColor;
}

.button-label {
  font-size: 17px;
  line-height: 1.2;
  font-weight: 750;
  letter-spacing: -0.01em;
}

.action-button:hover:not(:disabled) .button-ring {
  border-color: rgba(191, 219, 254, 0.58);
  transform: scale(1.025);
}

.action-button:hover:not(:disabled) .button-core {
  background: #1d4ed8;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.26),
    0 22px 58px rgba(37, 99, 235, 0.36);
}

.action-button.is-pressed .button-core {
  transform: scale(0.96);
}

.is-recording .button-signal {
  animation: listeningPulse 1.8s ease-out infinite;
}

.is-recording .button-ring {
  border-color: rgba(226, 232, 240, 0.44);
  background: rgba(255, 255, 255, 0.04);
}

.is-recording .button-core {
  background: #f8fafc;
  box-shadow: 0 18px 48px rgba(2, 6, 23, 0.28);
}

.is-recording:hover:not(:disabled) .button-core {
  background: #fff;
  box-shadow: 0 22px 58px rgba(2, 6, 23, 0.34);
}

.is-recording .button-icon {
  color: #0f172a;
}

.is-loading .button-signal {
  animation: loadingPulse 1.7s ease-in-out infinite;
}

.is-loading .button-core {
  background: rgba(37, 99, 235, 0.9);
}

.loading-icon {
  width: 52px;
  height: 52px;
  animation: loadingSpin 1.1s linear infinite;
}

@keyframes listeningPulse {
  0% {
    transform: scale(0.92);
    opacity: var(--signal-opacity, 0.32);
  }

  75%,
  100% {
    transform: scale(1.2);
    opacity: 0;
  }
}

@keyframes loadingPulse {
  0%,
  100% {
    transform: scale(0.94);
    opacity: 0.22;
  }

  50% {
    transform: scale(1.08);
    opacity: 0.48;
  }
}

@keyframes loadingSpin {
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 640px) {
  .action-button {
    --visual-size: 150px;
    width: 204px;
    min-height: 202px;
    gap: 16px;
  }

  .button-icon {
    width: 52px;
    height: 52px;
  }

  .stop-icon {
    width: 38px;
    height: 38px;
  }

  .button-label {
    font-size: 16px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .button-visual,
  .button-signal,
  .button-ring,
  .button-core,
  .loading-icon {
    animation: none !important;
    transition: none !important;
  }
}
</style>
