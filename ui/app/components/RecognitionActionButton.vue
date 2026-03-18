<script setup lang="ts">
const props = withDefaults(defineProps<{
  disabled?: boolean
  loading?: boolean
  isRecording?: boolean
  audioLevel?: number
}>(), {
  disabled: false,
  loading: false,
  isRecording: false,
  audioLevel: 0,
})

const emit = defineEmits<{
  click: []
}>()

function handleClick() {
  emit('click')
}

const visualScale = computed(() => {
  const level = Math.max(0, Math.min(1, props.audioLevel))
  return 1 + (level * 0.12)
})

const haloOpacity = computed(() => {
  const level = Math.max(0, Math.min(1, props.audioLevel))
  return 0.18 + (level * 0.42)
})

const waveScale = computed(() => {
  const level = Math.max(0, Math.min(1, props.audioLevel))
  return 1 + (level * 0.22)
})
</script>

<template>
  <button class="shazam-button" :class="{
    'is-loading': loading,
    'is-recording': isRecording
  }" :style="{
    '--visual-scale': String(visualScale),
    '--halo-opacity': String(haloOpacity),
    '--wave-scale': String(waveScale),
  }" :disabled="disabled" type="button" @click="handleClick">
    <span v-if="isRecording" class="record-wave wave-1" />
    <span v-if="isRecording" class="record-wave wave-2" />
    <span v-if="isRecording" class="record-wave wave-3" />

    <span class="shazam-button-outer" :class="{ 'pulse-ring': loading || isRecording }" />

    <span class="shazam-button-inner">
      <span class="micro-icon">🎙️</span>
    </span>
  </button>
</template>

<style scoped>
.shazam-button {
  position: relative;
  width: 210px;
  height: 210px;
  border: none;
  background: transparent;
  cursor: pointer;
  padding: 0;
  overflow: visible;
  transition: transform 0.2s ease;
}

.shazam-button:hover {
  transform: scale(1.02);
}

.shazam-button:disabled {
  cursor: default;
}

.shazam-button-outer {
  position: absolute;
  inset: 0;
  z-index: 2;
  border-radius: 999px;
  opacity: var(--halo-opacity, 0.22);
  background:
    radial-gradient(circle,
      rgba(96, 165, 250, 0.42) 0%,
      rgba(37, 99, 235, 0.20) 42%,
      rgba(37, 99, 235, 0.06) 68%,
      transparent 74%);
  transition: opacity 0.08s linear, transform 0.08s linear;
}

.shazam-button-inner {
  position: absolute;
  inset: 20px;
  z-index: 3;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(180deg, #3b82f6 0%, #1d4ed8 100%);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.22),
    0 18px 50px rgba(37, 99, 235, 0.35);
  transform: scale(var(--visual-scale, 1));
  transition: box-shadow 0.08s linear, transform 0.08s linear;
}

.micro-icon {
  font-size: 78px;
  transform: translateY(3px);
}

.record-wave {
  position: absolute;
  inset: -14px;
  z-index: 1;
  border-radius: 999px;
  border: 3px solid rgba(96, 165, 250, 0.55);
  box-shadow:
    0 0 0 6px rgba(59, 130, 246, 0.08),
    0 0 34px rgba(59, 130, 246, 0.22);
  opacity: 0;
  pointer-events: none;
  animation: record-wave 2s linear infinite;
}

.wave-1 {
  animation-delay: 0s;
}

.wave-2 {
  animation-delay: 0.5s;
}

.wave-3 {
  animation-delay: 1s;
}

.pulse-ring {
  animation: pulse 1.6s ease-in-out infinite;
}

.is-recording .record-wave {
  transform: scale(var(--wave-scale, 1));
}

@keyframes pulse {
  0% {
    transform: scale(0.98);
    opacity: 0.82;
  }

  50% {
    transform: scale(1.05);
    opacity: 1;
  }

  100% {
    transform: scale(0.98);
    opacity: 0.82;
  }
}

@keyframes record-wave {
  0% {
    transform: scale(0.92);
    opacity: 0.75;
  }

  60% {
    transform: scale(1.18);
    opacity: 0.24;
  }

  100% {
    transform: scale(1.34);
    opacity: 0;
  }
}

@media (max-width: 768px) {
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

  .record-wave {
    inset: -10px;
  }
}
</style>
