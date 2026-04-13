<script setup lang="ts">
import { computed } from 'vue'

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

const visualScale = computed(() => {
  const level = Math.max(0, Math.min(1, props.audioLevel))
  return 1 + (level * 0.1)
})

const haloOpacity = computed(() => {
  const level = Math.max(0, Math.min(1, props.audioLevel))
  return 0.22 + (level * 0.36)
})

const waveScale = computed(() => {
  const level = Math.max(0, Math.min(1, props.audioLevel))
  return 1 + (level * 0.2)
})
</script>

<template>
  <button
    class="action-button"
    :class="{
      'is-loading': loading,
      'is-recording': isRecording,
    }"
    :style="{
      '--visual-scale': String(visualScale),
      '--halo-opacity': String(haloOpacity),
      '--wave-scale': String(waveScale),
    }"
    :disabled="disabled"
    type="button"
    @click="emit('click')"
  >
    <span v-if="isRecording" class="wave wave-1" />
    <span v-if="isRecording" class="wave wave-2" />
    <span v-if="isRecording" class="wave wave-3" />

    <span class="button-halo" />
    <span class="button-core">
      <span class="button-icon">{{ isRecording ? '●' : '🎙️' }}</span>
    </span>
  </button>
</template>

<style scoped>
.action-button {
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

.action-button:hover {
  transform: scale(1.02);
}

.action-button:disabled {
  cursor: default;
}

.button-halo {
  position: absolute;
  inset: 0;
  z-index: 2;
  border-radius: 999px;
  opacity: var(--halo-opacity, 0.22);
  background:
    radial-gradient(circle,
      rgba(96, 165, 250, 0.48) 0%,
      rgba(37, 99, 235, 0.18) 45%,
      rgba(37, 99, 235, 0.04) 68%,
      transparent 74%);
  animation: breathe 2.4s ease-in-out infinite;
}

.button-core {
  position: absolute;
  inset: 20px;
  z-index: 3;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background:
    linear-gradient(180deg, rgba(96, 165, 250, 1) 0%, rgba(37, 99, 235, 1) 62%, rgba(29, 78, 216, 1) 100%);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.22),
    0 22px 55px rgba(37, 99, 235, 0.35);
  transform: scale(var(--visual-scale, 1));
  transition: transform 0.08s linear, box-shadow 0.08s linear, background 0.2s ease;
}

.is-recording .button-core {
  background:
    linear-gradient(180deg, rgba(96, 165, 250, 1) 0%, rgba(29, 78, 216, 1) 52%, rgba(2, 132, 199, 1) 100%);
}

.button-icon {
  font-size: 76px;
  transform: translateY(2px);
  filter: drop-shadow(0 6px 18px rgba(255, 255, 255, 0.12));
}

.is-recording .button-icon {
  color: #eff6ff;
  font-size: 54px;
  transform: translateY(-1px);
}

.wave {
  position: absolute;
  inset: -12px;
  z-index: 1;
  border-radius: 999px;
  border: 2px solid rgba(96, 165, 250, 0.48);
  opacity: 0;
  animation: record-wave 2s linear infinite;
}

.wave-2 {
  animation-delay: 0.45s;
}

.wave-3 {
  animation-delay: 0.9s;
}

.is-recording .wave {
  transform: scale(var(--wave-scale, 1));
}

@keyframes breathe {
  0%, 100% { transform: scale(0.98); opacity: 0.76; }
  50% { transform: scale(1.04); opacity: 1; }
}

@keyframes record-wave {
  0% {
    transform: scale(0.92);
    opacity: 0.76;
  }

  65% {
    transform: scale(1.18);
    opacity: 0.18;
  }

  100% {
    transform: scale(1.32);
    opacity: 0;
  }
}

@media (max-width: 768px) {
  .action-button {
    width: 190px;
    height: 190px;
  }

  .button-core {
    inset: 16px;
  }

  .button-icon {
    font-size: 66px;
  }

  .is-recording .button-icon {
    font-size: 48px;
  }
}
</style>
