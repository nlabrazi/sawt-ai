<script setup lang="ts">
import { computed, ref } from 'vue'

const props = withDefaults(
  defineProps<{
    disabled?: boolean
    loading?: boolean
    isRecording?: boolean
    audioLevel?: number
  }>(),
  {
    disabled: false,
    loading: false,
    isRecording: false,
    audioLevel: 0,
  },
)

const emit = defineEmits<{
  click: []
}>()

const isPressed = ref(false)

const safeLevel = computed(() => {
  return Math.max(0, Math.min(1, props.audioLevel))
})

const coreScale = computed(() => {
  if (props.loading) return 1
  if (props.isRecording) return 1 + safeLevel.value * 0.08
  return 1
})

const haloOpacity = computed(() => {
  if (props.loading) return 0.42
  if (props.isRecording) return 0.34 + safeLevel.value * 0.3
  return 0.26
})

const auraOpacity = computed(() => {
  if (props.loading) return 0.28
  if (props.isRecording) return 0.28 + safeLevel.value * 0.22
  return 0.16
})

function handleClick() {
  if (props.disabled) return
  emit('click')
}

function handlePressStart() {
  if (props.disabled) return
  isPressed.value = true
}

function handlePressEnd() {
  isPressed.value = false
}
</script>

<template>
  <button class="action-button" :class="{
    'is-loading': loading,
    'is-recording': isRecording,
    'is-pressed': isPressed,
  }" :style="{
      '--core-scale': String(coreScale),
      '--halo-opacity': String(haloOpacity),
      '--aura-opacity': String(auraOpacity),
    }" :disabled="disabled" type="button" @click="handleClick" @mousedown="handlePressStart" @mouseup="handlePressEnd"
    @mouseleave="handlePressEnd" @touchstart="handlePressStart" @touchend="handlePressEnd"
    @touchcancel="handlePressEnd">
    <span class="button-aura" aria-hidden="true" />
    <span class="button-halo" aria-hidden="true" />
    <span class="button-ring ring-1" aria-hidden="true" />
    <span class="button-ring ring-2" aria-hidden="true" />

    <span v-if="isRecording || loading" class="record-wave wave-1" aria-hidden="true" />
    <span v-if="isRecording || loading" class="record-wave wave-2" aria-hidden="true" />
    <span v-if="isRecording || loading" class="record-wave wave-3" aria-hidden="true" />

    <span class="button-core">
      <span class="button-shine" aria-hidden="true" />
      <span class="button-icon">
        {{ isRecording ? '●' : '🎙️' }}
      </span>
    </span>
  </button>
</template>

<style scoped>
.action-button {
  position: relative;
  width: 220px;
  height: 220px;
  padding: 0;
  border: none;
  background: transparent;
  border-radius: 999px;
  cursor: pointer;
  overflow: visible;
  isolation: isolate;
  transition: transform 0.24s ease, filter 0.24s ease;
  -webkit-tap-highlight-color: transparent;
}

.action-button:hover {
  transform: translateY(-2px) scale(1.03);
  filter: saturate(1.06);
}

.action-button.is-pressed {
  transform: scale(0.97);
}

.action-button:disabled {
  cursor: default;
}

.action-button:disabled:hover {
  transform: none;
  filter: none;
}

.button-aura,
.button-halo,
.button-ring,
.record-wave,
.button-core,
.button-shine {
  position: absolute;
  border-radius: 999px;
}

.button-aura {
  inset: -26px;
  z-index: 0;
  opacity: var(--aura-opacity, 0.16);
  background:
    radial-gradient(circle, rgba(59, 130, 246, 0.34) 0%, rgba(14, 165, 233, 0.14) 42%, transparent 72%);
  filter: blur(22px);
  animation: auraFloat 6s ease-in-out infinite;
}

.button-halo {
  inset: -8px;
  z-index: 1;
  opacity: var(--halo-opacity, 0.26);
  background:
    radial-gradient(circle, rgba(96, 165, 250, 0.48) 0%, rgba(37, 99, 235, 0.16) 52%, transparent 74%);
  animation: haloBreath 3s ease-in-out infinite;
}

.button-ring {
  inset: -2px;
  z-index: 2;
  border: 1px solid rgba(125, 211, 252, 0.14);
  pointer-events: none;
}

.ring-1 {
  animation: ringDrift 5s ease-in-out infinite;
}

.ring-2 {
  inset: -14px;
  border-color: rgba(96, 165, 250, 0.1);
  animation: ringDrift 5.8s ease-in-out infinite reverse;
}

.record-wave {
  inset: -10px;
  z-index: 1;
  border: 1.5px solid rgba(125, 211, 252, 0.28);
  opacity: 0;
  pointer-events: none;
  animation: recordWave 2.3s linear infinite;
}

.wave-2 {
  animation-delay: 0.65s;
}

.wave-3 {
  animation-delay: 1.3s;
}

.button-core {
  inset: 18px;
  z-index: 3;
  display: grid;
  place-items: center;
  overflow: hidden;
  transform: scale(var(--core-scale, 1));
  background:
    radial-gradient(circle at 30% 24%, rgba(147, 197, 253, 0.9) 0%, rgba(96, 165, 250, 0.72) 20%, transparent 42%),
    linear-gradient(180deg, #60a5fa 0%, #3b82f6 34%, #2563eb 72%, #0ea5e9 100%);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.3),
    inset 0 -24px 38px rgba(2, 6, 23, 0.16),
    0 24px 70px rgba(37, 99, 235, 0.32);
  transition:
    transform 80ms linear,
    box-shadow 180ms ease,
    filter 180ms ease;
}

.action-button:hover .button-core {
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.34),
    inset 0 -24px 38px rgba(2, 6, 23, 0.16),
    0 28px 84px rgba(37, 99, 235, 0.38);
}

.is-recording .button-core {
  background:
    radial-gradient(circle at 30% 24%, rgba(191, 219, 254, 0.92) 0%, rgba(96, 165, 250, 0.78) 18%, transparent 42%),
    linear-gradient(180deg, #60a5fa 0%, #2563eb 48%, #1d4ed8 70%, #0284c7 100%);
}

.is-loading .button-aura {
  animation: loadingAura 1.8s ease-in-out infinite;
}

.is-loading .button-halo {
  animation: loadingHalo 1.4s ease-in-out infinite;
}

.is-loading .ring-1 {
  animation: loadingRing 1.7s ease-in-out infinite;
}

.is-loading .ring-2 {
  animation: loadingRing 1.7s ease-in-out infinite 0.25s;
}

.is-loading .button-core {
  animation: loadingPulse 1.45s ease-in-out infinite;
}

.button-shine {
  inset: auto auto 54% -10%;
  width: 86%;
  height: 42%;
  opacity: 0.28;
  background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.34) 50%, transparent 100%);
  transform: rotate(-18deg);
  animation: shineMove 4.6s ease-in-out infinite;
}

.button-icon {
  position: relative;
  z-index: 2;
  font-size: 78px;
  line-height: 1;
  transform: translateY(2px);
  filter: drop-shadow(0 10px 18px rgba(255, 255, 255, 0.08));
  user-select: none;
}

.is-recording .button-icon {
  font-size: 52px;
  color: #eff6ff;
  transform: translateY(0);
  text-shadow: 0 0 20px rgba(255, 255, 255, 0.18);
}

@keyframes loadingPulse {

  0%,
  100% {
    transform: scale(1);
    filter: brightness(1);
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, 0.3),
      inset 0 -24px 38px rgba(2, 6, 23, 0.16),
      0 24px 70px rgba(37, 99, 235, 0.32);
  }

  50% {
    transform: scale(1.08);
    filter: brightness(1.16);
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, 0.35),
      inset 0 -24px 38px rgba(2, 6, 23, 0.12),
      0 32px 88px rgba(56, 189, 248, 0.44);
  }
}

@keyframes loadingHalo {

  0%,
  100% {
    transform: scale(0.96);
    opacity: 0.28;
  }

  50% {
    transform: scale(1.08);
    opacity: 0.52;
  }
}

@keyframes loadingAura {

  0%,
  100% {
    transform: scale(0.96);
    opacity: 0.18;
  }

  50% {
    transform: scale(1.08);
    opacity: 0.34;
  }
}

@keyframes loadingRing {

  0%,
  100% {
    transform: scale(0.98);
    opacity: 0.55;
  }

  50% {
    transform: scale(1.08);
    opacity: 1;
  }
}

@keyframes haloBreath {

  0%,
  100% {
    transform: scale(0.98);
    opacity: calc(var(--halo-opacity, 0.26) * 0.82);
  }

  50% {
    transform: scale(1.05);
    opacity: var(--halo-opacity, 0.26);
  }
}

@keyframes auraFloat {

  0%,
  100% {
    transform: scale(0.98) translateY(0);
  }

  50% {
    transform: scale(1.04) translateY(-3px);
  }
}

@keyframes ringDrift {

  0%,
  100% {
    transform: scale(0.99);
    opacity: 0.72;
  }

  50% {
    transform: scale(1.03);
    opacity: 1;
  }
}

@keyframes recordWave {
  0% {
    transform: scale(0.88);
    opacity: 0.62;
  }

  55% {
    opacity: 0.18;
  }

  100% {
    transform: scale(1.28);
    opacity: 0;
  }
}

@keyframes shineMove {

  0%,
  100% {
    transform: translateX(-8%) rotate(-18deg);
    opacity: 0.2;
  }

  50% {
    transform: translateX(12%) rotate(-18deg);
    opacity: 0.36;
  }
}

@media (max-width: 768px) {
  .action-button {
    width: 196px;
    height: 196px;
  }

  .button-core {
    inset: 16px;
  }

  .button-icon {
    font-size: 68px;
  }

  .is-recording .button-icon {
    font-size: 46px;
  }

  .button-aura {
    inset: -20px;
  }
}

@media (prefers-reduced-motion: reduce) {

  .action-button,
  .button-aura,
  .button-halo,
  .button-ring,
  .record-wave,
  .button-core,
  .button-shine {
    animation: none !important;
    transition: none !important;
  }
}
</style>
