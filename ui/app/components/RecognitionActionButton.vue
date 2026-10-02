<script setup lang="ts">
import { LoaderCircle, Mic, Square } from '@lucide/vue'
import { Motion } from 'motion-v'
import { computed } from 'vue'
import { useUiMotion } from '~/composables/useUiMotion'

const props = withDefaults(
  defineProps<{
    disabled?: boolean
    loading?: boolean
    isRecording?: boolean
    audioLevel?: number
    showLabel?: boolean
    loadingLabel?: string
  }>(),
  {
    disabled: false,
    loading: false,
    isRecording: false,
    audioLevel: 0,
    showLabel: true,
    loadingLabel: 'Analyse en cours',
  },
)

const emit = defineEmits<{
  click: []
}>()

const { canAnimate, spring } = useUiMotion()

const safeLevel = computed(() => Math.max(0, Math.min(1, props.audioLevel)))

const visualScale = computed(() => {
  if (!props.isRecording) return 1
  return 1 + safeLevel.value * 0.12
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
  return props.isRecording ? 'Arrêter et analyser' : ''
})

function handleClick() {
  if (props.disabled || props.loading) return
  emit('click')
}
</script>

<template>
  <Motion
    as="button"
    class="action-button"
    :class="{
      'is-loading': loading,
      'is-recording': isRecording,
    }"
    :while-hover="canAnimate && !disabled && !loading ? { scale: 1.04 } : undefined"
    :while-press="canAnimate && !disabled && !loading ? { scale: 0.95 } : undefined"
    :transition="spring"
    :aria-busy="loading"
    :aria-label="actionLabel"
    :aria-pressed="isRecording"
    :disabled="disabled || loading"
    type="button"
    @click="handleClick"
  >
    <Motion
      as="span"
      class="button-visual"
      aria-hidden="true"
      :initial="false"
      :animate="{ scale: canAnimate ? visualScale : 1 }"
      :transition="{ type: 'spring', stiffness: 360, damping: 22 }"
    >
      <Motion
        as="span"
        class="button-signal"
        :initial="false"
        :animate="{ opacity: signalOpacity, scale: canAnimate && loading ? [1, 1.15, 1] : 1 }"
        :transition="{ opacity: { duration: canAnimate ? 0.2 : 0 }, scale: { type: 'tween', duration: canAnimate ? 2.4 : 0, repeat: canAnimate && loading ? Infinity : 0 } }"
      />
      <Motion
        v-for="ripple in 2"
        :key="ripple"
        as="span"
        class="button-ripple"
        :initial="false"
        :animate="{ scale: canAnimate && isRecording ? [1, 1.48] : 1, opacity: canAnimate && isRecording ? [0.5, 0] : 0 }"
        :transition="{ type: 'tween', duration: canAnimate ? 2 : 0, delay: canAnimate ? (ripple - 1) * 1 : 0, repeat: canAnimate && isRecording ? Infinity : 0, ease: 'easeOut' }"
      />
      <Motion
        as="span"
        class="button-orbit"
        :initial="false"
        :animate="{ rotate: canAnimate ? 360 : 0, opacity: isRecording ? 0.6 : 0.35 }"
        :transition="{ rotate: { type: 'tween', duration: canAnimate ? (loading ? 8 : 28) : 0, repeat: canAnimate ? Infinity : 0, ease: 'linear' }, opacity: { duration: canAnimate ? 0.2 : 0 } }"
      >
        <span />
      </Motion>
      <Motion
        as="span"
        class="button-ring"
        :initial="false"
        :animate="{ scale: canAnimate && !isRecording && !loading && !disabled ? [1, 1.055, 1] : 1 }"
        :transition="{ type: 'tween', duration: canAnimate ? 3.6 : 0, repeat: canAnimate && !isRecording && !loading && !disabled ? Infinity : 0, ease: 'easeInOut' }"
      />
      <span class="button-core">
        <Motion
          :key="loading ? 'loading' : isRecording ? 'recording' : 'idle'"
          as="span"
          class="icon-shell"
          :initial="false"
          :animate="{ scale: canAnimate ? [0.65, 1] : 1, opacity: canAnimate ? [0, 1] : 1 }"
          :transition="spring"
        >
          <LoaderCircle v-if="loading" class="button-icon loading-icon" :stroke-width="1.9" />
          <Square v-else-if="isRecording" class="button-icon stop-icon" :stroke-width="2" />
          <Mic v-else class="button-icon" :stroke-width="1.8" />
        </Motion>
      </span>
    </Motion>

    <span v-if="showLabel && visibleLabel" class="button-label">{{ visibleLabel }}</span>
  </Motion>
</template>

<style scoped>
.action-button {
  --visual-size: 172px;
  width: 230px;
  min-height: 200px;
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
}

.button-signal,
.button-ripple,
.button-orbit,
.button-ring,
.button-core {
  position: absolute;
  border-radius: 999px;
}

.button-signal {
  inset: -14px;

  background: radial-gradient(circle, rgba(56, 189, 248, .5), rgba(59, 130, 246, .15));
  filter: blur(16px);
}

.button-ripple {
  inset: -4px;
  border: 1px solid rgba(125, 211, 252, .55);
  pointer-events: none;
}

.button-orbit {
  inset: -18px;
  border: 1px dashed rgba(147, 197, 253, .4);
  pointer-events: none;
}

.button-orbit > span {
  position: absolute;
  top: 23px;
  left: 25px;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: #a5f3fc;
  box-shadow: 0 0 12px rgba(125, 211, 252, .8);
}

.icon-shell { display: grid; place-items: center; }

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
  background: radial-gradient(circle at 30% 15%, #60a5fa, #2563eb 55%, #1e40af);
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
}

.action-button:hover:not(:disabled) .button-core {
  background: #1d4ed8;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.26),
    0 22px 58px rgba(37, 99, 235, 0.36);
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

.is-loading .button-core {
  background: rgba(37, 99, 235, 0.9);
}

.loading-icon {
  width: 52px;
  height: 52px;
  animation: loadingSpin 1.1s linear infinite;
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
    min-height: 178px;
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
