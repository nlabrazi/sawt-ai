<script setup lang="ts">
withDefaults(defineProps<{
  disabled?: boolean
  loading?: boolean
}>(), {
  disabled: false,
  loading: false,
})

defineEmits<{
  click: []
}>()
</script>

<template>
  <button
    class="shazam-button"
    :class="{ 'is-loading': loading }"
    :disabled="disabled"
    type="button"
    @click="$emit('click')"
  >
    <span class="shazam-button-outer" :class="{ 'pulse-ring': loading }" />
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
  transition: transform 0.25s ease;
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
  border-radius: 999px;
  background:
    radial-gradient(circle, rgba(96, 165, 250, 0.32) 0%, rgba(37, 99, 235, 0.16) 42%, rgba(37, 99, 235, 0.03) 68%, transparent 72%);
}

.shazam-button-inner {
  position: absolute;
  inset: 20px;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(180deg, #3b82f6 0%, #1d4ed8 100%);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.22),
    0 18px 50px rgba(37, 99, 235, 0.35);
  transition: box-shadow 0.25s ease, transform 0.25s ease;
}

.shazam-button:hover .shazam-button-inner {
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.22),
    0 24px 60px rgba(37, 99, 235, 0.45);
}

.is-loading:hover {
  transform: none;
}

.is-loading:hover .shazam-button-inner {
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.22),
    0 18px 50px rgba(37, 99, 235, 0.35);
}

.micro-icon {
  font-size: 78px;
  transform: translateY(3px);
}

.pulse-ring {
  animation: pulse 1.8s ease-in-out infinite;
}

@keyframes pulse {
  0% {
    transform: scale(0.98);
    opacity: 0.8;
  }

  50% {
    transform: scale(1.04);
    opacity: 1;
  }

  100% {
    transform: scale(0.98);
    opacity: 0.8;
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
}
</style>
