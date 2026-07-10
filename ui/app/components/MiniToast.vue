<script setup lang="ts">
import CircleCheck from '@lucide/vue/dist/esm/icons/circle-check.mjs'

defineProps<{
  open: boolean
  message: string
}>()
</script>

<template>
  <Teleport to="body">
    <Transition name="mini-toast">
      <div
        v-if="open"
        class="mini-toast"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        <CircleCheck class="mini-toast-icon" :stroke-width="2.25" aria-hidden="true" />
        <span>{{ message }}</span>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.mini-toast {
  position: fixed;
  left: 50%;
  bottom: calc(20px + env(safe-area-inset-bottom));
  z-index: 10000;
  max-width: calc(100vw - 24px);
  min-height: 42px;
  padding: 10px 14px;
  border: 1px solid rgba(134, 239, 172, 0.22);
  border-radius: 8px;
  background: rgba(9, 24, 24, 0.96);
  box-shadow: 0 14px 38px rgba(2, 6, 23, 0.36);
  color: #ecfdf5;
  display: inline-flex;
  align-items: center;
  gap: 9px;
  font-size: 14px;
  font-weight: 700;
  line-height: 1.3;
  text-align: left;
  pointer-events: none;
  transform: translateX(-50%);
  backdrop-filter: blur(14px);
}

.mini-toast-icon {
  width: 18px;
  height: 18px;
  flex: 0 0 auto;
  color: #4ade80;
}

.mini-toast-enter-active,
.mini-toast-leave-active {
  transition:
    opacity 160ms ease,
    transform 160ms ease;
}

.mini-toast-enter-from,
.mini-toast-leave-to {
  opacity: 0;
  transform: translate(-50%, 8px);
}

@media (min-width: 769px) {
  .mini-toast {
    bottom: calc(28px + env(safe-area-inset-bottom));
  }
}

@media (prefers-reduced-motion: reduce) {
  .mini-toast-enter-active,
  .mini-toast-leave-active {
    transition: none;
  }
}
</style>
