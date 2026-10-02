import { useReducedMotion } from 'motion-v'
import { computed, onMounted, ref } from 'vue'

export function useUiMotion() {
  const mounted = ref(false)
  const reducedMotion = useReducedMotion()
  onMounted(() => {
    mounted.value = true
  })

  // Keep SSR content visible and start motion only after the device preference is known.
  const canAnimate = computed(() => mounted.value && !reducedMotion.value)
  const spring = computed(() =>
    canAnimate.value ? { type: 'spring' as const, stiffness: 260, damping: 24 } : { duration: 0 },
  )

  return { canAnimate, spring }
}
