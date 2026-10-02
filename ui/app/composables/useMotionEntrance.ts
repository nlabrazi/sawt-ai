import { animate } from 'motion-v'
import { onBeforeUnmount, type Ref, watch } from 'vue'
import { useUiMotion } from '~/composables/useUiMotion'

// Animate native elements without wrapping them, preserving dialog refs and focus handling.
export function useMotionEntrance(target: Ref<HTMLElement | null>, distance = 24) {
  const { canAnimate } = useUiMotion()
  let cleanup: (() => void) | undefined

  watch(
    [target, canAnimate],
    ([element, enabled]) => {
      cleanup?.()
      cleanup = undefined
      if (!element || !enabled) return
      const controls = animate(
        element,
        { opacity: [0, 1], y: [distance, 0] },
        { duration: 0.4, ease: [0.22, 1, 0.36, 1] },
      )
      let active = true
      const clearStyles = () => {
        element.style.removeProperty('opacity')
        element.style.removeProperty('transform')
      }
      cleanup = () => {
        active = false
        controls.stop()
        clearStyles()
      }
      void controls.then(() => {
        if (active) clearStyles()
      })
    },
    { flush: 'post' },
  )
  onBeforeUnmount(() => cleanup?.())
}
