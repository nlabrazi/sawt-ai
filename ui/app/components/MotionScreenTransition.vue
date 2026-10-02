<script setup lang="ts">
import { animate } from 'motion-v'
import { nextTick, onBeforeUnmount, watch } from 'vue'
import { useUiMotion } from '~/composables/useUiMotion'

const { canAnimate } = useUiMotion()
const running = new Map<Element, { stop: () => void; finish: () => void }>()

function cancel(element: Element) {
  const animation = running.get(element)
  if (!animation) return
  animation.stop()
  animation.finish()
}

function transition(element: Element, done: () => void, entering: boolean) {
  cancel(element)
  if (!canAnimate.value || !(element instanceof HTMLElement)) {
    // Finish after Vue's current patch so out-in can mount the next screen.
    void nextTick(done)
    return
  }
  const controls = animate(
    element,
    entering ? { opacity: [0, 1], y: [14, 0] } : { opacity: [1, 0], y: [0, -8] },
    { duration: entering ? 0.32 : 0.16, ease: [0.22, 1, 0.36, 1] },
  )
  let finished = false
  const finish = () => {
    if (finished) return
    finished = true
    running.delete(element)
    element.style.removeProperty('opacity')
    element.style.removeProperty('transform')
    done()
  }
  running.set(element, { stop: () => controls.stop(), finish })
  void controls.then(finish)
}

watch(canAnimate, (enabled) => {
  if (!enabled) for (const element of running.keys()) cancel(element)
})
onBeforeUnmount(() => {
  for (const element of running.keys()) cancel(element)
})
</script>

<template>
  <Transition
    name="screen-transition"
    :css="false"
    mode="out-in"
    @enter="(element, done) => transition(element, done, true)"
    @leave="(element, done) => transition(element, done, false)"
    @enter-cancelled="cancel"
    @leave-cancelled="cancel"
  >
    <slot />
  </Transition>
</template>
