<script setup lang="ts">
import { Motion } from 'motion-v'
import { computed } from 'vue'
import { useUiMotion } from '~/composables/useUiMotion'

const props = withDefaults(defineProps<{ delay?: number; distance?: number }>(), {
  delay: 0,
  distance: 18,
})
const { canAnimate } = useUiMotion()
const entrance = computed(() =>
  canAnimate.value ? { opacity: [0, 1], y: [props.distance, 0] } : { opacity: 1, y: 0 },
)
</script>

<template>
  <Motion
    :initial="false"
    :animate="entrance"
    :transition="{ duration: canAnimate ? 0.5 : 0, delay: canAnimate ? delay : 0, ease: [0.22, 1, 0.36, 1] }"
  >
    <slot />
  </Motion>
</template>
