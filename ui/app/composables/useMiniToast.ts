import { onBeforeUnmount, ref } from 'vue'

export function useMiniToast(duration = 2000) {
  const message = ref('')
  const visible = ref(false)
  let dismissTimer: ReturnType<typeof setTimeout> | null = null

  function dismiss() {
    visible.value = false

    if (dismissTimer) {
      clearTimeout(dismissTimer)
      dismissTimer = null
    }
  }

  function show(nextMessage: string) {
    if (dismissTimer) clearTimeout(dismissTimer)

    message.value = nextMessage
    visible.value = true
    dismissTimer = setTimeout(dismiss, duration)
  }

  onBeforeUnmount(dismiss)

  return {
    message,
    visible,
    show,
  }
}
