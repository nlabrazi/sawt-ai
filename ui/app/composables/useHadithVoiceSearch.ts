import { computed, getCurrentScope, onScopeDispose, ref, watch } from 'vue'
import type { useHadithSearch } from '~/composables/useHadithSearch'
import { useMicrophoneRecorder } from '~/composables/useMicrophoneRecorder'

export function useHadithVoiceSearch(searchState: ReturnType<typeof useHadithSearch>) {
  const recorder = useMicrophoneRecorder(ref(30))
  const starting = ref(false)
  const recordingError = ref<string | null>(null)
  const busy = computed(
    () => starting.value || recorder.isRecording.value || recorder.isFinalizingRecording.value,
  )
  const error = computed(
    () => recorder.micError.value || recordingError.value || searchState.transcriptionError.value,
  )
  let version = 0
  let finalizing: Promise<void> | null = null

  function cancel() {
    version += 1
    recorder.cleanup()
    starting.value = false
    finalizing = null
    recordingError.value = null
    recorder.micError.value = null
    searchState.transcriptionError.value = null
    searchState.cancel()
  }

  function finish() {
    if (finalizing) return finalizing
    const activeVersion = version
    const pending = (async () => {
      const file = await recorder.stopRecording()
      if (activeVersion !== version) return
      if (!file || file.size === 0) {
        recordingError.value =
          'Aucun enregistrement disponible. Réessayez ou saisissez votre recherche.'
        return
      }
      await searchState.transcribeAndSearch(file)
    })().finally(() => {
      if (finalizing === pending) finalizing = null
    })
    finalizing = pending
    return pending
  }

  async function toggle() {
    if (searchState.loading.value || starting.value) return
    if (recorder.isRecording.value || recorder.isFinalizingRecording.value) {
      await finish()
      return
    }
    recordingError.value = null
    searchState.transcriptionError.value = null
    const activeVersion = ++version
    starting.value = true
    try {
      await recorder.startRecording()
    } finally {
      if (activeVersion === version) starting.value = false
    }
  }

  watch(recorder.maxDurationReached, (reached) => {
    if (reached) void finish()
  })
  if (getCurrentScope()) onScopeDispose(cancel)

  return { ...recorder, starting, busy, error, toggle, cancel }
}
