import { useRuntimeConfig } from '#app'
import { $fetch } from 'ofetch'
import { computed, ref } from 'vue'

const DEFAULT_IMAM_UNAVAILABLE_MESSAGE =
  'La reconnaissance de l’imam est temporairement indisponible.'

export type ImamDetectionHealth = {
  available: boolean
  status: 'available' | 'unavailable'
  message: string | null
}

export type UploadPolicy = {
  max_file_size_bytes: number
  max_audio_duration_seconds: number
  accepted_mime_types: string[]
  accepted_file_extensions: string[]
}

export type ApiHealthResponse = {
  status: 'ok'
  services: {
    imam_detection: ImamDetectionHealth
    upload_policy: UploadPolicy
  }
}

const imamDetection = ref<ImamDetectionHealth>({
  available: true,
  status: 'available',
  message: null,
})
const uploadPolicy = ref<UploadPolicy | null>(null)

let activeHealthRequest: Promise<void> | null = null

function applyImamDetectionHealth(nextHealth: ImamDetectionHealth) {
  imamDetection.value = nextHealth
}

function applyUploadPolicy(nextPolicy: UploadPolicy | null) {
  uploadPolicy.value = nextPolicy
}

export function resetApiHealthState() {
  imamDetection.value = {
    available: true,
    status: 'available',
    message: null,
  }
  uploadPolicy.value = null
  activeHealthRequest = null
}

export function useApiHealth() {
  const apiBaseUrl = useRuntimeConfig().public.apiBaseUrl.replace(/\/$/, '')
  const imamDetectionAvailable = computed(() => imamDetection.value.available)
  const imamDetectionMessage = computed(() => imamDetection.value.message)

  async function refreshHealth() {
    if (activeHealthRequest) {
      return activeHealthRequest
    }

    activeHealthRequest = (async () => {
      try {
        const response = await $fetch<ApiHealthResponse>(`${apiBaseUrl}/health`, {
          method: 'GET',
        })

        applyImamDetectionHealth(response.services.imam_detection)
        applyUploadPolicy(response.services.upload_policy)
      } catch (error) {
        console.error(error)
      } finally {
        activeHealthRequest = null
      }
    })()

    return activeHealthRequest
  }

  function markImamDetectionUnavailable(message = DEFAULT_IMAM_UNAVAILABLE_MESSAGE) {
    applyImamDetectionHealth({
      available: false,
      status: 'unavailable',
      message,
    })
  }

  return {
    imamDetection,
    imamDetectionAvailable,
    imamDetectionMessage,
    uploadPolicy,
    refreshHealth,
    markImamDetectionUnavailable,
  }
}
