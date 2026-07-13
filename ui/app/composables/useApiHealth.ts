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

export type DetectionPolicy = {
  min_accepted_similarity: number
  min_probable_similarity: number
  min_matched_word_count: number
  min_score_margin: number
}

export type ApiHealthResponse = {
  status: 'ok'
  services: {
    imam_detection: ImamDetectionHealth
    upload_policy: UploadPolicy
    detection_policy?: Partial<DetectionPolicy>
  }
}

const DEFAULT_DETECTION_POLICY: DetectionPolicy = {
  min_accepted_similarity: 0.8,
  min_probable_similarity: 0.6,
  min_matched_word_count: 3,
  min_score_margin: 0.08,
}

const imamDetection = ref<ImamDetectionHealth>({
  available: true,
  status: 'available',
  message: null,
})
const uploadPolicy = ref<UploadPolicy | null>(null)
const detectionPolicy = ref<DetectionPolicy>({ ...DEFAULT_DETECTION_POLICY })

let activeHealthRequest: Promise<void> | null = null

function applyImamDetectionHealth(nextHealth: ImamDetectionHealth) {
  imamDetection.value = nextHealth
}

function applyUploadPolicy(nextPolicy: UploadPolicy | null) {
  uploadPolicy.value = nextPolicy
}

function resolvePolicyNumber(value: unknown, fallback: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function applyDetectionPolicy(nextPolicy?: Partial<DetectionPolicy>) {
  detectionPolicy.value = {
    min_accepted_similarity: resolvePolicyNumber(
      nextPolicy?.min_accepted_similarity,
      DEFAULT_DETECTION_POLICY.min_accepted_similarity,
    ),
    min_probable_similarity: resolvePolicyNumber(
      nextPolicy?.min_probable_similarity,
      DEFAULT_DETECTION_POLICY.min_probable_similarity,
    ),
    min_matched_word_count: resolvePolicyNumber(
      nextPolicy?.min_matched_word_count,
      DEFAULT_DETECTION_POLICY.min_matched_word_count,
    ),
    min_score_margin: resolvePolicyNumber(
      nextPolicy?.min_score_margin,
      DEFAULT_DETECTION_POLICY.min_score_margin,
    ),
  }
}

export function resetApiHealthState() {
  imamDetection.value = {
    available: true,
    status: 'available',
    message: null,
  }
  uploadPolicy.value = null
  detectionPolicy.value = { ...DEFAULT_DETECTION_POLICY }
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
        applyDetectionPolicy(response.services.detection_policy)
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
    detectionPolicy,
    refreshHealth,
    markImamDetectionUnavailable,
  }
}
