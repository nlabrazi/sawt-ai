import { $fetch } from 'ofetch'

import { resetApiHealthState, useApiHealth } from '~/composables/useApiHealth'

vi.mock('ofetch', () => ({
  $fetch: vi.fn(),
}))

describe('useApiHealth', () => {
  beforeEach(() => {
    resetApiHealthState()
  })

  it('stores imam detection availability from the health endpoint', async () => {
    vi.mocked($fetch).mockResolvedValueOnce({
      status: 'ok',
      services: {
        imam_detection: {
          available: false,
          status: 'unavailable',
          message: 'La reconnaissance de l’imam est temporairement indisponible.',
        },
        upload_policy: {
          max_file_size_bytes: 12 * 1024 * 1024,
          max_audio_duration_seconds: 90,
          accepted_mime_types: ['audio/wav', 'audio/webm'],
          accepted_file_extensions: ['.wav', '.webm'],
        },
        detection_policy: {
          min_accepted_similarity: 0.82,
          min_probable_similarity: 0.65,
          min_matched_word_count: 4,
          min_score_margin: 0.1,
          progressive_analysis_step_seconds: 4,
        },
      },
    })

    const {
      imamDetectionAvailable,
      imamDetectionMessage,
      uploadPolicy,
      detectionPolicy,
      refreshHealth,
    } = useApiHealth()

    await refreshHealth()

    expect($fetch).toHaveBeenCalledWith('http://localhost:8000/health', {
      method: 'GET',
    })
    expect(imamDetectionAvailable.value).toBe(false)
    expect(imamDetectionMessage.value).toBe(
      'La reconnaissance de l’imam est temporairement indisponible.',
    )
    expect(uploadPolicy.value).toEqual({
      max_file_size_bytes: 12 * 1024 * 1024,
      max_audio_duration_seconds: 90,
      accepted_mime_types: ['audio/wav', 'audio/webm'],
      accepted_file_extensions: ['.wav', '.webm'],
    })
    expect(detectionPolicy.value).toEqual({
      min_accepted_similarity: 0.82,
      min_probable_similarity: 0.65,
      min_matched_word_count: 4,
      min_score_margin: 0.1,
      progressive_analysis_step_seconds: 4,
    })
  })

  it('can mark imam detection as unavailable after a runtime failure', () => {
    const { imamDetectionAvailable, imamDetectionMessage, markImamDetectionUnavailable } =
      useApiHealth()

    markImamDetectionUnavailable('Service imam indisponible.')

    expect(imamDetectionAvailable.value).toBe(false)
    expect(imamDetectionMessage.value).toBe('Service imam indisponible.')
  })

  it('keeps the default detection policy with an older health response', async () => {
    vi.mocked($fetch).mockResolvedValueOnce({
      status: 'ok',
      services: {
        imam_detection: {
          available: true,
          status: 'available',
          message: null,
        },
        upload_policy: {
          max_file_size_bytes: 12 * 1024 * 1024,
          max_audio_duration_seconds: 90,
          accepted_mime_types: ['audio/wav'],
          accepted_file_extensions: ['.wav'],
        },
      },
    })

    const { detectionPolicy, refreshHealth } = useApiHealth()

    await refreshHealth()

    expect(detectionPolicy.value).toEqual({
      min_accepted_similarity: 0.8,
      min_probable_similarity: 0.6,
      min_matched_word_count: 3,
      min_score_margin: 0.08,
      progressive_analysis_step_seconds: 5,
    })
  })
})
