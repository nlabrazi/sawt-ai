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
      },
    })

    const {
      imamDetectionAvailable,
      imamDetectionMessage,
      refreshHealth,
    } = useApiHealth()

    await refreshHealth()

    expect($fetch).toHaveBeenCalledWith('http://localhost:8000/health', {
      method: 'GET',
    })
    expect(imamDetectionAvailable.value).toBe(false)
    expect(imamDetectionMessage.value).toBe('La reconnaissance de l’imam est temporairement indisponible.')
  })

  it('can mark imam detection as unavailable after a runtime failure', () => {
    const {
      imamDetectionAvailable,
      imamDetectionMessage,
      markImamDetectionUnavailable,
    } = useApiHealth()

    markImamDetectionUnavailable('Service imam indisponible.')

    expect(imamDetectionAvailable.value).toBe(false)
    expect(imamDetectionMessage.value).toBe('Service imam indisponible.')
  })
})
