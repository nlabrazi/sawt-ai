import { $fetch } from 'ofetch'
import { useFeedback } from '~/composables/useFeedback'

vi.mock('ofetch', () => ({
  $fetch: vi.fn(),
}))

describe('useFeedback', () => {
  it('posts feedback to the backend and resets the sending flag', async () => {
    vi.mocked($fetch).mockResolvedValueOnce({ success: true })

    const { sending, error, sendFeedback } = useFeedback()

    await sendFeedback({
      is_correct: true,
      transcription_text: 'قل هو الله احد',
      detected_verse: null,
      correction: null,
      comment: null,
    })

    expect($fetch).toHaveBeenCalledWith('http://localhost:8000/feedback', {
      method: 'POST',
      body: {
        is_correct: true,
        transcription_text: 'قل هو الله احد',
        detected_verse: null,
        correction: null,
        comment: null,
      },
    })
    expect(sending.value).toBe(false)
    expect(error.value).toBeNull()
  })

  it('exposes a user-facing error when the request fails', async () => {
    vi.mocked($fetch).mockRejectedValueOnce(new Error('network'))

    const { error, sendFeedback } = useFeedback()

    await expect(sendFeedback({
      is_correct: false,
      transcription_text: 'text',
      detected_verse: null,
      correction: null,
      comment: null,
    })).rejects.toThrow('network')

    expect(error.value).toBe('Erreur pendant l’envoi du feedback.')
  })
})
