import { $fetch } from 'ofetch'
import { useRecognition } from '~/composables/useRecognition'

vi.mock('ofetch', () => ({
  $fetch: vi.fn(),
}))

describe('useRecognition', () => {
  it('stores the API result after the loading sequence', async () => {
    vi.useFakeTimers()
    vi.mocked($fetch).mockResolvedValueOnce({
      transcription_text: 'قل هو الله احد',
      verse: {
        sourate_id: 112,
        sourate_name: 'Al-Ikhlas',
        transliteration: 'Al-Ikhlas',
        start_verse: 1,
        end_verse: 4,
        text: 'قل هو الله احد',
        similarity: 0.92,
      },
      imam_predictions: [],
      imam_status: 'unknown',
      imam_detection_enabled: true,
    })

    const { loading, loadingStep, error, result, recognizeAudio } = useRecognition()
    const audioFile = new File(['audio'], 'recitation.webm', { type: 'audio/webm' })

    const pending = recognizeAudio(audioFile, true)

    await vi.runAllTimersAsync()
    await pending

    expect($fetch).toHaveBeenCalledWith('http://localhost:8000/recognize', {
      method: 'POST',
      body: expect.any(FormData),
    })
    expect(loading.value).toBe(false)
    expect(loadingStep.value).toBe('result-found')
    expect(error.value).toBeNull()
    expect(result.value?.transcription_text).toBe('قل هو الله احد')
  })

  it('sets an error when the backend request fails', async () => {
    vi.mocked($fetch).mockRejectedValueOnce(new Error('boom'))

    const { loading, error, result, recognizeAudio } = useRecognition()
    const audioFile = new File(['audio'], 'recitation.webm', { type: 'audio/webm' })

    await recognizeAudio(audioFile)

    expect(loading.value).toBe(false)
    expect(error.value).toBe('Erreur pendant la reconnaissance audio.')
    expect(result.value).toBeNull()
  })
})
