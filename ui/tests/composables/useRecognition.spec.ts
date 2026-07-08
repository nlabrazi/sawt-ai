import { $fetch } from 'ofetch'
import { useRecognition } from '~/composables/useRecognition'

vi.mock('ofetch', () => ({
  $fetch: vi.fn(),
}))

function createAbortError() {
  const error = new Error('The operation was aborted.')
  error.name = 'AbortError'
  return error
}

const successfulResponse = {
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
}

describe('useRecognition', () => {
  it('stores the API result after the loading sequence', async () => {
    vi.useFakeTimers()
    vi.mocked($fetch).mockResolvedValueOnce(successfulResponse)

    const { loading, loadingStep, error, result, recognizeAudio } = useRecognition()
    const audioFile = new File(['audio'], 'recitation.webm', { type: 'audio/webm' })

    const pending = recognizeAudio(audioFile, true)

    await vi.runAllTimersAsync()
    await pending

    expect($fetch).toHaveBeenCalledWith('http://localhost:8000/recognize', {
      method: 'POST',
      body: expect.any(FormData),
      signal: expect.any(AbortSignal),
    })
    expect(loading.value).toBe(false)
    expect(loadingStep.value).toBe('done')
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

  it('surfaces the backend detail when the API returns one', async () => {
    vi.mocked($fetch).mockRejectedValueOnce({
      data: {
        detail: 'Format audio invalide ou non pris en charge.',
      },
    })

    const { loading, error, result, recognizeAudio } = useRecognition()
    const audioFile = new File(['audio'], 'recitation.webm', { type: 'audio/webm' })

    await recognizeAudio(audioFile)

    expect(loading.value).toBe(false)
    expect(error.value).toBe('Format audio invalide ou non pris en charge.')
    expect(result.value).toBeNull()
  })

  it('aborts the in-flight request when reset is called', async () => {
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

    vi.mocked($fetch).mockImplementationOnce((_url, options) => {
      const signal = options?.signal as AbortSignal

      return new Promise((_, reject) => {
        signal.addEventListener('abort', () => reject(createAbortError()), { once: true })
      })
    })

    const { loading, loadingStep, error, result, recognizeAudio, reset } = useRecognition()
    const audioFile = new File(['audio'], 'recitation.webm', { type: 'audio/webm' })

    const pending = recognizeAudio(audioFile)

    expect(loading.value).toBe(true)

    reset()
    await pending

    expect(loading.value).toBe(false)
    expect(loadingStep.value).toBe('transcribing')
    expect(error.value).toBeNull()
    expect(result.value).toBeNull()
    expect(consoleErrorSpy).not.toHaveBeenCalled()
  })

  it('ignores a late response sequence after reset', async () => {
    vi.useFakeTimers()
    vi.mocked($fetch).mockResolvedValueOnce(successfulResponse)

    const { loading, loadingStep, error, result, recognizeAudio, reset } = useRecognition()
    const audioFile = new File(['audio'], 'recitation.webm', { type: 'audio/webm' })

    const pending = recognizeAudio(audioFile)

    await Promise.resolve()
    reset()
    await vi.runAllTimersAsync()
    await pending

    expect(loading.value).toBe(false)
    expect(loadingStep.value).toBe('transcribing')
    expect(error.value).toBeNull()
    expect(result.value).toBeNull()
  })
})
