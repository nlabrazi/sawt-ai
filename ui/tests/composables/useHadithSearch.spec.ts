import { $fetch } from 'ofetch'
import { effectScope } from 'vue'
import { useHadithSearch } from '~/composables/useHadithSearch'

vi.mock('ofetch', () => ({ $fetch: vi.fn() }))

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}
const result = { query: 'la colère', results: [] }

describe('useHadithSearch', () => {
  beforeEach(() => vi.mocked($fetch).mockReset())
  it('posts the trimmed query and default limit without altering the draft', async () => {
    vi.mocked($fetch).mockResolvedValueOnce(result)
    const state = useHadithSearch()
    state.query.value = '  la colère  '
    await state.search()
    expect($fetch).toHaveBeenCalledWith(
      'http://localhost:8000/hadith/search',
      expect.objectContaining({
        method: 'POST',
        body: { query: 'la colère', limit: 3 },
        retry: 0,
        timeout: 60_000,
      }),
    )
    expect(state.query.value).toBe('  la colère  ')
    expect(state.response.value).toEqual(result)
    expect(state.loading.value).toBe(false)
  })
  it.each([
    '',
    '  ',
    'ab',
    'x'.repeat(301),
  ])('rejects invalid input before calling the API', async (query) => {
    const state = useHadithSearch()
    state.query.value = query
    await state.search()
    expect(state.validationError.value).toBeTruthy()
    expect($fetch).not.toHaveBeenCalled()
  })
  it('ignores repeated submission while the same query is pending', async () => {
    const pending = deferred<typeof result>()
    vi.mocked($fetch).mockReturnValueOnce(pending.promise)
    const state = useHadithSearch()
    state.query.value = 'la colère'
    const request = state.search()
    await state.search()
    expect($fetch).toHaveBeenCalledTimes(1)
    expect(state.loading.value).toBe(true)
    pending.resolve(result)
    await request
  })
  it('aborts superseded requests and ignores responses arriving out of order', async () => {
    const first = deferred<typeof result>()
    const second = deferred<typeof result>()
    vi.mocked($fetch).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)
    const state = useHadithSearch()
    state.query.value = 'la colère'
    const oldRequest = state.search()
    const signal = vi.mocked($fetch).mock.calls[0]?.[1]?.signal as AbortSignal
    state.query.value = 'les intentions'
    const newRequest = state.search()
    expect(signal.aborted).toBe(true)
    first.resolve(result)
    await oldRequest
    expect(state.loading.value).toBe(true)
    expect(state.response.value).toBeNull()
    second.resolve({ query: 'les intentions', results: [] })
    await newRequest
    expect(state.response.value?.query).toBe('les intentions')
  })
  it('cancels immediately without exposing late errors', async () => {
    const pending = deferred<typeof result>()
    vi.mocked($fetch).mockReturnValueOnce(pending.promise)
    const state = useHadithSearch()
    state.query.value = 'la colère'
    const request = state.search()
    state.cancel()
    expect(state.loading.value).toBe(false)
    pending.reject(new Error('aborted'))
    await request
    expect(state.error.value).toBeNull()
  })
  it('resets the draft, results, errors and any pending request', async () => {
    const state = useHadithSearch()
    vi.mocked($fetch).mockResolvedValueOnce(result)
    state.query.value = 'la colère'
    await state.search()
    state.reset()
    expect(state.query.value).toBe('')
    expect(state.response.value).toBeNull()

    const pending = deferred<typeof result>()
    vi.mocked($fetch).mockReturnValueOnce(pending.promise)
    state.query.value = 'la colère'
    const request = state.search()
    const signal = vi.mocked($fetch).mock.calls[1]?.[1]?.signal as AbortSignal
    state.error.value = 'Ancienne erreur'
    state.validationError.value = 'Ancienne validation'
    state.reset()
    expect(signal.aborted).toBe(true)
    expect(state.query.value).toBe('')
    expect(state.loading.value).toBe(false)
    expect(state.error.value).toBeNull()
    expect(state.validationError.value).toBeNull()
    pending.resolve(result)
    await request
    expect(state.response.value).toBeNull()
  })
  it.each([
    503,
    422,
    undefined,
  ])('exposes a recoverable message for error %s', async (statusCode) => {
    vi.mocked($fetch).mockRejectedValueOnce({ statusCode })
    const state = useHadithSearch()
    state.query.value = 'la colère'
    await state.search()
    expect(state.error.value).toBeTruthy()
    expect(state.response.value).toBeNull()
    vi.mocked($fetch).mockResolvedValueOnce(result)
    await state.search()
    expect(state.error.value).toBeNull()
    expect(state.response.value).toEqual(result)
  })
  it('aborts when its Vue scope is disposed', async () => {
    const pending = deferred<typeof result>()
    vi.mocked($fetch).mockReturnValueOnce(pending.promise)
    const scope = effectScope()
    const state = scope.run(useHadithSearch)
    if (!state) throw new Error('Search scope was not created')
    state.query.value = 'la colère'
    const request = state.search()
    const signal = vi.mocked($fetch).mock.calls[0]?.[1]?.signal as AbortSignal
    scope.stop()
    expect(signal.aborted).toBe(true)
    pending.resolve(result)
    await request
    expect(state.response.value).toBeNull()
  })
})

describe('useHadithSearch voice queries', () => {
  const file = new File(['audio'], 'query.webm', { type: 'audio/webm' })
  const query = 'Trouve-moi les hadiths qui parlent du mariage'
  beforeEach(() => vi.mocked($fetch).mockReset())

  it('transcribes multipart audio then searches the visible editable query', async () => {
    const pendingSearch = deferred<typeof result>()
    vi.mocked($fetch).mockResolvedValueOnce({ query }).mockReturnValueOnce(pendingSearch.promise)
    const state = useHadithSearch()
    const request = state.transcribeAndSearch(file)
    expect(state.transcribing.value).toBe(true)
    expect(state.loading.value).toBe(true)
    const call = vi.mocked($fetch).mock.calls[0]
    if (!call) throw new Error('Missing transcription request')
    const [url, options] = call
    expect(url).toBe('http://localhost:8000/hadith/transcribe')
    expect((options?.body as FormData).get('file')).toBeInstanceOf(File)
    expect(options).toMatchObject({ method: 'POST', retry: 0, timeout: 60_000 })
    await Promise.resolve()
    expect(state.query.value).toBe(query)
    expect(state.transcribing.value).toBe(false)
    expect(state.loading.value).toBe(true)
    expect($fetch).toHaveBeenLastCalledWith(
      'http://localhost:8000/hadith/search',
      expect.objectContaining({ body: { query, limit: 3 } }),
    )
    pendingSearch.resolve({ query, results: [] })
    await request
    expect(state.response.value?.query).toBe(query)
    expect(state.loading.value).toBe(false)
    state.query.value = 'le divorce'
    vi.mocked($fetch).mockResolvedValueOnce({ query: 'le divorce', results: [] })
    await state.search()
    expect(state.response.value?.query).toBe('le divorce')
  })

  it('ignores a second voice submission while transcribing', async () => {
    const pending = deferred<{ query: string }>()
    vi.mocked($fetch).mockReturnValueOnce(pending.promise)
    const state = useHadithSearch()
    const request = state.transcribeAndSearch(file)
    await state.transcribeAndSearch(file)
    expect($fetch).toHaveBeenCalledTimes(1)
    state.cancel()
    pending.resolve({ query })
    await request
  })

  it.each([
    'cancel',
    'reset',
  ] as const)('discards a late transcription after %s', async (action) => {
    const pending = deferred<{ query: string }>()
    vi.mocked($fetch).mockReturnValueOnce(pending.promise)
    const state = useHadithSearch()
    state.query.value = 'ancien brouillon'
    const request = state.transcribeAndSearch(file)
    const signal = vi.mocked($fetch).mock.calls[0]?.[1]?.signal as AbortSignal
    state[action]()
    expect(signal.aborted).toBe(true)
    expect(state.transcribing.value).toBe(false)
    pending.resolve({ query })
    await request
    expect(state.query.value).toBe(action === 'reset' ? '' : 'ancien brouillon')
    expect($fetch).toHaveBeenCalledTimes(1)
    expect(state.response.value).toBeNull()
  })

  it('discards transcription superseded by a typed search', async () => {
    const pending = deferred<{ query: string }>()
    vi.mocked($fetch).mockReturnValueOnce(pending.promise).mockResolvedValueOnce(result)
    const state = useHadithSearch()
    const voice = state.transcribeAndSearch(file)
    state.query.value = result.query
    await state.search()
    pending.resolve({ query })
    await voice
    expect(state.query.value).toBe(result.query)
    expect(state.response.value).toEqual(result)
    expect($fetch).toHaveBeenCalledTimes(2)
  })

  it.each([
    400,
    413,
    415,
    422,
    503,
    undefined,
  ])('keeps the draft and allows typing after voice error %s', async (statusCode) => {
    vi.mocked($fetch).mockRejectedValueOnce({ statusCode })
    const state = useHadithSearch()
    state.query.value = result.query
    await state.transcribeAndSearch(file)
    expect(state.transcriptionError.value).toBeTruthy()
    expect(state.error.value).toBeNull()
    expect(state.query.value).toBe(result.query)
    expect(state.loading.value).toBe(false)
    expect(state.transcribing.value).toBe(false)
    expect($fetch).toHaveBeenCalledTimes(1)
    vi.mocked($fetch).mockResolvedValueOnce(result)
    await state.search()
    expect(state.transcriptionError.value).toBeNull()
    expect(state.response.value).toEqual(result)
  })

  it('retains the transcription when hadith retrieval fails', async () => {
    vi.mocked($fetch).mockResolvedValueOnce({ query }).mockRejectedValueOnce({ statusCode: 503 })
    const state = useHadithSearch()
    await state.transcribeAndSearch(file)
    expect(state.query.value).toBe(query)
    expect(state.error.value).toContain('temporairement indisponible')
    expect(state.transcriptionError.value).toBeNull()
    expect(state.loading.value).toBe(false)
  })
})
