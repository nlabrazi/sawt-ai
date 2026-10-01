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
    const state = scope.run(useHadithSearch)!
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
