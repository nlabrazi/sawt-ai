import { $fetch } from 'ofetch'
import { effectScope } from 'vue'
import { useQuranContent } from '~/composables/useQuranContent'
import { quranContentFixture } from '../fixtures/quran-content'

vi.mock('ofetch', () => ({ $fetch: vi.fn() }))

describe('useQuranContent', () => {
  beforeEach(() => {
    vi.mocked($fetch).mockReset()
  })

  it('requests the exact passage without caching and observes withdrawal on the next read', async () => {
    const firstResponse = quranContentFixture(2, 255, 255)
    vi.mocked($fetch)
      .mockResolvedValueOnce(firstResponse)
      .mockResolvedValueOnce({
        ...firstResponse,
        ayahs: [{ ...firstResponse.ayahs[0], tafsirs: [] }],
      })
    const { content, fetchContent } = useQuranContent()
    await fetchContent(2, 255, 255)
    expect($fetch).toHaveBeenCalledWith('http://localhost:8000/quran/content', {
      method: 'GET',
      query: { surah_id: 2, start_verse: 255, end_verse: 255 },
      signal: expect.any(AbortSignal),
      cache: 'no-store',
      retry: 0,
      timeout: 20000,
    })
    expect(content.value?.ayahs[0]?.tafsirs.map((entry) => entry.source)).toEqual([
      'ibn_kathir',
      'as_saadi',
    ])
    await fetchContent(2, 255, 255)
    expect($fetch).toHaveBeenCalledTimes(2)
    expect(content.value?.ayahs[0]?.tafsirs).toEqual([])
  })

  it('excludes pending, unreviewed, unknown-source and misplaced entries from public state', async () => {
    const response = quranContentFixture()
    const entry = response.ayahs[0]
    const verified = entry?.tafsirs[0]
    vi.mocked($fetch).mockResolvedValue({
      ...response,
      ayahs: [
        {
          ...entry,
          tafsirs: [
            verified,
            { ...verified, status: 'need_review', text_fr: 'Brouillon fictif privé' },
            { ...verified, ayah: 2 },
            { ...verified, surah_id: 2 },
            { ...verified, source: 'other' },
            { ...verified, reviewed_at: null },
          ],
        },
        { ...response.ayahs[1], translation: { ...response.ayahs[1]?.translation, ayah: 1 } },
      ],
    })
    const { content, fetchContent } = useQuranContent()
    await fetchContent(1, 1, 2)
    expect(content.value?.ayahs[0]?.tafsirs).toEqual([verified])
    expect(content.value?.ayahs[1]?.translation).toBeNull()
    expect(JSON.stringify(content.value)).not.toContain('Brouillon fictif privé')
    expect(content.value?.ayahs[1]?.tafsirs[0]?.ayah).toBe(2)
  })

  it('keeps translations but discards tafsirs when their storage is unavailable', async () => {
    const response = quranContentFixture()
    vi.mocked($fetch).mockResolvedValue({ ...response, tafsir_status: 'unavailable' })
    const { content, fetchContent, error } = useQuranContent()
    await fetchContent(1, 1, 2)
    expect(error.value).toBeNull()
    expect(content.value?.ayahs[0]?.translation).toEqual(response.ayahs[0]?.translation)
    expect(content.value?.ayahs.every((entry) => !entry.tafsirs.length)).toBe(true)
  })

  it.each([
    'wrong range',
    'wrong ayah',
    'http failure',
  ])('handles %s without exposing unrelated content or server details', async (failure) => {
    const response = quranContentFixture()
    if (failure === 'http failure') {
      vi.mocked($fetch).mockRejectedValue(new Error('Private provider details'))
    } else {
      vi.mocked($fetch).mockResolvedValue(
        failure === 'wrong range'
          ? { ...response, surah_id: 2 }
          : { ...response, ayahs: response.ayahs.toReversed() },
      )
    }
    const { content, fetchContent, error, loading } = useQuranContent()
    await fetchContent(1, 1, 2)
    expect(content.value).toBeNull()
    expect(error.value).toBe('Impossible de charger le contenu français. Réessayez.')
    expect(loading.value).toBe(false)
  })

  it('cancels the previous passage and ignores late responses while the new one loads', async () => {
    let resolvePrevious: (value: unknown) => void = () => {}
    let resolveCurrent: (value: unknown) => void = () => {}
    vi.mocked($fetch)
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolvePrevious = resolve
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveCurrent = resolve
          }),
      )
    const { content, fetchContent, loading } = useQuranContent()
    const previous = fetchContent(1, 1, 2)
    const previousSignal = vi.mocked($fetch).mock.calls[0]?.[1]?.signal
    const current = fetchContent(2, 255, 255)
    expect(previousSignal?.aborted).toBe(true)
    resolvePrevious(quranContentFixture())
    await previous
    expect(content.value).toBeNull()
    expect(loading.value).toBe(true)
    resolveCurrent(quranContentFixture(2, 255, 255))
    await current
    expect(content.value?.surah_id).toBe(2)
    expect(loading.value).toBe(false)
  })

  it('aborts and clears content on component disposal even if the request resolves later', async () => {
    let resolveResponse: (value: unknown) => void = () => {}
    vi.mocked($fetch).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveResponse = resolve
        }),
    )
    const scope = effectScope()
    const reader = scope.run(() => useQuranContent())
    if (!reader) throw new Error('Missing reader')
    const request = reader.fetchContent(1, 1, 2)
    const signal = vi.mocked($fetch).mock.calls[0]?.[1]?.signal
    scope.stop()
    expect(signal?.aborted).toBe(true)
    resolveResponse(quranContentFixture())
    await request
    expect(reader.content.value).toBeNull()
    expect(reader.error.value).toBeNull()
    expect(reader.loading.value).toBe(false)
  })
})
