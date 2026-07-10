import { $fetch } from 'ofetch'
import { clearTajwidCache, useTajwid } from '~/composables/useTajwid'

vi.mock('ofetch', () => ({
  $fetch: vi.fn(),
}))

describe('useTajwid', () => {
  beforeEach(() => {
    clearTajwidCache()
  })

  it('caches repeated tajwid requests', async () => {
    vi.mocked($fetch).mockResolvedValue({
      surah_id: 112,
      start_verse: 1,
      end_verse: 4,
      text: 'tajwid text',
      ayahs: [{ number: 1, tajwid_text: 'tajwid text' }],
    })

    const { fetchTajwid } = useTajwid()

    const first = await fetchTajwid(112, 1, 4)
    const second = await fetchTajwid(112, 1, 4)

    expect(first).toEqual(second)
    expect($fetch).toHaveBeenCalledTimes(1)
  })

  it('stores a user-facing error when the request fails', async () => {
    vi.mocked($fetch).mockRejectedValueOnce(new Error('upstream error'))

    const { error, fetchTajwid } = useTajwid()

    await expect(fetchTajwid(112, 1, 4)).rejects.toThrow('upstream error')
    expect(error.value).toBe('Impossible de charger le tajwid.')
  })

  it('evicts the least recently used entry when the cache limit is reached', async () => {
    vi.mocked($fetch).mockImplementation(async (_url, options) => {
      const query = options?.query as {
        surah_id: number
        start_verse: number
        end_verse: number
      }

      return {
        surah_id: query.surah_id,
        start_verse: query.start_verse,
        end_verse: query.end_verse,
        text: `tajwid-${query.surah_id}`,
        ayahs: [{ number: 1, tajwid_text: `tajwid-${query.surah_id}` }],
      }
    })

    const { fetchTajwid, getCachedTajwid } = useTajwid()

    for (let index = 1; index <= 32; index += 1) {
      await fetchTajwid(index, 1, 1)
    }

    await fetchTajwid(1, 1, 1)
    await fetchTajwid(33, 1, 1)

    expect(getCachedTajwid(1, 1, 1)?.text).toBe('tajwid-1')
    expect(getCachedTajwid(2, 1, 1)).toBeNull()
    expect(getCachedTajwid(33, 1, 1)?.text).toBe('tajwid-33')
  })
})
