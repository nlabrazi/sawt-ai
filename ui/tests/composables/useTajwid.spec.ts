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
})
