import { $fetch } from 'ofetch'
import { effectScope } from 'vue'
import { useTafsirReview } from '~/composables/useTafsirReview'
import { tafsirFixture } from '../fixtures/tafsir'

vi.mock('ofetch', () => ({ $fetch: vi.fn() }))
const password = 'fictitious-dedicated-password'

beforeEach(() => vi.mocked($fetch).mockReset())

it('requires login and sends the password only in the private request header', async () => {
  const review = useTafsirReview()
  const query = { status: 'need_review' as const, limit: 50, offset: 0 }
  await review.list(query)
  expect($fetch).not.toHaveBeenCalled()
  vi.mocked($fetch).mockResolvedValueOnce(undefined).mockResolvedValueOnce([tafsirFixture()])
  expect(await review.login(password)).toBe(true)
  expect(await review.list(query)).toEqual([tafsirFixture()])
  expect($fetch).toHaveBeenLastCalledWith(
    'http://localhost:8000/internal/tafsir',
    expect.objectContaining({
      headers: { Authorization: `Bearer ${password}` },
      query,
      cache: 'no-store',
      retry: 0,
    }),
  )
  review.logout()
  await review.list(query)
  expect($fetch).toHaveBeenCalledTimes(2)
  expect(review.authenticated.value).toBe(false)
})

it('saves and verifies the exact source and revision, blocking validation of stale content', async () => {
  const review = useTafsirReview()
  const entry = tafsirFixture('as_saadi')
  const saved = { ...entry, text_fr: 'Correction fictive.', updated_at: '2026-10-07T10:01:00Z' }
  vi.mocked($fetch)
    .mockResolvedValueOnce(undefined)
    .mockResolvedValueOnce(saved)
    .mockRejectedValueOnce({ statusCode: 409 })
  await review.login(password)
  await review.save(entry, saved.text_fr)
  expect($fetch).toHaveBeenLastCalledWith(
    'http://localhost:8000/internal/tafsir/2/255/as_saadi',
    expect.objectContaining({
      method: 'PATCH',
      body: { text_fr: saved.text_fr, expected_updated_at: entry.updated_at },
    }),
  )
  await review.verify(saved)
  expect($fetch).toHaveBeenLastCalledWith(
    'http://localhost:8000/internal/tafsir/2/255/as_saadi/verify',
    expect.objectContaining({
      method: 'POST',
      body: { expected_updated_at: saved.updated_at },
    }),
  )
  expect(review.conflict.value).toBe(true)
  await review.verify(saved)
  await review.save(saved, 'Autre correction fictive.')
  expect($fetch).toHaveBeenCalledTimes(3)
  vi.mocked($fetch).mockResolvedValueOnce([saved])
  await review.list({ status: 'need_review', limit: 50, offset: 0 })
  expect(review.conflict.value).toBe(false)
})

it('revokes local access on 401 and never logs a fetch error containing credentials', async () => {
  const log = vi.spyOn(console, 'error').mockImplementation(() => {})
  const review = useTafsirReview()
  vi.mocked($fetch)
    .mockResolvedValueOnce(undefined)
    .mockRejectedValueOnce({ statusCode: 401, secret: password })
  await review.login(password)
  await review.list({ status: 'need_review', limit: 50, offset: 0 })
  expect(review.authenticated.value).toBe(false)
  expect(review.error.value).toContain('Reconnectez-vous')
  expect(log).not.toHaveBeenCalled()
  log.mockRestore()
})

it('clears credentials and aborts pending requests when the review screen is disposed', async () => {
  const scope = effectScope()
  const review = scope.run(useTafsirReview)
  if (!review) throw new Error('Expected an active review scope')
  vi.mocked($fetch).mockResolvedValueOnce(undefined)
  await review.login(password)
  let resolve!: (value: unknown) => void
  vi.mocked($fetch).mockReturnValueOnce(
    new Promise((done) => {
      resolve = done
    }),
  )
  const pending = review.list({ status: 'need_review', limit: 50, offset: 0 })
  const signal = vi.mocked($fetch).mock.calls[1]?.[1]?.signal as AbortSignal
  scope.stop()
  expect(signal.aborted).toBe(true)
  resolve([tafsirFixture()])
  expect(await pending).toBeUndefined()
  expect(review.authenticated.value).toBe(false)
})
