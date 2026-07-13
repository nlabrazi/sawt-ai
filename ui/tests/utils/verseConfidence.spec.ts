import { isVerseConfident } from '~/utils/verseConfidence'

describe('isVerseConfident', () => {
  it('supports similarity ratios between 0 and 1', () => {
    expect(isVerseConfident(0.92, 0.8)).toBe(true)
    expect(isVerseConfident(0.79, 0.8)).toBe(false)
  })

  it('supports legacy similarity scores over 1', () => {
    expect(isVerseConfident(92, 0.8)).toBe(true)
    expect(isVerseConfident(79, 0.8)).toBe(false)
  })

  it('uses the threshold supplied by the API policy', () => {
    expect(isVerseConfident(0.81, 0.82)).toBe(false)
    expect(isVerseConfident(0.82, 0.82)).toBe(true)
  })
})
