import { isVerseConfident } from '~/utils/verseConfidence'

describe('isVerseConfident', () => {
  it('supports similarity ratios between 0 and 1', () => {
    expect(isVerseConfident(0.92)).toBe(true)
    expect(isVerseConfident(0.79)).toBe(false)
  })

  it('supports legacy similarity scores over 1', () => {
    expect(isVerseConfident(92)).toBe(true)
    expect(isVerseConfident(79)).toBe(false)
  })
})
